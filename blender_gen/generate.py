"""Procedurally render a labelled synthetic dataset for 6-DoF pose estimation.

Runs either through Blender's own interpreter or against the ``bpy`` pip module::

    blender --background --python blender_gen/generate.py -- --n 5000 --out data/synthetic
    .venv/bin/python blender_gen/generate.py --n 5000 --out data/synthetic

Both forms are supported deliberately: the requirement is that generation be
runnable headless and fully decoupled from the training code, and the pip module
keeps the whole pipeline installable without a GUI Blender on the machine.

Layout produced under ``--out``::

    images/{split}/rgb_000000.png      256x256 RGB render
    depth/{split}/depth_000000.npz     float16 metric depth, 0 = background
    labels/{split}/meta_000000.json    intrinsics, 6-DoF pose, material, seed
    dataset.json                       manifest: config, splits, per-class counts
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpy  # noqa: E402

import assets  # noqa: E402
import geometry as geo  # noqa: E402
import scene_builder as sb  # noqa: E402


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    """Parse args, honouring Blender's ``--`` separator when present."""
    argv = sys.argv[1:] if argv is None else argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    elif Path(sys.argv[0]).stem == "blender":
        argv = []

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, default=Path("data/synthetic"))
    p.add_argument("--n", type=int, default=5000, help="total samples across all splits")
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--engine", choices=("eevee", "cycles"), default="eevee")
    p.add_argument("--samples", type=int, default=32, help="render samples per image")
    p.add_argument("--categories", nargs="+", default=list(assets.CATEGORIES))
    p.add_argument("--splits", nargs=3, type=float, default=(0.8, 0.1, 0.1),
                   metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--dist", nargs=2, type=float, default=(2.8, 5.0))
    p.add_argument("--elev", nargs=2, type=float, default=(-25.0, 75.0))
    p.add_argument("--no-depth", action="store_true", help="skip depth maps (smaller dataset)")
    p.add_argument("--start", type=int, default=0, help="resume from this sample index")
    return p.parse_args(argv)


@contextlib.contextmanager
def quiet():
    """Silence Blender's per-frame render chatter on fd 1.

    Blender writes progress straight to the C-level stdout, so redirecting
    ``sys.stdout`` is not enough -- at 5,000 frames the log is otherwise unusable.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(devnull)
        os.close(saved)


def split_for(index, n, fracs):
    """Assign a sample to train/val/test by contiguous index ranges."""
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    if index < n_train:
        return "train"
    return "val" if index < n_train + n_val else "test"


# --------------------------------------------------------------------------
# Per-sample pose sampling
# --------------------------------------------------------------------------
def sample_pose(rng, K, bbox_local, args, max_tries=25):
    """Sample a camera pose and object rotation that keep the object framed.

    Rejection-samples rather than clamping: a clamped viewpoint distribution has
    spikes at the boundaries, which show up later as the model being oddly
    confident about a handful of orientations.
    """
    res = args.res
    for _ in range(max_tries):
        eye = geo.sample_viewpoint(rng, args.elev, (0.0, 360.0), args.dist)
        target = rng.normal(0.0, 0.25, size=3)
        cam_to_world = geo.look_at(eye, target, roll=rng.uniform(-np.pi, np.pi))

        world_to_cv = geo.world_to_cv_camera(cam_to_world)
        R_obj = geo.random_rotation(rng)
        scale = float(rng.uniform(0.8, 1.4))

        R = world_to_cv[:3, :3] @ R_obj
        t = world_to_cv[:3, 3]                       # object sits at the world origin

        uv, z = geo.project(K, R, t, scale, bbox_local)
        if np.any(z <= 0.1):
            continue
        centre, _ = geo.project(K, R, t, scale, np.zeros((1, 3)))
        if not np.all((centre[0] > 0.15 * res) & (centre[0] < 0.85 * res)):
            continue
        inside = np.sum((uv > -0.1 * res).all(1) & (uv < 1.1 * res).all(1))
        if inside >= 6:
            return cam_to_world, world_to_cv, R_obj, R, t, scale
    return None


# --------------------------------------------------------------------------
# Depth
# --------------------------------------------------------------------------
def read_depth_exr(path: Path) -> np.ndarray:
    """Load Blender's multilayer depth EXR as float16 metres, 0 = background."""
    import OpenEXR

    with OpenEXR.File(str(path)) as f:
        channels = f.parts[0].channels
        key = "depth.V" if "depth.V" in channels else next(iter(channels))
        depth = np.asarray(channels[key].pixels, dtype=np.float32)

    depth[depth >= sb.DEPTH_BACKGROUND * 0.5] = 0.0     # infinity sentinel -> background
    return depth.astype(np.float16)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    out = args.out.resolve()
    engine = "CYCLES" if args.engine == "cycles" else "BLENDER_EEVEE"

    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        if not args.no_depth:
            (out / "depth" / split).mkdir(parents=True, exist_ok=True)
    tmp_dir = out / "_tmp_depth"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    scene = sb.configure_render(args.res, engine, args.samples)
    counts = {c: 0 for c in args.categories}
    rejected = 0
    t0 = time.time()

    for idx in range(args.start, args.n):
        rng = np.random.default_rng(args.seed * 1_000_003 + idx)
        split = split_for(idx, args.n, args.splits)
        category = args.categories[int(rng.integers(len(args.categories)))]

        sb.reset_scene()
        obj = assets.build(category, rng)
        material = sb.apply_material(obj, rng)
        sb.randomise_lighting(rng)
        cam = sb.add_camera(rng)

        K = geo.intrinsics(cam.data.lens, cam.data.sensor_width,
                           args.res, args.res, cam.data.sensor_fit)
        bbox_local = np.array([list(c) for c in obj.bound_box], dtype=float)

        sampled = sample_pose(rng, K, bbox_local, args)
        if sampled is None:
            rejected += 1
            continue
        cam_to_world, world_to_cv, R_obj, R, t, scale = sampled

        sb.place_camera(cam, cam_to_world, geo)
        obj.rotation_quaternion = geo.quat_from_matrix(R_obj).tolist()
        obj.scale = (scale,) * 3
        bpy.context.view_layer.update()

        # Cross-check our analytic pose against Blender's own object matrix. If a
        # convention ever drifts, this fails here rather than silently poisoning labels.
        M_obj = np.array(obj.matrix_world).reshape(4, 4)
        expected = world_to_cv @ M_obj
        assert np.allclose(expected[:3, :3], scale * R, atol=1e-4), "rotation mismatch"
        assert np.allclose(expected[:3, 3], t, atol=1e-4), "translation mismatch"

        stem = f"{idx:06d}"
        rgb_path = out / "images" / split / f"rgb_{stem}"
        scene.render.filepath = str(rgb_path)

        depth_node = None
        if not args.no_depth:
            depth_node = sb.setup_depth_output(tmp_dir)
            depth_node.file_name = f"d_{stem}"

        with quiet():
            bpy.ops.render.render(write_still=True)

        if depth_node is not None:
            exr = next(tmp_dir.glob(f"d_{stem}*.exr"), None)
            if exr is not None:
                np.savez_compressed(out / "depth" / split / f"depth_{stem}.npz",
                                    depth=read_depth_exr(exr))
                exr.unlink()

        meta = {
            "index": idx,
            "split": split,
            "category": category,
            "category_id": args.categories.index(category),
            "image": f"images/{split}/rgb_{stem}.png",
            "depth": None if args.no_depth else f"depth/{split}/depth_{stem}.npz",
            "resolution": [args.res, args.res],
            "seed": int(args.seed * 1_000_003 + idx),
            "camera": {
                "lens_mm": float(cam.data.lens),
                "sensor_mm": float(cam.data.sensor_width),
                "sensor_fit": cam.data.sensor_fit,
                "K": K.tolist(),
            },
            # Object -> OpenCV camera. X_cam = scale * (R @ X_local) + t
            "pose_cv": {
                "R": R.tolist(),
                "t": t.tolist(),
                "scale": scale,
                "quat_wxyz": geo.quat_from_matrix(R).tolist(),
                "rot6d": geo.matrix_to_rot6d(R).tolist(),
            },
            "extrinsics_cv": {
                "R_world_to_cam": world_to_cv[:3, :3].tolist(),
                "t_world_to_cam": world_to_cv[:3, 3].tolist(),
            },
            "object": {"bbox_local": bbox_local.tolist(), "R_obj_world": R_obj.tolist()},
            "material": material,
            "blender": {"cam_matrix_world": cam_to_world.tolist()},
        }
        (out / "labels" / split / f"meta_{stem}.json").write_text(json.dumps(meta))
        counts[category] += 1

        done = idx - args.start + 1
        if done % 25 == 0 or idx == args.n - 1:
            rate = done / (time.time() - t0)
            eta = (args.n - idx - 1) / max(rate, 1e-6)
            print(f"[{idx + 1}/{args.n}] {rate:.2f} img/s  ETA {eta / 60:.1f} min",
                  file=sys.stderr, flush=True)

    with contextlib.suppress(OSError):
        tmp_dir.rmdir()

    manifest = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "engine": engine,
        "n_rendered": sum(counts.values()),
        "n_rejected": rejected,
        "per_category": counts,
        "split_sizes": {s: sum(1 for i in range(args.n) if split_for(i, args.n, args.splits) == s)
                        for s in ("train", "val", "test")},
        "pose_convention": "OpenCV (+X right, +Y down, +Z forward); X_cam = scale*(R@X_local)+t",
        "depth_units": "metres along camera +Z; 0 = background",
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out / "dataset.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {manifest['n_rendered']} images, {rejected} rejected, "
          f"{manifest['elapsed_sec']}s -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
