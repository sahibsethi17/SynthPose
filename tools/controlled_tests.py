"""Controlled experiments that render their own inputs to isolate one variable at a time.

The held-out test split measures average accuracy but cannot answer *why* the model is
right. These render purpose-built scenes where the correct answer is known analytically:

``equivariance``  Re-render one fixed scene with only the camera's roll changed. The true
                  object rotation in the camera frame changes by exactly that roll, so a
                  model reasoning geometrically must track it. Flat error across roll means
                  the geometry generalises; error that spikes at unusual rolls means the
                  model leans on an upright prior. It also gives a *consistency* number --
                  whether predicted rotations differ by the known roll step -- which needs
                  no ground-truth labels and is the same protocol the photograph test uses.

``extrapolation`` Render beyond the generator's distance and scale ranges to find where
                  accuracy falls off outside the training distribution.

``occlusion``     Hide a growing fraction of the object. Degradation should be graceful;
                  a cliff would suggest reliance on one specific cue.

    .venv/bin/python tools/controlled_tests.py --test equivariance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "blender_gen"))

import bpy  # noqa: E402

import assets  # noqa: E402
import geometry as geo  # noqa: E402
import rotation as rot  # noqa: E402
import scene_builder as sb  # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model import PoseNet  # noqa: E402


@contextmanager
def quiet():
    sys.stdout.flush()
    saved, null = os.dup(1), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(null)
        os.close(saved)


def load_net(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net = PoseNet(ckpt.get("backbone", "resnet18"), pretrained=False,
                  dropout=float(ckpt.get("args", {}).get("dropout", 0.0))).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


@torch.no_grad()
def infer(net, path_png, device, size=224):
    img = Image.open(path_png).convert("RGB").resize((size, size), Image.BILINEAR)
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(arr.transpose(2, 0, 1).copy())[None].to(device)
    R, _ = net(x)
    return R[0].cpu().numpy()


def build_scene(cat, seed, res):
    rng = np.random.default_rng(seed)
    sb.reset_scene()
    obj = assets.build(cat, rng)
    sb.apply_material(obj, rng)
    sb.randomise_lighting(rng)
    cam = sb.add_camera(rng)
    return rng, obj, cam


def render(tmp, name):
    bpy.context.scene.render.filepath = str(tmp / name)
    with quiet():
        bpy.ops.render.render(write_still=True)
    return Path(str(tmp / name) + ".png")


def run_equivariance(net, device, tmp, args):
    """Vary ONLY camera roll; everything else held fixed."""
    rolls = np.arange(0, 360, 30, dtype=float)
    abs_err, consistency = [], []

    for s in range(args.scenes):
        rng, obj, cam = build_scene(args.categories[s % len(args.categories)], 500 + s, args.res)
        eye = geo.sample_viewpoint(rng, (-15.0, 55.0), (0.0, 360.0), (3.4, 4.2))
        R_obj = geo.random_rotation(rng)
        obj.rotation_quaternion = geo.quat_from_matrix(R_obj).tolist()
        obj.scale = (1.1,) * 3

        preds, gts = [], []
        for roll in rolls:
            cam_to_world = geo.look_at(eye, roll=np.radians(roll))
            sb.place_camera(cam, cam_to_world, geo)
            K_world_to_cv = geo.world_to_cv_camera(cam_to_world)
            R_gt = K_world_to_cv[:3, :3] @ R_obj
            R_pred = infer(net, render(tmp, "eq"), device)
            preds.append(R_pred)
            gts.append(R_gt)
            abs_err.append(geo.geodesic_error_deg(R_pred, R_gt))

        # Consistency: does the prediction rotate by the roll step we actually applied?
        for i in range(len(rolls) - 1):
            pred_delta = geo.geodesic_error_deg(preds[i], preds[i + 1])
            true_delta = geo.geodesic_error_deg(gts[i], gts[i + 1])
            consistency.append(abs(pred_delta - true_delta))

    abs_err = np.array(abs_err).reshape(args.scenes, len(rolls))
    print(f"\n  absolute error by camera roll ({args.scenes} scenes x {len(rolls)} rolls):")
    print(f"    {'roll':>6s} {'mean deg':>9s} {'median':>8s}")
    per_roll = []
    for j, roll in enumerate(rolls):
        col = abs_err[:, j]
        per_roll.append({"roll": float(roll), "mean": float(col.mean())})
        print(f"    {roll:5.0f}d {col.mean():9.2f} {np.median(col):8.2f}")
    c = np.array(consistency)
    spread = abs_err.mean(0).max() - abs_err.mean(0).min()
    print(f"\n    spread across rolls: {spread:.2f} deg "
          f"(flat => roll-equivariant, no upright prior)")
    print(f"    relative-rotation consistency error: mean {c.mean():.2f} deg  "
          f"median {np.median(c):.2f} deg")
    return {"per_roll": per_roll, "spread_deg": float(spread),
            "consistency_mean_deg": float(c.mean()),
            "consistency_median_deg": float(np.median(c)),
            "overall_mean_deg": float(abs_err.mean())}


def run_extrapolation(net, device, tmp, args):
    """Push distance and scale outside the ranges the generator ever produced."""
    train_dist, train_scale = (2.8, 5.0), (0.8, 1.4)
    results = {}

    for label, values, kind in (
            ("distance_m", [2.0, 2.4, 2.8, 3.5, 4.2, 5.0, 5.8, 6.6, 7.5], "dist"),
            ("scale", [0.5, 0.65, 0.8, 1.1, 1.4, 1.7, 2.0], "scale")):
        print(f"\n  by {label} (training range "
              f"{train_dist if kind == 'dist' else train_scale}):")
        print(f"    {'value':>8s} {'in train?':>10s} {'mean deg':>9s} {'median':>8s}")
        rows = []
        for v in values:
            errs = []
            for s in range(args.scenes):
                rng, obj, cam = build_scene(args.categories[s % len(args.categories)],
                                            900 + s, args.res)
                R_obj = geo.random_rotation(rng)
                obj.rotation_quaternion = geo.quat_from_matrix(R_obj).tolist()
                obj.scale = ((v if kind == "scale" else 1.1),) * 3
                d = v if kind == "dist" else 3.9
                eye = geo.sample_viewpoint(rng, (-15.0, 55.0), (0.0, 360.0), (d, d))
                cam_to_world = geo.look_at(eye, roll=rng.uniform(-np.pi, np.pi))
                sb.place_camera(cam, cam_to_world, geo)
                R_gt = geo.world_to_cv_camera(cam_to_world)[:3, :3] @ R_obj
                errs.append(geo.geodesic_error_deg(infer(net, render(tmp, "ex"), device), R_gt))
            e = np.array(errs)
            lo, hi = train_dist if kind == "dist" else train_scale
            inside = "yes" if lo <= v <= hi else "NO"
            rows.append({"value": float(v), "in_train": inside == "yes",
                         "mean": float(e.mean()), "median": float(np.median(e))})
            print(f"    {v:8.2f} {inside:>10s} {e.mean():9.2f} {np.median(e):8.2f}")
        results[label] = rows
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test", choices=("equivariance", "extrapolation", "all"), default="all")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/improved/best.pt"))
    p.add_argument("--scenes", type=int, default=12)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--categories", nargs="+", default=list(assets.CATEGORIES))
    p.add_argument("--out", type=Path, default=Path("results/controlled_tests.json"))
    p.add_argument("--tmp", type=Path,
                   default=Path(os.environ.get("SCRATCHPAD", "/tmp")) / "controlled")
    args = p.parse_args()

    args.tmp.mkdir(parents=True, exist_ok=True)
    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_net(args.checkpoint, device)
    sb.configure_render(args.res, "BLENDER_EEVEE", 24)

    out = {}
    if args.test in ("equivariance", "all"):
        print("=== camera-roll equivariance ===")
        out["equivariance"] = run_equivariance(net, device, args.tmp, args)
    if args.test in ("extrapolation", "all"):
        print("\n=== extrapolation beyond training ranges ===")
        out["extrapolation"] = run_extrapolation(net, device, args.tmp, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
