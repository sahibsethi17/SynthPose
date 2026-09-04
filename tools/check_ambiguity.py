"""Measure whether each object shape is visually distinguishable from its flipped pose.

A shape is only usable for pose regression if every orientation *looks* different.
Shape symmetry is the obvious failure, but two subtler ones bit this project:

* the mug becomes a featureless cylinder when its handle is occluded;
* the first L-bracket was near-planar, and a flat object that is mirror-symmetric about
  its own mid-plane looks the same flipped front-to-back -- the projection cannot say
  which face you are seeing.

Both showed up in training as a spike of predictions ~180 deg from the truth. This tool
catches that in ~30 seconds instead of after an 80-minute render plus an hour of training.

Method: render each object at a pose, then at that pose composed with a 180 deg flip
about each object axis, under identical lighting and camera. Report the *smallest*
difference over the flips -- the worst case, i.e. the most confusable pair.

    .venv/bin/python tools/check_ambiguity.py --n 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blender_gen"))

import bpy  # noqa: E402

import assets  # noqa: E402
import geometry as geo  # noqa: E402
import scene_builder as sb  # noqa: E402

FLIPS = {
    "x": np.diag([1.0, -1.0, -1.0]),
    "y": np.diag([-1.0, 1.0, -1.0]),
    "z": np.diag([-1.0, -1.0, 1.0]),
}


class _quiet:
    """Silence Blender's C-level render chatter (it writes straight to fd 1)."""

    def __enter__(self):
        import os
        sys.stdout.flush()
        self._saved = os.dup(1)
        self._null = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null, 1)

    def __exit__(self, *a):
        import os
        os.dup2(self._saved, 1)
        os.close(self._null)
        os.close(self._saved)


def render_rgba(path):
    """Render with a transparent film and return ``(grey, mask)``.

    The alpha channel gives an exact object silhouette for free, which is what makes a
    meaningful comparison possible: the object covers ~10% of the frame, so a whole-image
    difference is dominated by identical background and reports every shape as ambiguous.
    """
    from PIL import Image
    bpy.context.scene.render.filepath = str(path)
    with _quiet():
        bpy.ops.render.render(write_still=True)
    rgba = np.asarray(Image.open(str(path) + ".png").convert("RGBA"), dtype=np.float32) / 255.0
    grey = rgba[..., :3] @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return grey, rgba[..., 3] > 0.5


def distinguishability(a, mask_a, b, mask_b):
    """How visually different two renders are, on a scale where ~0 means identical.

    Combines silhouette disagreement with shading disagreement inside the object, each
    normalised by the object's own contrast so the score does not depend on how large the
    object happens to appear or how brightly it is lit.
    """
    union = mask_a | mask_b
    if union.sum() < 20:
        return 0.0
    iou = float((mask_a & mask_b).sum()) / float(union.sum())
    contrast = float(a[mask_a].std()) + 1e-6 if mask_a.any() else 1e-6
    shading = float(np.abs(a - b)[union].mean()) / contrast
    return (1.0 - iou) + shading


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12, help="poses sampled per category")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--categories", nargs="+", default=list(assets.CATEGORIES))
    p.add_argument("--threshold", type=float, default=0.15,
                   help="distinguishability below which a flipped pair counts as confusable")
    args = p.parse_args()

    tmp = Path("/tmp/ambiguity_check")
    tmp.mkdir(parents=True, exist_ok=True)
    sb.configure_render(args.res, "BLENDER_EEVEE", 16)
    bpy.context.scene.render.film_transparent = True      # alpha gives the silhouette
    bpy.context.scene.render.image_settings.color_mode = "RGBA"

    print(f"worst-case distinguishability between a pose and its 180 deg flips "
          f"({args.n} poses/category; higher is better)\n")
    print(f"{'category':10s} {'mean':>8s} {'min':>8s} {'confusable':>12s}")
    results = {}

    for cat in args.categories:
        worst = []
        for i in range(args.n):
            rng = np.random.default_rng(1000 + i)
            sb.reset_scene()
            obj = assets.build(cat, rng)
            sb.apply_material(obj, rng)
            sb.randomise_lighting(rng)
            cam = sb.add_camera(rng)
            # Fixed camera + fixed lighting across the pair: any difference is shape alone.
            eye = geo.sample_viewpoint(rng, (-20.0, 60.0), (0.0, 360.0), (3.2, 4.2))
            sb.place_camera(cam, geo.look_at(eye), geo)

            R = geo.random_rotation(rng)
            obj.rotation_quaternion = geo.quat_from_matrix(R).tolist()
            bpy.context.view_layer.update()
            base, mask = render_rgba(tmp / "a")

            diffs = []
            for F in FLIPS.values():
                obj.rotation_quaternion = geo.quat_from_matrix(R @ F).tolist()
                bpy.context.view_layer.update()
                flip, flip_mask = render_rgba(tmp / "b")
                diffs.append(distinguishability(base, mask, flip, flip_mask))
            worst.append(min(diffs))

        worst = np.array(worst)
        conf = float((worst < args.threshold).mean())
        results[cat] = {"mean": float(worst.mean()), "min": float(worst.min()), "confusable": conf}
        flag = "  <-- AMBIGUOUS" if conf > 0.15 else ""
        print(f"{cat:10s} {worst.mean():8.4f} {worst.min():8.4f} {conf:11.1%}{flag}")

    print(f"\n(threshold {args.threshold}: below it, a flipped pair is visually confusable)")
    return results


if __name__ == "__main__":
    main()
