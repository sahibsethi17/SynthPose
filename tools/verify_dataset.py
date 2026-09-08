"""Verify that the dataset's pose labels actually describe its images.

This is the exit criterion for Phase 1. Every label here passes through a chain of
conventions -- Blender's Z-up world, its camera looking down -Z, the flip to
OpenCV's Y-down frame, the intrinsics under AUTO sensor fit, and PNG's top-left
row order. A sign error anywhere in that chain still produces a plausible-looking
dataset, and the failure only surfaces much later as a model that will not
converge. So rather than eyeballing renders, this tool closes the loop
geometrically:

1. Project the object's 3D bounding box with the stored ``K``, ``R``, ``t``, ``s``.
2. Derive the true silhouette independently, from the rendered depth map.
3. Assert the projection actually lands on the object.

If the Y axis were flipped, step 3 fails immediately instead of six hours into training.

    .venv/bin/python tools/verify_dataset.py --data data/synthetic --n 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "blender_gen"))
import geometry as geo  # noqa: E402

# Edge list for Blender's bound_box corner ordering.
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


def rect_iou(a, b):
    """IoU of two ``(x0, y0, x1, y1)`` rectangles."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def check_sample(root: Path, meta: dict, draw_to: Path | None):
    """Return per-sample verification metrics, or ``None`` if depth is unavailable."""
    K = np.array(meta["camera"]["K"])
    pose = meta["pose_cv"]
    R, t, s = np.array(pose["R"]), np.array(pose["t"]), pose["scale"]
    bbox_local = np.array(meta["object"]["bbox_local"])

    uv, z = geo.project(K, R, t, s, bbox_local)
    proj_rect = (uv[:, 0].min(), uv[:, 1].min(), uv[:, 0].max(), uv[:, 1].max())

    if not meta.get("depth"):
        return None
    blob = np.load(root / meta["depth"])
    depth = blob["depth"].astype(np.float32)
    # With a ground plane and distractors in frame, "depth > 0" is the whole scene, not
    # the target. Datasets generated with clutter carry an explicit target mask; fall back
    # to the depth foreground for the earlier flat-background datasets.
    fg = blob["mask"].astype(bool) if "mask" in blob.files else (depth > 0)
    if fg.sum() < 20:
        return None

    ys, xs = np.nonzero(fg)
    sil_rect = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)

    # Fraction of true object pixels falling inside the projected 3D box.
    inside = ((xs >= proj_rect[0] - 1) & (xs <= proj_rect[2] + 1) &
              (ys >= proj_rect[1] - 1) & (ys <= proj_rect[3] + 1))

    # The object centre's depth must lie inside the rendered depth range.
    fg_depth = depth[fg]
    fg_depth = fg_depth[fg_depth > 0]
    if fg_depth.size < 10:
        return None
    radius = s * np.abs(bbox_local).max() * np.sqrt(3)
    depth_ok = (fg_depth.min() - 1e-2 <= t[2] <= fg_depth.max() + radius)

    result = {
        "index": meta["index"],
        "category": meta["category"],
        "containment": float(inside.mean()),
        "iou": float(rect_iou(proj_rect, sil_rect)),
        "z_positive": bool(np.all(z > 0)),
        "depth_ok": bool(depth_ok),
        "t_z": float(t[2]),
        "fg_depth_range": [float(fg_depth.min()), float(fg_depth.max())],
    }

    if draw_to is not None:
        img = Image.open(root / meta["image"]).convert("RGB")
        d = ImageDraw.Draw(img)
        d.rectangle(sil_rect, outline=(255, 60, 60), width=1)          # depth silhouette
        for i, j in BOX_EDGES:                                          # projected 3D box
            d.line([tuple(uv[i]), tuple(uv[j])], fill=(60, 255, 90), width=1)
        centre, _ = geo.project(K, R, t, s, np.zeros((1, 3)))
        cx, cy = centre[0]
        d.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(80, 160, 255))
        # Object axes, to show orientation is recovered and not just position.
        axes, _ = geo.project(K, R, t, s, np.eye(3) * 0.6)
        for (ax, ay), col in zip(axes, [(255, 80, 80), (80, 255, 80), (80, 80, 255)]):
            d.line([(cx, cy), (ax, ay)], fill=col, width=2)
        img.save(draw_to)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=12, help="samples to draw overlays for")
    p.add_argument("--check-all", action="store_true", help="verify every sample in the split")
    p.add_argument("--out", type=Path, default=Path("results/verify"))
    p.add_argument("--min-containment", type=float, default=0.97)
    args = p.parse_args()

    label_dir = args.data / "labels" / args.split
    metas = sorted(label_dir.glob("meta_*.json"))
    if not metas:
        sys.exit(f"no labels found in {label_dir}")
    args.out.mkdir(parents=True, exist_ok=True)

    checked = metas if args.check_all else metas[:max(args.n, 64)]
    results, drawn = [], 0
    for mp in checked:
        meta = json.loads(mp.read_text())
        target = None
        if drawn < args.n:
            target = args.out / f"overlay_{meta['index']:06d}_{meta['category']}.png"
            drawn += 1
        r = check_sample(args.data, meta, target)
        if r is not None:
            results.append(r)

    if not results:
        sys.exit("no verifiable samples (depth maps required; re-run without --no-depth)")

    cont = np.array([r["containment"] for r in results])
    iou = np.array([r["iou"] for r in results])
    worst = min(results, key=lambda r: r["containment"])

    print(f"verified {len(results)} samples from split '{args.split}'")
    print(f"  containment (object pixels inside projected 3D box):")
    print(f"      mean {cont.mean():.4f}   min {cont.min():.4f}   "
          f"frac>=0.99 {np.mean(cont >= 0.99):.3f}")
    print(f"  rect IoU (projected box vs depth silhouette): mean {iou.mean():.4f}")
    print(f"  all bbox corners in front of camera: {all(r['z_positive'] for r in results)}")
    print(f"  centre depth within rendered range:  {all(r['depth_ok'] for r in results)}")
    print(f"  worst sample: #{worst['index']} ({worst['category']}) "
          f"containment={worst['containment']:.4f}")
    print(f"  overlays -> {args.out}")

    ok = (cont.mean() >= args.min_containment
          and all(r["z_positive"] for r in results)
          and all(r["depth_ok"] for r in results))
    print("\nRESULT:", "PASS - pose labels are geometrically consistent" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
