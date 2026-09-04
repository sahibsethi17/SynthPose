"""Run the trained model on real photographs -- the sim-to-real test.

    .venv/bin/python model/predict_real.py --images real_photos/ --category mug
    .venv/bin/python model/predict_real.py --images real_photos/ --turntable-step 30

**The canonical frame.** A rotation is only meaningful relative to a reference
orientation, and the model's reference is how the object was built in Blender, not
anything intrinsic to a real mug. For the synthetic mug that frame is::

    +Z  up through the cup's axis, from base to rim
    +X  horizontally out through the handle
    +Y  completes the right-handed frame

So "identity rotation" means: upright, handle pointing along +X, viewed head-on. To read
absolute numbers off a photograph you must orient the real mug the same way. If you do
not, the predictions are not wrong so much as unanchored.

**Two ways to score this.**

*Qualitative* (no setup): predict a pose per photo and draw the predicted orientation
onto the image. A human judges whether it is close. Honest and quick, but not a number.

*Quantitative without pose labels* (``--turntable-step``): photograph the object rotating
in known increments -- a mug on a plate turned 30 deg at a time. Absolute pose labels are
still impossible to write down by hand, but the **relative** rotation between consecutive
frames is known exactly, and comparing predicted relative rotations against it is a real
error measurement. This sidesteps the labelling problem that motivated synthetic data in
the first place.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "blender_gen"))

import geometry as geo           # noqa: E402
import rotation as rot           # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from model import PoseNet        # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tif", ".tiff"}
BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
UNIT_BOX = np.array([[x, y, z] for x in (-.5, .5) for y in (-.35, .35) for z in (-.5, .5)])
# Reorder to match the edge list's expected corner ordering.
UNIT_BOX = UNIT_BOX[[0, 1, 3, 2, 4, 5, 7, 6]]


def load_square(path, size=224):
    """Centre-crop to square, then resize -- matching the square renders the model saw.

    Real photos are rarely square and rarely frame the object the way the renders do.
    Centre-cropping assumes the object is roughly centred, which is why the capture
    instructions ask for that. A detect-and-crop front end would remove the assumption.
    """
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    img = img.crop(((w - side) // 2, (h - side) // 2,
                    (w - side) // 2 + side, (h - side) // 2 + side))
    return img.resize((size, size), Image.BILINEAR)


@torch.no_grad()
def predict(net, img, device, t_mean, t_std):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(arr.transpose(2, 0, 1).copy())[None].to(device)
    R, t_norm = net(x)
    return R[0].cpu().numpy(), (t_norm * t_std + t_mean)[0].cpu().numpy()


def draw_pose(img, R, t, fov_deg, out_path, label="", box_scale=0.8):
    """Overlay the predicted orientation: a wireframe box plus the object's axes.

    Read the **axes**, not the box size. Drawing anything at a metric position needs the
    camera's focal length, and a phone's is unknown here -- so ``--fov`` and
    ``--box-scale`` are presentation knobs, tuned until the box roughly wraps the object.
    Neither touches the predicted rotation, which is what is actually being tested.
    """
    size = img.size[0]
    f_px = (size / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    K = np.array([[f_px, 0, size / 2.0], [0, f_px, size / 2.0], [0, 0, 1.0]])

    canvas = img.resize((size * 2, size * 2), Image.LANCZOS)
    d = ImageDraw.Draw(canvas)
    uv, z = geo.project(K, R, t, 1.0, UNIT_BOX * box_scale)
    if np.all(z > 0):
        for i, j in BOX_EDGES:
            d.line([tuple(uv[i] * 2), tuple(uv[j] * 2)], fill=(255, 70, 220), width=2)
    c, _ = geo.project(K, R, t, 1.0, np.zeros((1, 3)))
    cx, cy = c[0] * 2
    ax, _ = geo.project(K, R, t, 1.0, np.eye(3) * 0.45)
    for (x, y), col in zip(ax * 2, [(255, 80, 80), (80, 255, 80), (90, 140, 255)]):
        d.line([(cx, cy), (x, y)], fill=col, width=5)

    if label:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        d.rectangle([0, 0, canvas.size[0], 34], fill=(18, 18, 20))
        d.text((8, 7), label, fill=(235, 235, 235), font=font)
    canvas.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=Path, required=True, help="folder of photographs")
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/improved/best.pt"))
    p.add_argument("--out", type=Path, default=Path("results/real"))
    p.add_argument("--fov", type=float, default=33.4,
                   help="FOV in degrees, for DRAWING ONLY (training renders spanned ~26-50)")
    p.add_argument("--box-scale", type=float, default=0.8,
                   help="size of the drawn box, for DRAWING ONLY; tune so it wraps the object")
    p.add_argument("--turntable-step", type=float, default=None,
                   help="known degrees between consecutive photos, sorted by filename")
    p.add_argument("--category", default=None, help="label for the report only")
    args = p.parse_args()

    paths = sorted(q for q in args.images.iterdir() if q.suffix.lower() in IMAGE_EXT)
    if not paths:
        sys.exit(f"no images found in {args.images} (looked for {sorted(IMAGE_EXT)})")

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = PoseNet(ckpt.get("backbone", "resnet18"), pretrained=False,
                  dropout=float(ckpt.get("args", {}).get("dropout", 0.0))).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    t_mean = torch.tensor(ckpt["t_mean"], dtype=torch.float32, device=device)
    t_std = torch.tensor(ckpt["t_std"], dtype=torch.float32, device=device)

    args.out.mkdir(parents=True, exist_ok=True)
    preds, report = [], []
    for path in paths:
        img = load_square(path)
        R, t = predict(net, img, device, t_mean, t_std)
        preds.append(R)
        draw_pose(img, R, t, args.fov, args.out / f"pred_{path.stem}.png",
                  label=path.name, box_scale=args.box_scale)
        report.append({"image": path.name, "R": R.tolist(), "t_over_s": t.tolist()})
        print(f"  {path.name:28s} t/s = [{t[0]:6.2f} {t[1]:6.2f} {t[2]:6.2f}]")

    print(f"\n{len(paths)} photographs -> {args.out}")

    result = {"category": args.category, "n": len(paths), "predictions": report}

    if args.turntable_step is not None and len(preds) > 1:
        # Relative rotation between consecutive frames has known ground truth even though
        # absolute pose does not -- this is the only quantitative handle on real photos.
        expected = abs(args.turntable_step)
        errs = []
        for a, b in zip(preds[:-1], preds[1:]):
            rel = float(np.degrees(np.arccos(
                np.clip((np.trace(a.T @ b) - 1.0) / 2.0, -1.0, 1.0))))
            errs.append(abs(rel - expected))
            print(f"    predicted step {rel:6.1f} deg   expected {expected:5.1f}   "
                  f"error {abs(rel - expected):5.1f}")
        errs = np.array(errs)
        result["turntable"] = {"expected_step_deg": expected,
                               "mean_step_error_deg": float(errs.mean()),
                               "median_step_error_deg": float(np.median(errs))}
        print(f"\n  relative-rotation error: mean {errs.mean():.1f} deg   "
              f"median {np.median(errs):.1f} deg")
        print("  (random predictions would give ~40-60 deg here; compare against "
              "the 16 deg synthetic test error)")

    (args.out / "predictions.json").write_text(json.dumps(result, indent=2))
    print(f"  -> {args.out / 'predictions.json'}")


if __name__ == "__main__":
    main()
