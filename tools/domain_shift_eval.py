"""Quantify how far the model degrades under corruptions that mimic real photographs.

The dataset renders objects against flat, uncluttered colour under clean synthetic
lighting. Real photographs differ along several axes at once, and this project's title
claims synthetic-to-*real* transfer, so the size of that gap is a result the README owes
the reader. Photographs are the real test, but they need a camera and a defined canonical
frame; this measures the same axes in a controlled way, with exact ground truth, and says
which part of the domain gap actually costs accuracy.

Each corruption is applied to the held-out test split at increasing severity:

* ``background``  -- composite the object onto procedural clutter using its depth mask.
                     This is the closest proxy for "photographed on a real desk", and the
                     dataset's single biggest unrealism.
* ``noise``       -- sensor noise.
* ``blur``        -- defocus / camera shake.
* ``jpeg``        -- compression artefacts, which every phone photo carries.
* ``colour``      -- white-balance shift, i.e. lighting the renders never saw.

    .venv/bin/python tools/domain_shift_eval.py --checkpoint checkpoints/improved/best.pt
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import rotation as rot           # noqa: E402
from dataset import IMAGENET_MEAN, IMAGENET_STD, PoseDataset  # noqa: E402
from model import PoseNet        # noqa: E402


def procedural_background(h, w, rng, complexity):
    """A cluttered background: smoothed colour noise plus a few hard-edged shapes.

    Not a photograph, but it supplies what flat colour does not -- competing edges,
    texture and colour variation that the model must learn to ignore.
    """
    small = rng.random((max(2, int(h / 32)), max(2, int(w / 32)), 3)).astype(np.float32)
    bg = np.asarray(Image.fromarray((small * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                    dtype=np.float32) / 255.0

    img = Image.fromarray((bg * 255).astype(np.uint8))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for _ in range(int(complexity * 12)):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        x1, y1 = x0 + rng.integers(10, w // 2), y0 + rng.integers(10, h // 2)
        col = tuple(int(c) for c in rng.integers(0, 255, 3))
        (d.rectangle if rng.random() < 0.5 else d.ellipse)([x0, y0, x1, y1], fill=col)
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0 + 2.0 * (1.0 - complexity)))
    return np.asarray(img, dtype=np.float32) / 255.0


def corrupt(arr, mask, kind, severity, rng):
    """Apply one corruption at ``severity`` in [0, 1]. ``arr`` is HxWx3 float in [0,1]."""
    if severity <= 0:
        return arr
    h, w = arr.shape[:2]

    if kind == "background":
        bg = procedural_background(h, w, rng, severity)
        m = mask[..., None].astype(np.float32)
        return arr * m + bg * (1.0 - m)

    if kind == "noise":
        return np.clip(arr + rng.normal(0, 0.12 * severity, arr.shape).astype(np.float32), 0, 1)

    if kind == "blur":
        img = Image.fromarray((arr * 255).astype(np.uint8))
        return np.asarray(img.filter(ImageFilter.GaussianBlur(radius=3.0 * severity)),
                          dtype=np.float32) / 255.0

    if kind == "jpeg":
        quality = int(round(95 - 88 * severity))
        buf = io.BytesIO()
        Image.fromarray((arr * 255).astype(np.uint8)).save(buf, "JPEG", quality=max(quality, 5))
        buf.seek(0)
        return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0

    if kind == "colour":
        gain = 1.0 + rng.uniform(-0.45, 0.45, size=3).astype(np.float32) * severity
        return np.clip(arr * gain, 0, 1)

    raise ValueError(kind)


@torch.no_grad()
def evaluate(net, ds, device, kind, severity, img_size=224, seed=0):
    errs = []
    for i in range(len(ds)):
        rec = ds.records[i]
        img = Image.open(rec["image"]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0

        mask = None
        if kind == "background":
            depth = np.load(ROOT / "data" / "synthetic" / rec_rel(rec))["depth"].astype(np.float32)
            mask = depth > 0

        arr = corrupt(arr, mask, kind, severity, np.random.default_rng(seed * 100003 + i))
        arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8))
                         .resize((img_size, img_size), Image.BILINEAR), dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(arr.transpose(2, 0, 1).copy())[None].to(device)

        R_pred, _ = net(x)
        errs.append(float(rot.geodesic_error_deg(R_pred, torch.from_numpy(rec["R"])[None])[0]))
    e = np.array(errs)
    return {"mean": float(e.mean()), "median": float(np.median(e)),
            "acc30": float((e < 30).mean())}


def rec_rel(rec):
    """Depth path for a record, relative to the dataset root."""
    stem = Path(rec["image"]).stem.replace("rgb_", "depth_")
    split = Path(rec["image"]).parent.name
    return Path("depth") / split / f"{stem}.npz"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/improved/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=300, help="test images per condition")
    p.add_argument("--out", type=Path, default=Path("results/domain_shift.json"))
    args = p.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    net = PoseNet(ckpt.get("backbone", "resnet18"), pretrained=False,
                  dropout=float(ckpt.get("args", {}).get("dropout", 0.0))).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    ds = PoseDataset(args.data, args.split, augment=False)
    ds.records = ds.records[:args.n]

    base = evaluate(net, ds, device, "none", 0.0)
    print(f"clean synthetic ({args.n} images): mean {base['mean']:.2f} deg   "
          f"median {base['median']:.2f}   within 30 deg {base['acc30']:.1%}\n")
    print(f"{'corruption':12s} {'severity':>9s} {'mean deg':>9s} {'median':>8s} "
          f"{'<30 deg':>8s} {'vs clean':>9s}")

    results = {"clean": base, "conditions": {}}
    for kind in ("background", "noise", "blur", "jpeg", "colour"):
        for sev in (0.35, 0.7, 1.0):
            m = evaluate(net, ds, device, kind, sev)
            results["conditions"][f"{kind}@{sev}"] = m
            print(f"{kind:12s} {sev:9.2f} {m['mean']:9.2f} {m['median']:8.2f} "
                  f"{m['acc30']:7.1%} {m['mean'] - base['mean']:+8.2f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
