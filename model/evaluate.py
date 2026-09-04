"""Evaluate a trained checkpoint on a held-out split and render qualitative comparisons.

    .venv/bin/python model/evaluate.py --checkpoint checkpoints/baseline/best.pt --split test

Writes ``results/metrics_<split>.json`` and side-by-side prediction images to
``results/predictions/``.

The headline number is **geodesic rotation error in degrees**: the single angle the
prediction would have to be turned through to reach the truth. Chance is ~126 deg (the
mean angle between two uniformly random rotations), which is the reference point any
result here should be read against.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))
sys.path.insert(0, str(ROOT / "blender_gen"))

import geometry as geo           # noqa: E402
import rotation as rot           # noqa: E402
from dataset import PoseDataset  # noqa: E402
from model import PoseNet        # noqa: E402

BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6),
             (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))
CATEGORIES = ("suzanne", "mug", "bracket")


def pick_device(req):
    if req != "auto":
        return torch.device(req)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def run_inference(net, ds, device, t_mean, t_std, batch_size=64):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    R_all, t_all, err_all, cat_all, rel_all = [], [], [], [], []
    for img, R_gt, t_gt, cat in loader:
        R_pred, t_norm = net(img.to(device))
        t_pred = (t_norm * t_std + t_mean).cpu()
        err_all.append(rot.geodesic_error_deg(R_pred, R_gt.to(device)))
        rel_all.append(torch.norm(t_pred - t_gt, dim=1) / torch.norm(t_gt, dim=1))
        R_all.append(R_pred.cpu())
        t_all.append(t_pred)
        cat_all.append(cat)
    return (torch.cat(R_all).numpy(), torch.cat(t_all).numpy(),
            torch.cat(err_all).numpy(), torch.cat(cat_all).numpy(),
            torch.cat(rel_all).numpy())


def draw_comparison(ds, idx, R_pred, t_pred, err, out_path, size=320):
    """Render ground-truth (green) and predicted (magenta) 3D boxes side by side."""
    rec = ds.records[idx]
    meta = json.loads(Path(rec["meta_path"]).read_text())
    K = np.array(meta["camera"]["K"])
    bbox = np.array(meta["object"]["bbox_local"])
    s = meta["pose_cv"]["scale"]
    R_gt, t_gt = np.array(meta["pose_cv"]["R"]), np.array(meta["pose_cv"]["t"])

    base = Image.open(rec["image"]).convert("RGB")
    k = size / base.size[0]
    base = base.resize((size, size), Image.LANCZOS)

    def panel(R, t, colour, title):
        im = base.copy()
        d = ImageDraw.Draw(im)
        uv, z = geo.project(K, R, t, s, bbox)
        if np.all(z > 0):
            for i, j in BOX_EDGES:
                d.line([tuple(uv[i] * k), tuple(uv[j] * k)], fill=colour, width=2)
            c, _ = geo.project(K, R, t, s, np.zeros((1, 3)))
            cx, cy = c[0] * k
            ax, _ = geo.project(K, R, t, s, np.eye(3) * 0.6)
            for (x, y), col in zip(ax * k, [(255, 80, 80), (80, 255, 80), (90, 140, 255)]):
                d.line([(cx, cy), (x, y)], fill=col, width=3)
        return im, title

    # Predicted t is t/s; scale back by the true s so both boxes are drawn comparably.
    left, lt = panel(R_gt, t_gt, (60, 255, 90), "ground truth")
    right, rt = panel(R_pred, t_pred * s, (255, 70, 220), f"predicted  ({err:.1f} deg err)")

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    pad, top = 10, 28
    canvas = Image.new("RGB", (size * 2 + pad * 3, size + top + pad), (18, 18, 20))
    dr = ImageDraw.Draw(canvas)
    for i, (im, title) in enumerate([(left, lt), (right, rt)]):
        x = pad + i * (size + pad)
        canvas.paste(im, (x, top))
        dr.text((x, 6), title, fill=(235, 235, 235), font=font)
    canvas.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/baseline/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--device", default="auto")
    p.add_argument("--n-vis", type=int, default=8, help="qualitative comparisons to render")
    args = p.parse_args()

    device = pick_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Rebuild the exact architecture that was trained. Dropout adds a layer, so a
    # checkpoint trained with it will not load into a model built with the default.
    trained_args = ckpt.get("args", {})
    net = PoseNet(ckpt.get("backbone", "resnet18"), pretrained=False,
                  dropout=float(trained_args.get("dropout", 0.0))).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()

    t_mean = torch.tensor(ckpt["t_mean"], dtype=torch.float32, device=device)
    t_std = torch.tensor(ckpt["t_std"], dtype=torch.float32, device=device)

    ds = PoseDataset(args.data, args.split, augment=False)
    R_pred, t_pred, err, cat, rel = run_inference(net, ds, device, t_mean, t_std)

    per_cat = {}
    for cid, name in enumerate(CATEGORIES):
        m = cat == cid
        if m.any():
            per_cat[name] = {
                "n": int(m.sum()),
                "rot_mean_deg": round(float(err[m].mean()), 2),
                "rot_median_deg": round(float(np.median(err[m])), 2),
                "acc_10deg": round(float((err[m] < 10).mean()), 4),
                "acc_30deg": round(float((err[m] < 30).mean()), 4),
                "trans_rel_err": round(float(rel[m].mean()), 4),
            }

    metrics = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n": int(len(err)),
        "chance_rot_error_deg": 126.0,
        "overall": {
            "rot_mean_deg": round(float(err.mean()), 2),
            "rot_median_deg": round(float(np.median(err)), 2),
            "rot_p90_deg": round(float(np.percentile(err, 90)), 2),
            "acc_10deg": round(float((err < 10).mean()), 4),
            "acc_30deg": round(float((err < 30).mean()), 4),
            "trans_rel_err": round(float(rel.mean()), 4),
        },
        "per_category": per_cat,
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))

    o = metrics["overall"]
    print(f"{args.split}: n={metrics['n']}   (chance = ~126 deg)")
    print(f"  rotation error   mean {o['rot_mean_deg']:6.2f} deg   "
          f"median {o['rot_median_deg']:6.2f}   p90 {o['rot_p90_deg']:6.2f}")
    print(f"  accuracy         <10 deg {o['acc_10deg']:.1%}    <30 deg {o['acc_30deg']:.1%}")
    print(f"  translation      rel err {o['trans_rel_err']:.4f}")
    print("  per category:")
    for name, v in per_cat.items():
        print(f"    {name:9s} n={v['n']:4d}  mean {v['rot_mean_deg']:6.2f}  "
              f"med {v['rot_median_deg']:6.2f}  <10deg {v['acc_10deg']:.1%}  "
              f"t_err {v['trans_rel_err']:.4f}")

    # Qualitative panels spanning the error range, best to worst.
    vis_dir = args.out / "predictions"
    vis_dir.mkdir(parents=True, exist_ok=True)
    order = np.argsort(err)
    picks = order[np.linspace(0, len(order) - 1, args.n_vis).astype(int)]
    for rank, i in enumerate(picks):
        draw_comparison(ds, int(i), R_pred[i], t_pred[i], err[i],
                        vis_dir / f"pred_{rank:02d}_{CATEGORIES[cat[i]]}_{err[i]:.0f}deg.png")
    print(f"\n  metrics -> {args.out / f'metrics_{args.split}.json'}")
    print(f"  {args.n_vis} comparisons -> {vis_dir}")


if __name__ == "__main__":
    main()
