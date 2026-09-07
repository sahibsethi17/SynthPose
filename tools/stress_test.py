"""Condition test error on capture parameters to find systematic weaknesses.

A single mean error hides structure. If the model is much worse at particular
elevations, distances or apparent sizes, that is a fixable property of the dataset or the
input pipeline -- and it is invisible in an aggregate number. This runs inference once and
slices the resulting per-sample errors by every parameter the generator recorded.

    .venv/bin/python tools/stress_test.py --checkpoint checkpoints/improved/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import rotation as rot           # noqa: E402
from dataset import PoseDataset  # noqa: E402
from model import PoseNet        # noqa: E402


def load_net(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    net = PoseNet(ckpt.get("backbone", "resnet18"), pretrained=False,
                  dropout=float(ckpt.get("args", {}).get("dropout", 0.0))).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net, ckpt


@torch.no_grad()
def per_sample_errors(net, ds, device):
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4)
    errs = []
    for img, R_gt, _, _ in loader:
        R_pred, _ = net(img.to(device))
        errs.append(rot.geodesic_error_deg(R_pred, R_gt.to(device)))
    return torch.cat(errs).numpy()


def bin_report(name, values, errs, edges, unit=""):
    """Print mean error and flip rate per bin of ``values``."""
    print(f"\n  by {name}:")
    print(f"    {'range':>18s} {'n':>6s} {'mean deg':>9s} {'median':>8s} "
          f"{'<30 deg':>8s} {'>150 deg':>9s}")
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        if m.sum() < 10:
            continue
        e = errs[m]
        rows.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                     "mean": float(e.mean()), "median": float(np.median(e)),
                     "acc30": float((e < 30).mean()), "flip": float((e > 150).mean())})
        print(f"    {lo:7.2f}-{hi:<7.2f}{unit:>4s} {m.sum():6d} {e.mean():9.2f} "
              f"{np.median(e):8.2f} {(e < 30).mean():7.1%} {(e > 150).mean():8.1%}")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/improved/best.pt"))
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, default=Path("results/stress_test.json"))
    args = p.parse_args()

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    net, _ = load_net(args.checkpoint, device)
    ds = PoseDataset(args.data, args.split, augment=False)
    errs = per_sample_errors(net, ds, device)

    # Recover the generator's parameters for each sample.
    elev, dist, scale, lens, cover, tz = [], [], [], [], [], []
    for rec in ds.records:
        m = json.loads(Path(rec["meta_path"]).read_text())
        R_wc = np.array(m["extrinsics_cv"]["R_world_to_cam"])
        t_wc = np.array(m["extrinsics_cv"]["t_world_to_cam"])
        eye = -R_wc.T @ t_wc
        elev.append(np.degrees(np.arcsin(eye[2] / max(np.linalg.norm(eye), 1e-9))))
        dist.append(np.linalg.norm(eye))
        scale.append(m["pose_cv"]["scale"])
        lens.append(m["camera"]["lens_mm"])
        tz.append(m["pose_cv"]["t"][2])
        d = np.load(args.data / m["depth"])["depth"].astype(np.float32)
        cover.append(float((d > 0).mean()))
    elev, dist, scale = np.array(elev), np.array(dist), np.array(scale)
    lens, cover, tz = np.array(lens), np.array(cover), np.array(tz)

    print(f"stress test on {len(errs)} {args.split} images")
    print(f"  overall: mean {errs.mean():.2f} deg   median {np.median(errs):.2f}   "
          f"within 30 deg {(errs < 30).mean():.1%}   flips {(errs > 150).mean():.1%}")

    out = {"overall": {"n": int(len(errs)), "mean": float(errs.mean()),
                       "median": float(np.median(errs)),
                       "acc30": float((errs < 30).mean()),
                       "flip": float((errs > 150).mean())}, "slices": {}}

    out["slices"]["camera_elevation_deg"] = bin_report(
        "camera elevation", elev, errs, [-25, -10, 5, 20, 35, 50, 65, 75.01], "deg")
    out["slices"]["object_coverage_pct"] = bin_report(
        "object coverage (fraction of frame)", cover, errs,
        [0, .04, .08, .12, .16, .22, .60])
    out["slices"]["camera_distance_m"] = bin_report(
        "camera distance", dist, errs, [2.8, 3.2, 3.6, 4.0, 4.4, 5.001], "m")
    out["slices"]["object_scale"] = bin_report(
        "object scale", scale, errs, [0.8, 0.95, 1.1, 1.25, 1.401])
    out["slices"]["focal_length_mm"] = bin_report(
        "focal length", lens, errs, [40, 50, 60, 70, 80.01], "mm")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n  -> {args.out}")


if __name__ == "__main__":
    main()
