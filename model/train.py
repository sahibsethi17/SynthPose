"""Train PoseNet on the synthetic dataset.

    .venv/bin/python model/train.py --data data/synthetic --epochs 40

Logs per-epoch metrics to ``checkpoints/<run>/log.csv`` and keeps the checkpoint with
the best validation rotation error.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

import rotation as rot          # noqa: E402
from dataset import PoseDataset  # noqa: E402
from model import PoseNet        # noqa: E402


def pick_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--run", default="baseline")
    p.add_argument("--backbone", default="resnet18", choices=list(("resnet18", "resnet34", "resnet50")))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--trans-weight", type=float, default=1.0)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--cache", action="store_true", help="hold decoded images in RAM")
    p.add_argument("--limit-train", type=int, default=0, help="debug: cap training samples")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def evaluate(net, loader, device, t_mean, t_std):
    """Return validation metrics in interpretable units (degrees, and % of t/s)."""
    net.eval()
    errs, rel, losses = [], [], []
    for img, R_gt, t_gt, _ in loader:
        img, R_gt, t_gt = img.to(device), R_gt.to(device), t_gt.to(device)
        R_pred, t_norm = net(img)
        t_pred = t_norm * t_std + t_mean

        errs.append(rot.geodesic_error_deg(R_pred, R_gt))
        rel.append((torch.norm(t_pred - t_gt, dim=1) / torch.norm(t_gt, dim=1)).cpu())
        losses.append(rot.frobenius_loss(R_pred, R_gt).item())

    errs = torch.cat(errs).numpy()
    rel = torch.cat(rel).numpy()
    return {
        "rot_mean_deg": float(errs.mean()),
        "rot_median_deg": float(np.median(errs)),
        "rot_acc_10deg": float((errs < 10).mean()),
        "rot_acc_30deg": float((errs < 30).mean()),
        "trans_rel_err": float(rel.mean()),
        "val_rot_loss": float(np.mean(losses)),
    }


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    train_ds = PoseDataset(args.data, "train", args.img_size, args.cache, not args.no_augment)
    val_ds = PoseDataset(args.data, "val", args.img_size, args.cache, False)
    if args.limit_train:
        train_ds.records = train_ds.records[:args.limit_train]

    # Standardise the translation target using TRAIN statistics only -- computing them
    # over val/test would leak information about the held-out splits into training.
    t_mean_np, t_std_np = train_ds.target_stats()
    t_mean = torch.tensor(t_mean_np, dtype=torch.float32, device=device)
    t_std = torch.tensor(t_std_np, dtype=torch.float32, device=device)

    common = dict(num_workers=args.workers, pin_memory=(device.type == "cuda"),
                  persistent_workers=args.workers > 0)
    train_ld = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          drop_last=True, **common)
    val_ld = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **common)

    net = PoseNet(args.backbone, pretrained=not args.no_pretrained,
                  dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_ld), pct_start=0.15)
    huber = nn.SmoothL1Loss()

    run_dir = args.out / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(
        {**{k: str(v) for k, v in vars(args).items()},
         "device": str(device), "train_size": len(train_ds), "val_size": len(val_ds),
         "t_mean": t_mean_np.tolist(), "t_std": t_std_np.tolist()}, indent=2))

    csv_path = run_dir / "log.csv"
    fields = ["epoch", "train_loss", "train_rot_loss", "train_trans_loss", "lr",
              "rot_mean_deg", "rot_median_deg", "rot_acc_10deg", "rot_acc_30deg",
              "trans_rel_err", "val_rot_loss", "epoch_sec"]
    with csv_path.open("w", newline="") as f:
        csv.DictWriter(f, fields).writeheader()

    print(f"device={device}  backbone={args.backbone}  train={len(train_ds)}  val={len(val_ds)}")
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        net.train()
        t0 = time.time()
        sums = np.zeros(3)
        for img, R_gt, t_gt, _ in train_ld:
            img, R_gt, t_gt = img.to(device), R_gt.to(device), t_gt.to(device)
            R_pred, t_norm = net(img)

            rot_loss = rot.frobenius_loss(R_pred, R_gt)
            trans_loss = huber(t_norm, (t_gt - t_mean) / t_std)
            loss = rot_loss + args.trans_weight * trans_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            sched.step()
            sums += [loss.item(), rot_loss.item(), trans_loss.item()]

        n = len(train_ld)
        metrics = evaluate(net, val_ld, device, t_mean, t_std)
        row = {"epoch": epoch, "train_loss": sums[0] / n, "train_rot_loss": sums[1] / n,
               "train_trans_loss": sums[2] / n, "lr": sched.get_last_lr()[0],
               **metrics, "epoch_sec": round(time.time() - t0, 1)}
        with csv_path.open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow({k: row[k] for k in fields})

        print(f"ep {epoch:3d}/{args.epochs}  loss {row['train_loss']:.4f}  "
              f"val_rot {metrics['rot_mean_deg']:6.2f}deg (med {metrics['rot_median_deg']:5.2f})  "
              f"acc@10 {metrics['rot_acc_10deg']:.3f}  "
              f"t_err {metrics['trans_rel_err']:.4f}  {row['epoch_sec']:.0f}s", flush=True)

        if metrics["rot_mean_deg"] < best:
            best = metrics["rot_mean_deg"]
            torch.save({"model": net.state_dict(), "epoch": epoch, "metrics": metrics,
                        "args": vars(args), "t_mean": t_mean_np, "t_std": t_std_np,
                        "backbone": args.backbone},
                       run_dir / "best.pt")

    print(f"\nbest val rotation error: {best:.2f} deg  ->  {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
