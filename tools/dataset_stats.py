"""Recompute dataset statistics from the labels on disk and repair the manifest.

``generate.py`` writes ``dataset.json`` from the counters of the run that produced
it, so a run resumed with ``--start`` reports only the samples *that* invocation
rendered. Deriving the stats from disk instead makes the manifest correct
regardless of how many invocations it took, and doubles as an integrity check that
every label has its image and depth map.

    .venv/bin/python tools/dataset_stats.py --data data/synthetic --write
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/synthetic"))
    p.add_argument("--write", action="store_true", help="update dataset.json in place")
    args = p.parse_args()

    splits, per_cat, missing = {}, Counter(), []
    dist, scale, lens, elev, depth_cov = [], [], [], [], []

    for split in ("train", "val", "test"):
        metas = sorted((args.data / "labels" / split).glob("meta_*.json"))
        splits[split] = len(metas)
        for mp in metas:
            m = json.loads(mp.read_text())
            per_cat[m["category"]] += 1

            for rel in (m["image"], m.get("depth")):
                if rel and not (args.data / rel).exists():
                    missing.append(rel)

            t = np.array(m["pose_cv"]["t"])
            dist.append(float(np.linalg.norm(t)))
            scale.append(m["pose_cv"]["scale"])
            lens.append(m["camera"]["lens_mm"])

            # Camera elevation, recovered from the world-to-camera extrinsics.
            R_wc = np.array(m["extrinsics_cv"]["R_world_to_cam"])
            t_wc = np.array(m["extrinsics_cv"]["t_world_to_cam"])
            eye = -R_wc.T @ t_wc
            elev.append(float(np.degrees(np.arcsin(eye[2] / max(np.linalg.norm(eye), 1e-9)))))

    def summarise(name, xs):
        a = np.array(xs)
        return {"name": name, "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                "mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3)}

    # Foreground coverage on a sample of depth maps -- how much of the frame the object fills.
    rng = np.random.default_rng(0)
    train_metas = sorted((args.data / "labels" / "train").glob("meta_*.json"))
    for mp in rng.choice(train_metas, size=min(300, len(train_metas)), replace=False):
        m = json.loads(Path(mp).read_text())
        d = np.load(args.data / m["depth"])["depth"].astype(np.float32)
        depth_cov.append(float((d > 0).mean()))

    stats = {
        "total": sum(splits.values()),
        "split_sizes": splits,
        "per_category": dict(per_cat),
        "missing_files": missing,
        "distributions": [summarise("camera_distance_m", dist), summarise("object_scale", scale),
                          summarise("focal_length_mm", lens), summarise("camera_elevation_deg", elev)],
        "object_frame_coverage": summarise("fg_fraction_of_image", depth_cov),
    }

    print(f"total samples: {stats['total']}   splits: {splits}")
    print(f"per category : {stats['per_category']}")
    for d in stats["distributions"]:
        print(f"  {d['name']:24s} min {d['min']:8.3f}  max {d['max']:8.3f}  "
              f"mean {d['mean']:8.3f}  std {d['std']:6.3f}")
    c = stats["object_frame_coverage"]
    print(f"  {'object % of image':24s} min {c['min']*100:7.2f}%  max {c['max']*100:7.2f}%  "
          f"mean {c['mean']*100:7.2f}%")
    print(f"missing files: {len(missing)}")

    if args.write:
        manifest_path = args.data / "dataset.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update({k: stats[k] for k in
                         ("total", "split_sizes", "per_category", "distributions",
                          "object_frame_coverage")})
        manifest["n_rendered"] = stats["total"]
        manifest["note"] = "counts recomputed from disk by tools/dataset_stats.py"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"updated {manifest_path}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
