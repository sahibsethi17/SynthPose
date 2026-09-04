"""PyTorch Dataset over the Blender-generated samples.

Reads only the JSON labels and PNGs written by ``blender_gen/generate.py``; nothing
here imports ``bpy``, so training runs with no Blender dependency at all.

**Regression target.** The model predicts ``R`` and ``t / s`` rather than ``R`` and
``t``. Object scale and camera distance were randomised independently, so apparent
size determines only their ratio -- a small near object and a large far one produce
identical pixels. ``t`` is genuinely unrecoverable from one image; ``t / s`` is fully
determined. Training on ``t`` would be optimising toward a target the input does not
determine, and the irreducible error would look like model failure.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PoseDataset(Dataset):
    """Yields ``(image, R, t_over_s, category_id)`` for one split."""

    def __init__(self, root, split="train", img_size=224, cache=False, augment=False):
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == "train"

        label_dir = self.root / "labels" / split
        paths = sorted(label_dir.glob("meta_*.json"))
        if not paths:
            raise FileNotFoundError(f"no labels in {label_dir}")

        self.records = []
        for p in paths:
            m = json.loads(p.read_text())
            t = np.array(m["pose_cv"]["t"], dtype=np.float32)
            self.records.append({
                "meta_path": p,          # evaluation re-reads K / bbox for visualisation
                "image": self.root / m["image"],
                "R": np.array(m["pose_cv"]["R"], dtype=np.float32),
                "t_over_s": t / np.float32(m["pose_cv"]["scale"]),
                "category_id": m["category_id"],
            })
        self._cache = [None] * len(self.records) if cache else None

    def __len__(self):
        return len(self.records)

    def target_stats(self):
        """Mean/std of ``t/s`` over this split, for output standardisation."""
        arr = np.stack([r["t_over_s"] for r in self.records])
        return arr.mean(0), arr.std(0) + 1e-8

    def _load_image(self, idx):
        if self._cache is not None and self._cache[idx] is not None:
            return self._cache[idx]
        img = Image.open(self.records[idx]["image"]).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if self._cache is not None:
            self._cache[idx] = arr
        return arr

    def _augment(self, arr):
        """Pose-preserving augmentation.

        Every transform here is photometric or an occlusion. Geometric transforms --
        flip, rotate, crop, translate -- are deliberately excluded: they change the
        object's true pose while leaving the stored label untouched, so they would teach
        the model a wrong answer with full confidence. A horizontal flip in particular
        turns a rotation into its mirror image, which is not a rotation at all.

        Occlusion (cutout) is safe and valuable: hiding part of the object does not move
        it, and it discourages the memorisation of whole training images that the
        4k-sample baseline showed.
        """
        rng = np.random

        arr = arr * rng.uniform(0.75, 1.25) + rng.uniform(-0.08, 0.08)   # brightness/contrast
        arr = arr * rng.uniform(0.9, 1.1, size=3).astype(np.float32)      # colour cast
        if rng.rand() < 0.15:                                             # desaturate
            grey = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            arr = np.repeat(grey[:, :, None], 3, axis=2)
        if rng.rand() < 0.5:                                              # sensor noise
            arr = arr + rng.normal(0.0, rng.uniform(0.01, 0.05), arr.shape).astype(np.float32)
        arr = np.clip(arr, 0.0, 1.0)

        if rng.rand() < 0.5:                                              # cutout
            h, w = arr.shape[:2]
            for _ in range(rng.randint(1, 3)):
                ch, cw = rng.randint(h // 10, h // 4), rng.randint(w // 10, w // 4)
                y, x = rng.randint(0, h - ch), rng.randint(0, w - cw)
                arr[y:y + ch, x:x + cw] = rng.rand()
        return arr

    def __getitem__(self, idx):
        rec = self.records[idx]
        arr = self._load_image(idx).astype(np.float32) / 255.0

        if self.augment:
            arr = self._augment(arr)

        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        image = torch.from_numpy(arr.transpose(2, 0, 1).copy())
        return (image,
                torch.from_numpy(rec["R"]),
                torch.from_numpy(rec["t_over_s"]),
                rec["category_id"])
