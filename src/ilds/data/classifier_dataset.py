"""Stage2 classifier dataset: real crop patches + class-balanced sampling (no synthetic positives).

- Positives: real crops produced by build_datasets.py (manifest.csv).
- Negatives: negatives/ dir (hard-negative mining + random non-lights) -> idx 0 = not_a_light.
- Samples of rare classes not in the ClassSpace: dropped (not in the model; routed to human at inference).
- Long-tail handling: training uses a WeightedRandomSampler, weight prop to (1/count)^alpha (sqrt
  sampling); the head is not deleted, rare classes are oversampled; negatives are weighted to a target fraction.
"""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import cv2

from ..config import CLS_IMGSZ, MANIFEST, NEG_DIR, SAMPLE_ALPHA, NEG_FRACTION
from .augment import augment_patch


def _resize_pad(img, size):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), np.uint8)
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    r = cv2.resize(img, (nw, nh))
    out = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = r
    return out


class RealCropDataset(Dataset):
    def __init__(self, class_space, split="train", train=True,
                 manifest=MANIFEST, neg_dir=NEG_DIR):
        self.cs = class_space
        self.train = train
        self.samples = []          # (path, idx)
        # positives (active classes only)
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                kb = int(row["kb_id"])
                if self.cs.is_active(kb):
                    self.samples.append((row["path"], self.cs.to_idx(kb)))
        # negatives (split-agnostic; merged in as needed)
        if Path(neg_dir).exists():
            for p in sorted(Path(neg_dir).glob("*.jpg")):
                self.samples.append((str(p), 0))   # not_a_light

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, idx = self.samples[i]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((CLS_IMGSZ, CLS_IMGSZ, 3), np.uint8)
        img = _resize_pad(img, CLS_IMGSZ)
        if self.train:
            img = augment_patch(img)
        x = torch.from_numpy(img[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        return x, idx

    # ---- class-balanced sampling weights ----
    def make_sampler(self):
        labels = np.array([idx for _, idx in self.samples])
        counts = np.bincount(labels, minlength=self.cs.num_classes).astype(float)
        counts[counts == 0] = 1
        # sqrt sampling: weight prop to (1/count)^alpha
        cls_w = (1.0 / counts) ** SAMPLE_ALPHA
        # negative target fraction: scale idx0's total weight to NEG_FRACTION
        cls_w = cls_w / cls_w.sum()
        if counts[0] > 0:
            pos_mass = cls_w[1:].sum()
            cls_w[0] = pos_mass * NEG_FRACTION / (1 - NEG_FRACTION)
        sample_w = cls_w[labels]
        return WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                     num_samples=len(self.samples), replacement=True)

    def class_counts(self):
        labels = np.array([idx for _, idx in self.samples])
        return np.bincount(labels, minlength=self.cs.num_classes)
