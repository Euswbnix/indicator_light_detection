"""Stage2 分类器数据集：真实裁剪 patch + 类别均衡采样（无合成正样本）。

- 正样本：build_datasets.py 产出的真实裁剪（manifest.csv）。
- 负样本：negatives/ 目录（难负样本挖掘 + 随机非灯）→ idx 0 = not_a_light。
- 不在 ClassSpace 里的稀有类样本：丢弃（不进模型，推理时转人工）。
- 长尾处理：训练时用 WeightedRandomSampler，权重 ∝ (1/count)^alpha（平方根采样），
  头部不删、稀有超采；负样本按目标占比加权。
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
        # 正样本（仅 active 类）
        with open(manifest) as f:
            for row in csv.DictReader(f):
                if row["split"] != split:
                    continue
                kb = int(row["kb_id"])
                if self.cs.is_active(kb):
                    self.samples.append((row["path"], self.cs.to_idx(kb)))
        # 负样本（仅训练/验证各自需要时；按 split 不区分，统一并入）
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

    # ---- 类别均衡采样权重 ----
    def make_sampler(self):
        labels = np.array([idx for _, idx in self.samples])
        counts = np.bincount(labels, minlength=self.cs.num_classes).astype(float)
        counts[counts == 0] = 1
        # 平方根采样：权重 ∝ (1/count)^alpha
        cls_w = (1.0 / counts) ** SAMPLE_ALPHA
        # 负样本目标占比：把 idx0 的总权重缩放到 NEG_FRACTION
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
