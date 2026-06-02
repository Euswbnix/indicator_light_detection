"""训练 Stage2 MobileViT-XXS 分类器（真实裁剪 + 类别均衡采样）。

前置：先跑 build_datasets.py 生成 crops/ + manifest.csv + active_classes.json。
用法： ~/anaconda3/bin/python -m src.ilds.train.train_classifier --epochs 50 --focal

长尾已在数据层被删的头部保留、训练层用均衡采样+Focal压制：
- WeightedRandomSampler（平方根采样，头部不删、稀有超采）
- FocalLoss（难样本/稀有类权重更高）
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .. import config
from ..class_space import ClassSpace
from ..models.classifier import build_classifier
from ..data.classifier_dataset import RealCropDataset


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logp = torch.log_softmax(logits, dim=1)
        logp_t = logp.gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    config.WEIGHTS.mkdir(exist_ok=True)
    cs = ClassSpace.load()
    print(f"分类器支持 {cs.num_classes-1} 个灯类 + not_a_light")

    train_ds = RealCropDataset(cs, split="train", train=True)
    val_ds = RealCropDataset(cs, split="val", train=False)
    print(f"训练 {len(train_ds)} patch | 验证 {len(val_ds)} patch")
    print("各类训练样本数:", train_ds.class_counts().tolist())

    sampler = train_ds.make_sampler()
    train_dl = DataLoader(train_ds, batch_size=args.bs, sampler=sampler, num_workers=4, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.bs, num_workers=2)

    model = build_classifier(cs.num_classes).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    # 类别权重（逆频，给 Focal 用）
    counts = train_ds.class_counts().astype(float)
    counts[counts == 0] = 1
    cw = torch.tensor((counts.sum() / counts) ** 0.5, dtype=torch.float32, device=args.device)
    cw = cw / cw.mean()
    crit = FocalLoss(weight=cw) if args.focal else nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        run = 0.0
        for x, y in train_dl:
            x, y = x.to(args.device), y.to(args.device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run += loss.item()
        sched.step()

        # 验证：整体 + 宏平均（长尾下宏平均更诚实）
        model.eval()
        K = cs.num_classes
        per_correct = np.zeros(K); per_total = np.zeros(K)
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(args.device)
                pred = model(x).argmax(1).cpu().numpy()
                for t, p in zip(y.numpy(), pred):
                    per_total[t] += 1
                    per_correct[t] += (t == p)
        seen = per_total > 0
        micro = per_correct.sum() / max(1, per_total.sum())
        macro = (per_correct[seen] / per_total[seen]).mean()
        print(f"== ep{ep}: loss {run/len(train_dl):.3f}  micro_acc {micro:.4f}  macro_acc {macro:.4f}")
        if macro > best:
            best = macro
            torch.save({"model": model.state_dict(), "kb_ids": cs.kb_ids},
                       config.WEIGHTS / "classifier_best.pt")
            print(f"   saved best (macro {macro:.4f})")
    print(f"完成，最佳 macro_acc {best:.4f}")


if __name__ == "__main__":
    main()
