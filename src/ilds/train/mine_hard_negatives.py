"""难负样本挖掘：检测器自己的误检 → not_a_light 训练材料。

闭环：训好检测器 → 在标注图上低阈值跑 → 检出框与真实 bbox 做 IoU 比对 →
IoU≈0 的框就是误检（时钟/温度/反光/驾驶员小人...）→ 裁出来存为 not_a_light。
喂回分类器重训，越迭代拒绝能力越强。零人工标注。

用法： ~/anaconda3/bin/python -m src.ilds.train.mine_hard_negatives \
          --weights runs/detector_p2/weights/best.pt --out datasets/classifier/negatives
"""
import argparse
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

from .. import config


def iou_xyxy(a, boxes):
    if len(boxes) == 0:
        return np.zeros((0,))
    x1 = np.maximum(a[0], boxes[:, 0]); y1 = np.maximum(a[1], boxes[:, 1])
    x2 = np.minimum(a[2], boxes[:, 2]); y2 = np.minimum(a[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (aa + ab - inter + 1e-9)


def load_gt(label_path, W, H):
    """读 YOLO label（cls cx cy w h 归一化）→ xyxy 像素。"""
    if not label_path.exists():
        return np.zeros((0, 4))
    out = []
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cx, cy, w, h = map(float, p[1:5])
        out.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
    return np.array(out) if out else np.zeros((0, 4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", default=str(config.DET_DS / "images" / "train"))
    ap.add_argument("--labels", default=str(config.DET_DS / "labels" / "train"))
    ap.add_argument("--out", default=str(config.CLS_DS / "negatives"))
    ap.add_argument("--iou", type=float, default=0.1, help="低于此 IoU 视为误检")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    imgs = sorted(Path(args.images).glob("*.jpg"))
    n = 0
    for ip in imgs:
        img = cv2.imread(str(ip)); H, W = img.shape[:2]
        gt = load_gt(Path(args.labels) / (ip.stem + ".txt"), W, H)
        r = model.predict(img, imgsz=config.DET_IMGSZ, conf=config.DET_CONF, verbose=False)[0]
        if r.boxes is None:
            continue
        for box in r.boxes.xyxy.cpu().numpy():
            if len(gt) == 0 or iou_xyxy(box, gt).max() < args.iou:   # 不命中任何真灯 = 误检
                x0, y0, x1, y1 = [int(v) for v in box]
                crop = img[max(0, y0):y1, max(0, x0):x1]
                if crop.size:
                    cv2.imwrite(str(out / f"hn_{ip.stem}_{n}.jpg"), crop); n += 1
    print(f"挖出 {n} 个难负样本 → {out}")


if __name__ == "__main__":
    main()
