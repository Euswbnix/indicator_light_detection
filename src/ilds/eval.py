"""End-to-end evaluation of the two-stage pipeline on the val split.

Runs the full pipeline (detector + tiling/NMS + classifier + reject) on each val
image, matches predictions to ground truth by IoU, and reports the metrics that
actually matter for the product:
  - detection recall   : GT lights found (any class)
  - cls acc            : of the found lights, fraction with the correct class
  - END-TO-END         : fraction of GT lights both found AND correctly classified
  - false positives    : predicted lights that match no GT (per image)

Usage:
  python -m src.ilds.eval --coco raw_data/export_2026-06-05-17/instances.json \
      --det runs/detector_p2/weights/best.pt --cls weights/classifier_best.pt
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import torch
from ultralytics import YOLO

from . import config
from .taxonomy import KB_BY_ID
from .class_space import ClassSpace
from .models.classifier import build_classifier
from .inference.pipeline import TwoStagePipeline


def iou(a, boxes):
    if len(boxes) == 0:
        return np.zeros((0,))
    x1 = np.maximum(a[0], boxes[:, 0]); y1 = np.maximum(a[1], boxes[:, 1])
    x2 = np.minimum(a[2], boxes[:, 2]); y2 = np.minimum(a[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (aa + ab - inter + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="original COCO json (has GT classes)")
    ap.add_argument("--det", default="runs/detector_p2/weights/best.pt")
    ap.add_argument("--cls", default=str(config.WEIGHTS / "classifier_best.pt"))
    ap.add_argument("--val-images", default=str(config.DET_DS / "images" / "val"))
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    val_dir = Path(args.val_images)
    val_names = {p.name for p in val_dir.glob("*.jpg")}

    # ground truth (boxes + kb_id) for val images, from the original COCO
    coco = json.load(open(args.coco, encoding="utf-8"))
    cat = {c["id"]: (int(c["name"]) if str(c["name"]).isdigit() else None) for c in coco["categories"]}
    img_by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = defaultdict(list)
    for a in coco["annotations"]:
        fn = img_by_id[a["image_id"]]
        if fn not in val_names:
            continue
        kb = cat.get(a["category_id"])
        if kb is None:
            continue
        x, y, w, h = a["bbox"]
        gt[fn].append(([x, y, x + w, y + h], kb))

    # models
    ckpt = torch.load(args.cls, map_location="cpu")
    cs = ClassSpace(ckpt["kb_ids"])
    clf = build_classifier(cs.num_classes)
    clf.load_state_dict(ckpt["model"])
    det = YOLO(args.det)
    pipe = TwoStagePipeline(det, clf, cs, device=args.device)

    n_gt = n_det = n_correct = n_fp = 0
    per = defaultdict(lambda: [0, 0, 0])   # kb -> [gt, detected, correct]

    for fn in sorted(val_names):
        img = cv2.imread(str(val_dir / fn))
        if img is None:
            continue
        results, _ = pipe(img)
        preds = [(np.array(r["bbox"], dtype=float), r["kb_id"]) for r in results]
        pred_boxes = np.array([p[0] for p in preds]) if preds else np.zeros((0, 4))
        used = [False] * len(preds)
        matched = set()
        for gbox, gkb in gt.get(fn, []):
            n_gt += 1; per[gkb][0] += 1
            if len(preds) == 0:
                continue
            ious = iou(np.array(gbox, dtype=float), pred_boxes)
            for j in np.argsort(ious)[::-1]:
                if ious[j] < args.iou:
                    break
                if used[j]:
                    continue
                used[j] = True; matched.add(int(j))
                n_det += 1; per[gkb][1] += 1
                if preds[j][1] == gkb:
                    n_correct += 1; per[gkb][2] += 1
                break
        n_fp += sum(1 for j in range(len(preds)) if j not in matched)

    print(f"\nval images {len(val_names)} | GT lights {n_gt}")
    print(f"detection recall          : {n_det/max(1,n_gt):.3f}  ({n_det}/{n_gt})")
    print(f"cls acc (of detected)     : {n_correct/max(1,n_det):.3f}")
    print(f"END-TO-END (found+correct): {n_correct/max(1,n_gt):.3f}")
    print(f"false positives           : {n_fp}  ({n_fp/max(1,len(val_names)):.2f}/image)")
    print("\nper-class (recall / end-to-end / n):")
    for kb in sorted(per):
        g, d, c = per[kb]
        print(f"  {kb:>3} {KB_BY_ID[kb]['name']:<18} recall {d/max(1,g):.2f}  e2e {c/max(1,g):.2f}  (n={g})")


if __name__ == "__main__":
    main()
