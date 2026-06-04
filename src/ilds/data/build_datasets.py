"""Annotations -> two training sets (YOLO detector + ViT classifier crops).

Tool-agnostic: input supports COCO JSON and Pascal VOC XML (auto-detected). Both resolve the
box "category name" to a kb_id (a number string or a knowledge-base Chinese name).
- COCO: exported by CVAT / Label Studio / Roboflow, etc.
- Pascal VOC: supported by almost every tool (LabelImg / makesense / VoTT...); the universal fallback.

Outputs:
  datasets/detector/{images,labels}/{train,val} + data.yaml   (all classes -> 0, single class)
  datasets/classifier/crops/<kb_id>/*.jpg + manifest.csv      (per-box crop, 1.5x expansion)
  datasets/classifier/active_classes.json                     (classes with train samples >= MIN)

Split is done per image so boxes from one image never straddle train/val.

Usage:
  build_datasets.py --ann path/to/annotations.json --images IMAGE_DIR    # COCO
  build_datasets.py --ann path/to/voc_xml_dir       --images IMAGE_DIR    # Pascal VOC
"""
import argparse, json, csv, random, shutil
from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import cv2

from .. import config
from ..taxonomy import KB_BY_ID
from ..class_space import ClassSpace

_NAME_TO_KB = {c["name"]: kb for kb, c in KB_BY_ID.items()}


def resolve_kb(label):
    """Label name -> kb_id. Accepts a number string or a knowledge-base Chinese name.
    unknown / empty / 0 / out of 1-123 / unmatched name -> None (treated as "unknown": detector only)."""
    s = str(label).strip()
    if s.isdigit():
        n = int(s)
        return n if n in KB_BY_ID else None     # invalid ids like 0, 999 -> unknown
    return _NAME_TO_KB.get(s)                    # "未知"(unknown), garbage names -> None


# ---- Parse into a unified intermediate: {file_name: (W, H, [(x,y,w,h,kb_id), ...])} ----

def parse_coco(ann_path):
    coco = json.load(open(ann_path, encoding="utf-8"))
    cat = {c["id"]: resolve_kb(c["name"]) for c in coco["categories"]}
    recs = {}
    for im in coco["images"]:
        recs[im["file_name"]] = (im["width"], im["height"], [])
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    for a in coco["annotations"]:
        kb = cat.get(a["category_id"])      # unknown / out-of-KB -> None (kept, detector only)
        x, y, w, h = a["bbox"]
        recs[by_id[a["image_id"]]][2].append((x, y, w, h, kb))
    return recs


def parse_voc(voc_dir):
    recs = {}
    for xml in sorted(Path(voc_dir).glob("*.xml")):
        t = ET.parse(xml).getroot()
        fname = t.findtext("filename") or (xml.stem + ".jpg")
        size = t.find("size")
        W = int(size.findtext("width")); H = int(size.findtext("height"))
        boxes = []
        for obj in t.findall("object"):
            kb = resolve_kb(obj.findtext("name"))   # unknown / out-of-KB -> None (kept, detector only)
            b = obj.find("bndbox")
            x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
            x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
            boxes.append((x1, y1, x2 - x1, y2 - y1, kb))
        recs[fname] = (W, H, boxes)
    return recs


def load_annotations(ann_path):
    p = Path(ann_path)
    if p.is_file() and p.suffix.lower() == ".json":
        print("detected COCO JSON")
        return parse_coco(p)
    if p.is_dir() and any(p.glob("*.xml")):
        print("detected Pascal VOC (XML dir)")
        return parse_voc(p)
    raise SystemExit(f"unrecognized annotation format: {ann_path} (need COCO .json or a VOC dir with .xml)")


# ---- Common build ----

def build(recs, images_dir, seed=0):
    random.seed(seed)
    images_dir = Path(images_dir)
    names = [fn for fn, (_, _, boxes) in recs.items() if boxes]
    random.shuffle(names)
    n_val = int(len(names) * config.VAL_RATIO)
    val = set(names[:n_val])

    for d in [config.DET_DS / "images" / "train", config.DET_DS / "images" / "val",
              config.DET_DS / "labels" / "train", config.DET_DS / "labels" / "val",
              config.CROPS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = []
    train_counts = Counter()
    miss = 0
    for fn in names:
        W, H, boxes = recs[fn]
        split = "val" if fn in val else "train"
        src = images_dir / fn
        if not src.exists():
            miss += 1; continue
        img = cv2.imread(str(src))
        # YOLO detection (class 0) -- all boxes go in, incl. unknown/out-of-KB (they are "lights" too)
        shutil.copy2(src, config.DET_DS / "images" / split / Path(fn).name)
        lines = [f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}"
                 for (x, y, w, h, kb) in boxes]
        (config.DET_DS / "labels" / split / (Path(fn).stem + ".txt")).write_text("\n".join(lines))
        # ViT crops (1.5x expansion) -- only crop those with a kb_id; unknown/out-of-KB -> review dir, not the classifier
        for k, (x, y, w, h, kb) in enumerate(boxes):
            cx, cy = x + w / 2, y + h / 2
            ew, eh = w * config.CLS_CROP_EXPAND, h * config.CLS_CROP_EXPAND
            x0, y0 = max(0, int(cx - ew / 2)), max(0, int(cy - eh / 2))
            x1, y1 = min(W, int(cx + ew / 2)), min(H, int(cy + eh / 2))
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            if kb is None:                                  # unknown / out-of-KB
                rev = config.CLS_DS / "unknown_review"; rev.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(rev / f"{Path(fn).stem}_{k}.jpg"), crop)
                continue
            outdir = config.CROPS_DIR / str(kb); outdir.mkdir(exist_ok=True)
            cp = outdir / f"{Path(fn).stem}_{k}.jpg"
            cv2.imwrite(str(cp), crop)
            manifest.append([str(cp), kb, split, Path(fn).name])
            if split == "train":
                train_counts[kb] += 1

    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["path", "kb_id", "split", "source"]); w.writerows(manifest)
    cs = ClassSpace.from_counts(train_counts, config.MIN_SAMPLES); cs.save(config.ACTIVE_CLASSES)
    (config.DET_DS / "data.yaml").write_text(
        f"path: {config.DET_DS}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['indicator_light']\n")

    print(f"images {len(names)} (val {len(val)}, missing {miss}) | crops {len(manifest)} | train classes {len(train_counts)}")
    print(f"in model (>= {config.MIN_SAMPLES}): {cs.num_classes-1} classes -> {cs.kb_ids}")
    excluded = sorted(kb for kb in train_counts if not cs.is_active(kb))
    print(f"too few samples (-> human): {len(excluded)} classes -> {excluded}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO .json or a Pascal VOC .xml directory")
    ap.add_argument("--images", required=True, help="image directory")
    ap.add_argument("--min-samples", type=int, default=None)
    a = ap.parse_args()
    if a.min_samples is not None:
        config.MIN_SAMPLES = a.min_samples
    build(load_annotations(a.ann), a.images)
