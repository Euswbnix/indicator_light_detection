"""标注 → 两个训练集（YOLO 检测 + ViT 分类裁剪）。

兼容多种工具：输入支持 COCO JSON 与 Pascal VOC XML（自动识别），
两者都把"框的类别名"解析成 kb_id（数字串或知识库中文名）。
- COCO：CVAT / Label Studio / Roboflow 等导出。
- Pascal VOC：几乎所有工具都支持（LabelImg / makesense / VoTT…），最通用的兜底。

产出：
  datasets/detector/{images,labels}/{train,val} + data.yaml   （类别全归 0，单类）
  datasets/classifier/crops/<kb_id>/*.jpg + manifest.csv      （按框裁剪，外扩1.5×）
  datasets/classifier/active_classes.json                     （训练集样本>=MIN的类）

划分按"图"进行，保证同一张图的框不跨 train/val。

用法：
  build_datasets.py --ann path/to/annotations.json --images 图片目录       # COCO
  build_datasets.py --ann path/to/voc_xml_dir       --images 图片目录       # Pascal VOC
"""
import argparse, json, csv, random, shutil
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
import cv2

from .. import config
from ..taxonomy import KB_BY_ID
from ..class_space import ClassSpace

_NAME_TO_KB = {c["name"]: kb for kb, c in KB_BY_ID.items()}


def resolve_kb(label):
    """标签名 -> kb_id。支持数字串或知识库中文名。
    未知 / 留空 / 0 / 超出 1-123 / 对不上名字 -> None（按"未知"处理：仅喂检测器）。"""
    s = str(label).strip()
    if s.isdigit():
        n = int(s)
        return n if n in KB_BY_ID else None     # 0、999 等无效号 -> 未知
    return _NAME_TO_KB.get(s)                    # '未知'、乱名 -> None


# ---- 解析为统一中间结构: {file_name: (W, H, [(x,y,w,h,kb_id), ...])} ----

def parse_coco(ann_path):
    coco = json.load(open(ann_path, encoding="utf-8"))
    img = {im["id"]: im for im in coco["images"]}
    cat = {c["id"]: resolve_kb(c["name"]) for c in coco["categories"]}
    recs = {}
    for im in coco["images"]:
        recs[im["file_name"]] = (im["width"], im["height"], [])
    by_id = {im["id"]: im["file_name"] for im in coco["images"]}
    for a in coco["annotations"]:
        kb = cat.get(a["category_id"])      # 未知/库外 -> None（保留，仅喂检测器）
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
            kb = resolve_kb(obj.findtext("name"))   # 未知/库外 -> None（保留，仅喂检测器）
            b = obj.find("bndbox")
            x1 = float(b.findtext("xmin")); y1 = float(b.findtext("ymin"))
            x2 = float(b.findtext("xmax")); y2 = float(b.findtext("ymax"))
            boxes.append((x1, y1, x2 - x1, y2 - y1, kb))
        recs[fname] = (W, H, boxes)
    return recs


def load_annotations(ann_path):
    p = Path(ann_path)
    if p.is_file() and p.suffix.lower() == ".json":
        print("识别为 COCO JSON")
        return parse_coco(p)
    if p.is_dir() and any(p.glob("*.xml")):
        print("识别为 Pascal VOC（XML 目录）")
        return parse_voc(p)
    raise SystemExit(f"无法识别标注格式：{ann_path}（需 COCO .json 或含 .xml 的 VOC 目录）")


# ---- 通用构建 ----

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
        # YOLO 检测（类别 0）—— 所有框都进，含未知/库外（它们也是"灯"）
        shutil.copy2(src, config.DET_DS / "images" / split / Path(fn).name)
        lines = [f"0 {(x+w/2)/W:.6f} {(y+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}"
                 for (x, y, w, h, kb) in boxes]
        (config.DET_DS / "labels" / split / (Path(fn).stem + ".txt")).write_text("\n".join(lines))
        # ViT 裁剪（外扩 1.5×）—— 只裁有编号的；未知/库外另存 review，不进分类器
        for k, (x, y, w, h, kb) in enumerate(boxes):
            cx, cy = x + w / 2, y + h / 2
            ew, eh = w * config.CLS_CROP_EXPAND, h * config.CLS_CROP_EXPAND
            x0, y0 = max(0, int(cx - ew / 2)), max(0, int(cy - eh / 2))
            x1, y1 = min(W, int(cx + ew / 2)), min(H, int(cy + eh / 2))
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            if kb is None:                                  # 未知/库外
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

    print(f"图 {len(names)}（val {len(val)}，缺图 {miss}） | 裁剪 {len(manifest)} | 训练集类别 {len(train_counts)}")
    print(f"进模型(>= {config.MIN_SAMPLES}): {cs.num_classes-1} 类 -> {cs.kb_ids}")
    excluded = sorted(kb for kb in train_counts if not cs.is_active(kb))
    print(f"样本不足(转人工): {len(excluded)} 类 -> {excluded}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True, help="COCO .json 或 Pascal VOC 的 .xml 目录")
    ap.add_argument("--images", required=True, help="图片目录")
    ap.add_argument("--min-samples", type=int, default=None)
    a = ap.parse_args()
    if a.min_samples is not None:
        config.MIN_SAMPLES = a.min_samples
    build(load_annotations(a.ann), a.images)
