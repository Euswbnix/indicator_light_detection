"""一份 COCO 标注 → 派生两个训练集（YOLO 检测 + ViT 分类裁剪）。

输入：CVAT / Label Studio 导出的 COCO JSON（标注了 box + 类别）。
约定：COCO category 的 name 是 kb_id 数字字符串（如 "120"）或知识库中文名。

产出：
  datasets/detector/{images,labels}/{train,val} + data.yaml   （类别全归 0，单类）
  datasets/classifier/crops/<kb_id>/*.jpg + manifest.csv      （按框裁剪，外扩1.5×）
  datasets/classifier/active_classes.json                     （训练集样本>=MIN的类）

划分按“图”进行，保证同一张图的框不跨 train/val。
"""
import argparse, json, csv, random, shutil
from pathlib import Path
from collections import defaultdict, Counter
import cv2

from .. import config
from ..taxonomy import KB_BY_ID, KB_IDS
from ..class_space import ClassSpace

_NAME_TO_KB = {c["name"]: kb for kb, c in KB_BY_ID.items()}


def resolve_kb(cat_name):
    """COCO 类别名 -> kb_id。支持数字串或中文名。"""
    s = str(cat_name).strip()
    if s.isdigit():
        return int(s)
    if s in _NAME_TO_KB:
        return _NAME_TO_KB[s]
    return None    # 未知/未匹配（如 '未知'）→ 丢弃，不进训练


def build(coco_json, images_dir, seed=0):
    random.seed(seed)
    coco = json.load(open(coco_json, encoding="utf-8"))
    images_dir = Path(images_dir)

    img_by_id = {im["id"]: im for im in coco["images"]}
    cat_kb = {c["id"]: resolve_kb(c["name"]) for c in coco["categories"]}

    # 按图聚合标注
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        kb = cat_kb.get(a["category_id"])
        if kb is None:
            continue
        anns_by_img[a["image_id"]].append((a["bbox"], kb))   # bbox=[x,y,w,h]

    img_ids = [i for i in img_by_id if i in anns_by_img]
    random.shuffle(img_ids)
    n_val = int(len(img_ids) * config.VAL_RATIO)
    val_ids = set(img_ids[:n_val])

    # 清理/建目录
    for d in [config.DET_DS / "images" / "train", config.DET_DS / "images" / "val",
              config.DET_DS / "labels" / "train", config.DET_DS / "labels" / "val",
              config.CROPS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    train_counts = Counter()

    for iid in img_ids:
        im = img_by_id[iid]
        split = "val" if iid in val_ids else "train"
        src = images_dir / im["file_name"]
        if not src.exists():
            print("缺图:", src); continue
        W, H = im["width"], im["height"]
        image = cv2.imread(str(src))

        # ---- YOLO 检测集（类别全 0）----
        shutil.copy2(src, config.DET_DS / "images" / split / src.name)
        lines = []
        for (x, y, w, h), kb in anns_by_img[iid]:
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"0 {cx:.6f} {cy:.6f} {w/W:.6f} {h/H:.6f}")
        (config.DET_DS / "labels" / split / (src.stem + ".txt")).write_text("\n".join(lines))

        # ---- ViT 分类裁剪（外扩 1.5×）----
        for k, ((x, y, w, h), kb) in enumerate(anns_by_img[iid]):
            cx, cy = x + w / 2, y + h / 2
            ew, eh = w * config.CLS_CROP_EXPAND, h * config.CLS_CROP_EXPAND
            x0, y0 = max(0, int(cx - ew / 2)), max(0, int(cy - eh / 2))
            x1, y1 = min(W, int(cx + ew / 2)), min(H, int(cy + eh / 2))
            crop = image[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            outdir = config.CROPS_DIR / str(kb)
            outdir.mkdir(exist_ok=True)
            cp = outdir / f"{src.stem}_{k}.jpg"
            cv2.imwrite(str(cp), crop)
            manifest_rows.append([str(cp), kb, split, src.name])
            if split == "train":
                train_counts[kb] += 1

    # manifest
    with open(config.MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "kb_id", "split", "source"])
        w.writerows(manifest_rows)

    # 数据驱动的 active 类集
    cs = ClassSpace.from_counts(train_counts, config.MIN_SAMPLES)
    cs.save()

    print(f"图 {len(img_ids)}（val {len(val_ids)}） | 裁剪 {len(manifest_rows)} | 训练集类别 {len(train_counts)}")
    print(f"进模型(>= {config.MIN_SAMPLES}): {cs.num_classes-1} 类 -> {cs.kb_ids}")
    excluded = [kb for kb in train_counts if not cs.is_active(kb)]
    print(f"样本不足(转人工): {len(excluded)} 类 -> {sorted(excluded)}")

    # detector data.yaml
    (config.DET_DS / "data.yaml").write_text(
        f"path: {config.DET_DS}\ntrain: images/train\nval: images/val\n"
        f"nc: 1\nnames: ['indicator_light']\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", required=True, help="COCO 标注 json")
    ap.add_argument("--images", required=True, help="图片目录")
    ap.add_argument("--min-samples", type=int, default=None)
    a = ap.parse_args()
    if a.min_samples is not None:
        config.MIN_SAMPLES = a.min_samples
    build(a.coco, a.images)
