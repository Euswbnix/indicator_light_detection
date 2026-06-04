"""管线自测 —— 不需要训练权重、不需要真实数据。

自己造假数据，端到端验证整条代码管线能不能跑通（测水管，不测精度）。

用法：
    ~/anaconda3/bin/python tests/smoke_test.py

每步打 ✓/✗，最后给总结。全 ✓ = 管线代码没问题。
所有产物写到临时目录，跑完自动删，不碰真实 datasets/。
"""
import sys, json, tempfile, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, BAD = [], []
def check(name, cond):
    (OK if cond else BAD).append(name)
    print(f"   {'✓' if cond else '✗ 失败:'} {name}")


def main():
    print("\n========== 仪表灯检测 管线自测 ==========\n")

    print("[1] 模块导入")
    try:
        import torch, cv2  # noqa
        from ultralytics import YOLO
        from ultralytics.nn.modules import Detect
        from src.ilds import config, taxonomy
        from src.ilds.class_space import ClassSpace
        from src.ilds.models.classifier import build_classifier, build_twohead
        from src.ilds.inference.nms import diou_nms
        from src.ilds.inference.tiling import plan_tiles
        from src.ilds.inference.pipeline import TwoStagePipeline
        from src.ilds.data.build_datasets import load_annotations, build
        check("导入全部模块", True)
    except Exception as e:
        check(f"导入失败: {e}", False)
        print("\n无法继续。检查是否用 ~/anaconda3/bin/python 运行。")
        return 1

    import torch, cv2

    print("[2] 分类器前向")
    y = build_classifier(124)(torch.randn(2, 3, 128, 128))
    check("单头 MobileViT-XXS 输出 (2,124)", tuple(y.shape) == (2, 124))
    sh, co = build_twohead(taxonomy.NUM_SHAPE_FAMILIES, taxonomy.NUM_COLORS)(torch.randn(2, 3, 128, 128))
    check("双头(形状/颜色)前向", sh.shape[0] == 2 and co.shape[1] == taxonomy.NUM_COLORS)

    print("[3] 检测器实例化")
    det = YOLO(str(ROOT / "src/ilds/models/yolov8s-p2.yaml"))
    nl = next(m.nl for m in det.model.modules() if isinstance(m, Detect))
    check("YOLOv8s-P2 检测头 4 尺度(含 P2)", nl == 4)

    print("[4] NMS / 切片")
    keep = diou_nms(np.array([[10, 10, 30, 30], [32, 10, 52, 30], [11, 11, 31, 31]]),
                    np.array([.9, .85, .8]), 0.55)
    check("DIoU-NMS 相邻小灯保留、重复去除", sorted(keep) == [0, 1])
    check("切片规划 1920x1080→8 块", len(plan_tiles(1920, 1080, 640, 0.25)) == 8)

    print("[5] 数据管线(造假标注 → build_datasets)")
    tmp = Path(tempfile.mkdtemp())
    config.DET_DS = tmp / "det"
    config.CROPS_DIR = tmp / "cls/crops"
    config.CLS_DS = tmp / "cls"
    config.MANIFEST = tmp / "cls/manifest.csv"
    config.ACTIVE_CLASSES = tmp / "cls/active.json"
    config.MIN_SAMPLES = 1
    (tmp / "cls").mkdir(parents=True)
    imgd = tmp / "img"; imgd.mkdir()
    cv2.imwrite(str(imgd / "t.jpg"), np.random.randint(0, 255, (300, 400, 3), np.uint8))
    coco = {
        "images": [{"id": 1, "file_name": "t.jpg", "width": 400, "height": 300}],
        "categories": [{"id": 1, "name": "120"}, {"id": 2, "name": "84"}, {"id": 3, "name": "未知"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 40, 40]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 40, 40]},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [200, 200, 30, 30]},  # 未知
        ],
    }
    aj = tmp / "a.json"; json.dump(coco, open(aj, "w"), ensure_ascii=False)
    from src.ilds.data.build_datasets import load_annotations, build
    build(load_annotations(aj), imgd)

    det_lbl = (config.DET_DS / "labels/train/t.txt").read_text().strip().split("\n")
    ncrops = len(list(config.CROPS_DIR.rglob("*.jpg")))
    nrev = len(list((config.CLS_DS / "unknown_review").glob("*.jpg"))) \
        if (config.CLS_DS / "unknown_review").exists() else 0
    check("检测器拿到 3 个框(含未知)", len(det_lbl) == 3)
    check("分类器只裁 2 个(有编号)", ncrops == 2)
    check("未知框另存 review", nrev == 1)
    from src.ilds.class_space import ClassSpace
    cs = ClassSpace.load(config.ACTIVE_CLASSES)
    check("active_classes 生成 {84,120}", set(cs.kb_ids) == {84, 120})

    print("[6] 端到端推理(未训练模型，测水管)")
    from src.ilds.inference.pipeline import TwoStagePipeline
    from src.ilds.models.classifier import build_classifier
    pipe = TwoStagePipeline(det, build_classifier(cs.num_classes), cs, device="cpu")
    results, post = pipe(cv2.imread(str(imgd / "t.jpg")))
    check("完整管线返回 (list, dict)", isinstance(results, list) and isinstance(post, dict))

    shutil.rmtree(tmp)

    print(f"\n========== 结果: {len(OK)} 通过, {len(BAD)} 失败 ==========")
    if BAD:
        print("失败项:", BAD)
        return 1
    print("✅ 全部跑通。管线代码没问题，等数据到位即可训练。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
