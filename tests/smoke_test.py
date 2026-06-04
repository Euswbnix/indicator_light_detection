"""Pipeline smoke test -- no trained weights, no real data needed.

Generates fake data and verifies the whole code pipeline runs end to end
(tests the plumbing, not accuracy).

Usage:
    python tests/smoke_test.py

Prints a check / cross per step, then a summary. All checks passing = the code is fine.
All artifacts are written to a temp dir and deleted afterwards; the real datasets/ is untouched.
"""
import sys, json, tempfile, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OK, BAD = [], []
def check(name, cond):
    (OK if cond else BAD).append(name)
    print(f"   {'[OK]' if cond else '[FAIL]'} {name}")


def main():
    print("\n========== indicator-light pipeline smoke test ==========\n")

    print("[1] module imports")
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
        check("import all modules", True)
    except Exception as e:
        check(f"import failed: {e}", False)
        print("\nCannot continue. Make sure you run with a Python 3.9 env (torch + ultralytics + cv2).")
        return 1

    import torch, cv2

    print("[2] classifier forward")
    y = build_classifier(124)(torch.randn(2, 3, 128, 128))
    check("single-head MobileViT-XXS output (2,124)", tuple(y.shape) == (2, 124))
    sh, co = build_twohead(taxonomy.NUM_SHAPE_FAMILIES, taxonomy.NUM_COLORS)(torch.randn(2, 3, 128, 128))
    check("two-head (shape/color) forward", sh.shape[0] == 2 and co.shape[1] == taxonomy.NUM_COLORS)

    print("[3] detector instantiation")
    det = YOLO(str(ROOT / "src/ilds/models/yolov8s-p2.yaml"))
    nl = next(m.nl for m in det.model.modules() if isinstance(m, Detect))
    check("YOLOv8s-P2 detect head has 4 scales (incl. P2)", nl == 4)

    print("[4] NMS / tiling")
    keep = diou_nms(np.array([[10, 10, 30, 30], [32, 10, 52, 30], [11, 11, 31, 31]]),
                    np.array([.9, .85, .8]), 0.55)
    check("DIoU-NMS keeps adjacent small lights, drops duplicate", sorted(keep) == [0, 1])
    check("tiling plan 1920x1080 -> 8 tiles", len(plan_tiles(1920, 1080, 640, 0.25)) == 8)

    print("[5] data pipeline (fake annotation -> build_datasets)")
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
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [200, 200, 30, 30]},  # unknown ("未知")
        ],
    }
    aj = tmp / "a.json"; json.dump(coco, open(aj, "w"), ensure_ascii=False)
    from src.ilds.data.build_datasets import load_annotations, build
    build(load_annotations(aj), imgd)

    det_lbl = (config.DET_DS / "labels/train/t.txt").read_text().strip().split("\n")
    ncrops = len(list(config.CROPS_DIR.rglob("*.jpg")))
    nrev = len(list((config.CLS_DS / "unknown_review").glob("*.jpg"))) \
        if (config.CLS_DS / "unknown_review").exists() else 0
    check("detector gets 3 boxes (incl. unknown)", len(det_lbl) == 3)
    check("classifier crops only 2 (labeled ones)", ncrops == 2)
    check("unknown box saved to review", nrev == 1)
    from src.ilds.class_space import ClassSpace
    cs = ClassSpace.load(config.ACTIVE_CLASSES)
    check("active_classes == {84,120}", set(cs.kb_ids) == {84, 120})

    print("[6] end-to-end inference (untrained models, plumbing only)")
    from src.ilds.inference.pipeline import TwoStagePipeline
    from src.ilds.models.classifier import build_classifier
    pipe = TwoStagePipeline(det, build_classifier(cs.num_classes), cs, device="cpu")
    results, post = pipe(cv2.imread(str(imgd / "t.jpg")))
    check("full pipeline returns (list, dict)", isinstance(results, list) and isinstance(post, dict))

    shutil.rmtree(tmp)

    print(f"\n========== result: {len(OK)} passed, {len(BAD)} failed ==========")
    if BAD:
        print("failed:", BAD)
        return 1
    print("All checks passed. The pipeline code is sound; ready to train once data is in.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
