"""训练 Stage1 YOLOv8s-P2 单类检测器。

前置：datasets/detector/ 下需有 YOLO 格式（images/{train,val} + labels/{train,val}），
所有标注类别 id 均为 0（单类 indicator_light）。见 datasets/detector/data.yaml。

用法： ~/anaconda3/bin/python -m src.ilds.train.train_detector --epochs 100
"""
import argparse
from ultralytics import YOLO
from .. import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--data", default=str(config.DET_DS / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=config.DET_IMGSZ)
    ap.add_argument("--bs", type=int, default=8)        # 1280 输入显存占用大
    args = ap.parse_args()

    model = YOLO(str(config.DET_MODEL_YAML))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.bs,
        project=str(config.RUNS),
        name="detector_p2",
        # 小目标友好的增强：保守缩放，避免把小灯缩没
        scale=0.3, mosaic=1.0, close_mosaic=15,
        hsv_h=0.0,           # 检测不需要颜色抖动（颜色判别交给 Stage2）
        single_cls=True,
    )


if __name__ == "__main__":
    main()
