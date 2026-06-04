"""Train the Stage1 YOLOv8s-P2 single-class detector.

Prerequisite: datasets/detector/ must hold YOLO format (images/{train,val} + labels/{train,val})
with every annotation class id = 0 (single class indicator_light). See datasets/detector/data.yaml.

Usage: python -m src.ilds.train.train_detector --epochs 100
"""
import argparse
from ultralytics import YOLO
from .. import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--data", default=str(config.DET_DS / "data.yaml"))
    ap.add_argument("--imgsz", type=int, default=config.DET_IMGSZ)
    ap.add_argument("--bs", type=int, default=8)        # 1280 input is memory-heavy
    args = ap.parse_args()

    model = YOLO(str(config.DET_MODEL_YAML))
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.bs,
        project=str(config.RUNS),
        name="detector_p2",
        # small-object-friendly augmentation: conservative scaling so small lights aren't shrunk away
        scale=0.3, mosaic=1.0, close_mosaic=15,
        hsv_h=0.0,           # no hue jitter for detection (color discrimination is Stage2's job)
        single_cls=True,
    )


if __name__ == "__main__":
    main()
