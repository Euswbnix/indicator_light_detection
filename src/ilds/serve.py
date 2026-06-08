"""Inference HTTP server for the mini-program.

POST /api/recognize/image  (multipart, field "image")
  -> runs the two-stage pipeline, keeps only whitelist (product) fault lights,
     returns a LIST of detected lights.

Response:
{
  "code": 0, "msg": "success",
  "data": {
    "recognized": true,
    "count": 2,
    "lights": [
      {"lightId": "light-108", "lightName": "发动机故障警报灯",
       "color": "yellow", "colorText": "黄色", "confidence": 0.93,
       "bbox": {"x": 850, "y": 511, "w": 75, "h": 76}},
      ...
    ]
  }
}

Run:
  pip install fastapi uvicorn python-multipart
  python -m src.ilds.serve --det runs/detector_p2/weights/best.pt \
      --cls weights/classifier_best.pt --port 8000
"""
import argparse
import io

import numpy as np
import cv2
import torch
from ultralytics import YOLO

from . import config
from .taxonomy import KB_BY_ID, norm_color
from .class_space import ClassSpace
from .models.classifier import build_classifier
from .inference.pipeline import TwoStagePipeline

# Product whitelist: fault-light kb_ids the mini-program supports
# (mirror of dashboard-guide/images/lights/; keep in sync with the product).
WHITELIST = {
    5, 13, 16, 18, 19, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
    42, 44, 45, 46, 65, 66, 67, 68, 69, 70, 71, 74, 77, 78, 79, 80, 81, 89, 90,
    103, 104, 105, 106, 108, 109, 110, 115, 116, 117, 118, 119, 122, 123,
}


def build_app(det_path, cls_path, device="cpu"):
    from fastapi import FastAPI, File, UploadFile

    ckpt = torch.load(cls_path, map_location="cpu")
    cs = ClassSpace(ckpt["kb_ids"])
    clf = build_classifier(cs.num_classes)
    clf.load_state_dict(ckpt["model"])
    det = YOLO(det_path)
    pipe = TwoStagePipeline(det, clf, cs, device=device)

    app = FastAPI(title="indicator-light recognition")

    @app.get("/health")
    def health():
        return {"code": 0, "msg": "ok", "data": {"classes": cs.num_classes - 1}}

    @app.post("/api/recognize/image")
    async def recognize(image: UploadFile = File(...)):
        raw = await image.read()
        arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return {"code": 1001, "msg": "invalid image", "data": None}

        results, _post = pipe(arr)
        lights = []
        for r in results:
            kb = r["kb_id"]
            if kb not in WHITELIST:          # non-whitelist (turn signals, status...) -> filtered
                continue
            x0, y0, x1, y1 = r["bbox"]
            zh = KB_BY_ID[kb]["color"]
            lights.append({
                "lightId": f"light-{kb}",
                "lightName": KB_BY_ID[kb]["name"],
                "color": norm_color(zh),
                "colorText": zh,
                "confidence": round(r["cls_conf"], 4),
                "bbox": {"x": int(x0), "y": int(y0), "w": int(x1 - x0), "h": int(y1 - y0)},
            })
        lights.sort(key=lambda d: d["confidence"], reverse=True)
        return {"code": 0, "msg": "success",
                "data": {"recognized": len(lights) > 0, "count": len(lights), "lights": lights}}

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default="runs/detector_p2/weights/best.pt")
    ap.add_argument("--cls", default=str(config.WEIGHTS / "classifier_best.pt"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(build_app(args.det, args.cls, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
