"""Two-stage end-to-end inference: detect -> (optional slice) -> NMS -> crop -> classify -> reject -> post.

Output per light: bbox, kb_id, name, color, det_conf, cls_conf.
Boxes that are not_a_light or have cls_conf < threshold are dropped (false-positive fallback).

Detection runs on the full image at DET_IMGSZ by default (this matches how the detector is
trained/validated). SAHI-style tiling is available via tile=True for cases where the detector
genuinely misses tiny lights at full resolution.
"""
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from .. import config
from ..taxonomy import KB_BY_ID, TURN_LEFT, TURN_RIGHT
from ..class_space import ClassSpace
from .tiling import sliced_predict
from .nms import diou_nms


class TwoStagePipeline:
    def __init__(self, detector, classifier, class_space: ClassSpace, device="cpu", tile=False):
        """
        detector: ultralytics YOLO model (single class).
        classifier: torch MobileViTXXS (eval mode), output dim = class_space.num_classes.
        class_space: classes this model version supports (active).
        tile: if True, use SAHI sliced inference; else full-image detection at DET_IMGSZ (default).
        """
        self.det = detector
        self.cls = classifier.to(device).eval()
        self.cs = class_space
        self.device = device
        self.tile = tile

    # ---- Stage1: run the detector on an image at a given input size ----
    def _detect(self, img_bgr, imgsz):
        r = self.det.predict(img_bgr, imgsz=imgsz, conf=config.DET_CONF, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros((0,))
        return r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()

    # ---- Stage2: batch-classify cropped patches ----
    @torch.no_grad()
    def _classify(self, image_bgr, boxes):
        H, W = image_bgr.shape[:2]
        patches = []
        for (x0, y0, x1, y1) in boxes:
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            bw, bh = (x1 - x0) * config.CLS_CROP_EXPAND, (y1 - y0) * config.CLS_CROP_EXPAND
            ex0, ey0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
            ex1, ey1 = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
            patch = image_bgr[ey0:ey1, ex0:ex1]
            patches.append(_resize_pad(patch, config.CLS_IMGSZ))
        if not patches:
            return np.zeros((0,), int), np.zeros((0,))
        # BGR -> RGB to match training (classifier_dataset flips channels); critical!
        arr = np.stack(patches)[:, :, :, ::-1]
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(0, 3, 1, 2).float() / 255.0
        x = x.to(self.device)
        prob = F.softmax(self.cls(x), dim=1)
        conf, idx = prob.max(dim=1)
        return idx.cpu().numpy(), conf.cpu().numpy()

    def __call__(self, image_bgr):
        # 1. detection
        if self.tile:
            boxes, scores = sliced_predict(
                image_bgr, lambda c: self._detect(c, config.TILE),
                tile=config.TILE, overlap=config.TILE_OVERLAP, full_pass=config.DO_FULL_PASS)
        else:
            boxes, scores = self._detect(image_bgr, config.DET_IMGSZ)   # full image @ 1280
        # 2. NMS dedup
        if len(boxes):
            keep = diou_nms(boxes, scores, iou_thresh=config.NMS_IOU)
            boxes, scores = boxes[keep], scores[keep]
        # 3. classify
        cls_idx, cls_conf = self._classify(image_bgr, boxes)
        # 4. reject fallback
        results = []
        for box, dconf, ci, cconf in zip(boxes, scores, cls_idx, cls_conf):
            if ci == 0 or cconf < config.CLS_REJECT_THRESH:   # not_a_light or unsure -> human
                continue
            kb = int(self.cs.idx_to_kb[ci])
            results.append(dict(
                bbox=[float(v) for v in box], kb_id=kb,
                name=KB_BY_ID[kb]["name"], color=KB_BY_ID[kb]["color"],
                det_conf=float(dconf), cls_conf=float(cconf)))
        # 5. post-process (geometry: e.g. hazard = left + right turn together)
        post = self._postprocess(results)
        return results, post

    def _postprocess(self, results):
        flags = {}
        kbs = {r["kb_id"] for r in results}
        if TURN_LEFT in kbs and TURN_RIGHT in kbs:
            flags["hazard"] = True
        return flags


def _resize_pad(img, size):
    """Aspect-preserving resize, then center-pad to size x size."""
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size, 3), np.uint8)
    s = size / max(h, w)
    nh, nw = max(1, int(h * s)), max(1, int(w * s))
    r = cv2.resize(img, (nw, nh))
    out = np.zeros((size, size, 3), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = r
    return out
