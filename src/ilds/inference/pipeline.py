"""Two-stage end-to-end inference: detect -> slice/merge/dedup -> crop -> classify -> reject -> post-process.

Output per light: bbox, kb_id, name, color, det_conf, cls_conf.
Boxes that are not_a_light or have cls_conf < threshold are dropped (false-positive fallback).
"""
import numpy as np
import torch
import torch.nn.functional as F

from .. import config
from ..taxonomy import KB_BY_ID, TURN_LEFT, TURN_RIGHT
from ..class_space import ClassSpace
from .tiling import sliced_predict
from .nms import diou_nms


class TwoStagePipeline:
    def __init__(self, detector, classifier, class_space: ClassSpace, device="cpu"):
        """
        detector: ultralytics YOLO model (single class).
        classifier: torch MobileViTXXS (eval mode), output dim = class_space.num_classes.
        class_space: classes this model version supports (active). Lights outside it fall back to
            low confidence -> routed to human.
        """
        self.det = detector
        self.cls = classifier.to(device).eval()
        self.cs = class_space
        self.device = device

    # ---- Stage1: detect within a single tile ----
    def _detect_crop(self, crop_bgr):
        r = self.det.predict(crop_bgr, imgsz=config.TILE, conf=config.DET_CONF,
                             verbose=False)[0]
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
            patch = _resize_pad(patch, config.CLS_IMGSZ)
            patches.append(patch)
        if not patches:
            return np.zeros((0,), int), np.zeros((0,))
        x = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).float() / 255.0
        x = x.to(self.device)
        logits = self.cls(x)
        prob = F.softmax(logits, dim=1)
        conf, idx = prob.max(dim=1)
        return idx.cpu().numpy(), conf.cpu().numpy()

    def __call__(self, image_bgr):
        # 1. sliced detection + full-image pass
        boxes, scores = sliced_predict(
            image_bgr, self._detect_crop,
            tile=config.TILE, overlap=config.TILE_OVERLAP, full_pass=config.DO_FULL_PASS)
        # 2. global DIoU-NMS dedup (overlap-band duplicates, large/small duplicates)
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
        # 5. post-process (geometry relations such as hazard lights)
        post = self._postprocess(results)
        return results, post

    def _postprocess(self, results):
        flags = {}
        kbs = {r["kb_id"] for r in results}
        if TURN_LEFT in kbs and TURN_RIGHT in kbs:
            flags["hazard"] = True   # left + right turn signals on together = hazard lights
        return flags


def _resize_pad(img, size):
    """Aspect-preserving resize, then center-pad to size x size."""
    import cv2
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
