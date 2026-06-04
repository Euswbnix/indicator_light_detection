"""SAHI-style slicing: cut a large image into overlapping tiles so small lights are effectively
enlarged inside each tile, improving recall.

- The overlap band guarantees a light sitting on a seam is fully contained in at least one tile.
- Also run once on the full image to keep the large central icon (not cut off).
- Each tile's local coords are mapped back to global image coords; a single DIoU-NMS dedups at the end.
"""
import numpy as np


def plan_tiles(W, H, tile, overlap_ratio):
    """Return [(x0,y0,x1,y1), ...] covering the whole image, adjacent tiles overlapping by overlap_ratio."""
    step = max(1, int(tile * (1 - overlap_ratio)))
    xs = list(range(0, max(1, W - tile + 1), step))
    ys = list(range(0, max(1, H - tile + 1), step))
    if not xs or xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    if not ys or ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


def sliced_predict(image_bgr, predict_fn, tile=640, overlap=0.25, full_pass=True):
    """
    image_bgr: np.ndarray HxWx3
    predict_fn: callable(crop_bgr) -> (boxes_xyxy[N,4] in local coords, scores[N])
    Returns: global-coord boxes[N,4], scores[N] (no NMS; merging is left to the caller).
    """
    H, W = image_bgr.shape[:2]
    all_boxes, all_scores = [], []

    for (x0, y0, x1, y1) in plan_tiles(W, H, tile, overlap):
        crop = image_bgr[y0:y1, x0:x1]
        b, s = predict_fn(crop)
        if len(b):
            b = np.asarray(b, dtype=float).copy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            all_boxes.append(b)
            all_scores.append(np.asarray(s, dtype=float))

    if full_pass:
        b, s = predict_fn(image_bgr)
        if len(b):
            all_boxes.append(np.asarray(b, dtype=float))
            all_scores.append(np.asarray(s, dtype=float))

    if not all_boxes:
        return np.zeros((0, 4)), np.zeros((0,))
    return np.concatenate(all_boxes, 0), np.concatenate(all_scores, 0)
