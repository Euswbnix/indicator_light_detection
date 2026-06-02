"""SAHI 风格切片：把大图切成带重叠的 tile，小灯在 tile 里等效放大，提升召回。

- 重叠带保证骑在切线上的灯总有一个 tile 完整包含它。
- 另跑一次全图，保住中央大图标（不会被切断）。
- 各 tile 的局部坐标映射回原图全局坐标，最后统一 DIoU-NMS 去重。
"""
import numpy as np


def plan_tiles(W, H, tile, overlap_ratio):
    """返回 [(x0,y0,x1,y1), ...] 覆盖整图、相邻重叠 overlap_ratio。"""
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
    predict_fn: callable(crop_bgr) -> (boxes_xyxy[N,4] 局部坐标, scores[N])
    返回: 全局坐标 boxes[N,4], scores[N]（未做 NMS，交给上层合并）。
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
