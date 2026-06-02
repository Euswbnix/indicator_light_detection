"""DIoU-NMS。

密集小灯场景：两个相邻小灯的框可能高度重叠（IoU>0.5）但中心点不同。
标准 NMS 会把它们合并掉。DIoU 在 IoU 基础上扣除“中心点距离惩罚”，
中心分得开的框即使框重叠也予以保留 —— 正好解决状态栏一排小灯被并框的问题。
"""
import numpy as np


def _diou(box, boxes):
    """box:(4,) boxes:(N,4) xyxy -> diou (N,)。"""
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a1 = (box[2] - box[0]) * (box[3] - box[1])
    a2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = inter / (a1 + a2 - inter + 1e-9)
    # 中心点距离
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    cxs = (boxes[:, 0] + boxes[:, 2]) / 2; cys = (boxes[:, 1] + boxes[:, 3]) / 2
    center = (cx - cxs) ** 2 + (cy - cys) ** 2
    # 最小外接框对角线
    ex1 = np.minimum(box[0], boxes[:, 0]); ey1 = np.minimum(box[1], boxes[:, 1])
    ex2 = np.maximum(box[2], boxes[:, 2]); ey2 = np.maximum(box[3], boxes[:, 3])
    diag = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-9
    return iou - center / diag


def diou_nms(boxes, scores, iou_thresh=0.55):
    """boxes:(N,4) xyxy, scores:(N,) -> keep indices。"""
    if len(boxes) == 0:
        return []
    boxes = np.asarray(boxes, dtype=float)
    scores = np.asarray(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        d = _diou(boxes[i], boxes[order[1:]])
        order = order[1:][d <= iou_thresh]
    return keep
