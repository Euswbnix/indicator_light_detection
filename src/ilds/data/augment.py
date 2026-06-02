"""分维度增强 —— 抗光照，但不破坏颜色语义。

核心矛盾：常规做法用大幅 HSV 抖动让模型对颜色鲁棒，但本任务颜色是语义
（P灯红/黄/绿是不同类）。因此：
  - 亮度(V)、饱和度(S)：放开抖（含降饱和，模拟阳光冲淡 / 过曝泛白）
  - 色相(H)：只 ±10°！大幅旋转会把红/黄/绿混为一谈
  - 额外模拟手机自动白平衡偏移、过曝、模糊、JPEG 伪影
"""
import random
import numpy as np
import cv2

from ..config import AUG


def _rand(a, b):
    return random.uniform(a, b)


def augment_patch(img_bgr, aug=AUG):
    """输入 uint8 BGR patch，返回增强后的 uint8 BGR。"""
    img = img_bgr.copy()

    # HSV 分维度
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + _rand(-aug["hue_deg"], aug["hue_deg"]) / 2) % 180  # H: ±10° (OpenCV H是0-180)
    hsv[..., 1] = np.clip(hsv[..., 1] * _rand(*aug["saturation"]), 0, 255)          # S
    hsv[..., 2] = np.clip(hsv[..., 2] * _rand(*aug["brightness"]), 0, 255)          # V
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 白平衡偏移（独立缩放 B/R 通道）
    if aug["wb_shift"] > 0:
        wb = aug["wb_shift"]
        b = 1 + _rand(-wb, wb); r = 1 + _rand(-wb, wb)
        img = img.astype(np.float32)
        img[..., 0] = np.clip(img[..., 0] * b, 0, 255)
        img[..., 2] = np.clip(img[..., 2] * r, 0, 255)
        img = img.astype(np.uint8)

    # 过曝泛白（提亮后裁剪，模拟灯芯饱和）
    if random.random() < aug["overexpose_p"]:
        g = _rand(1.5, 2.5)
        img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)

    # 模糊
    if random.random() < aug["blur_p"]:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # JPEG 伪影
    if random.random() < aug["jpeg_p"]:
        q = random.randint(30, 75)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img
