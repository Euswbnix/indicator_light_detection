"""Per-channel augmentation -- robust to lighting without breaking color semantics.

Core tension: the usual approach uses heavy HSV jitter to make a model color-robust, but here
color IS semantic (the P-light red/yellow/green are different classes). Therefore:
  - brightness (V), saturation (S): jitter freely (incl. desaturation, simulating sun washout / blown-out white)
  - hue (H): only +/-10 deg! Large rotation conflates red/yellow/green
  - also simulate phone auto white-balance shift, overexposure, blur, JPEG artifacts
"""
import random
import numpy as np
import cv2

from ..config import AUG


def _rand(a, b):
    return random.uniform(a, b)


def augment_patch(img_bgr, aug=AUG):
    """Input uint8 BGR patch, return augmented uint8 BGR."""
    img = img_bgr.copy()

    # per-channel HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + _rand(-aug["hue_deg"], aug["hue_deg"]) / 2) % 180  # H: +/-10 deg (OpenCV H is 0-180)
    hsv[..., 1] = np.clip(hsv[..., 1] * _rand(*aug["saturation"]), 0, 255)          # S
    hsv[..., 2] = np.clip(hsv[..., 2] * _rand(*aug["brightness"]), 0, 255)          # V
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # white-balance shift (scale B/R channels independently)
    if aug["wb_shift"] > 0:
        wb = aug["wb_shift"]
        b = 1 + _rand(-wb, wb); r = 1 + _rand(-wb, wb)
        img = img.astype(np.float32)
        img[..., 0] = np.clip(img[..., 0] * b, 0, 255)
        img[..., 2] = np.clip(img[..., 2] * r, 0, 255)
        img = img.astype(np.uint8)

    # overexposure / blown-out white (brighten then clip, simulating a saturated light core)
    if random.random() < aug["overexpose_p"]:
        g = _rand(1.5, 2.5)
        img = np.clip(img.astype(np.float32) * g, 0, 255).astype(np.uint8)

    # blur
    if random.random() < aug["blur_p"]:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # JPEG artifacts
    if random.random() < aug["jpeg_p"]:
        q = random.randint(30, 75)
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(enc, cv2.IMREAD_COLOR)

    return img
