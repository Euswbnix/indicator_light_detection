"""Synthetic classifier samples -- render light patches from the 123 knowledge-base icon PNGs.

NOTE: synthetic POSITIVES are no longer used for training (the project decided that zero-sample
classes are too rare and should be routed to human instead). `render_patch` is kept for reference/
visualization only. `make_negative` is still useful for generating not_a_light negatives.

Original rationale (kept for context): the biggest win of the two-stage design is that the
classifier sees cropped small patches, which are in the same domain as the icon PNGs -- add a
background + glow and you have a training sample, without pasting icons back onto a dashboard.

Two render tracks (same icon, same class, different texture):
  - LCD-screen render: anti-aliased, smooth, possibly gradient (central large icon)
  - LED-panel render: pixelated, hard backlit edges, glow (top-panel small lights)
"""
import random
from pathlib import Path
import numpy as np
import cv2

from ..config import KB_DIR, CLS_IMGSZ


def _load_icon(png_path):
    """Read a PNG, return RGBA."""
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _alpha_mask(rgba):
    """Foreground mask of the icon (from alpha channel, or non-white)."""
    a = rgba[..., 3]
    if a.min() < 250:
        return a > 30
    gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY)
    return gray < 220


def _add_glow(bgr, mask, color, strength=0.6):
    """Add glow to the foreground to simulate self-emission."""
    glow = np.zeros_like(bgr, np.float32)
    glow[mask] = color
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=max(2, bgr.shape[0] // 20))
    out = bgr.astype(np.float32) + glow * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def render_patch(png_path, style="led", size=CLS_IMGSZ):
    """Render one icon into a light patch (uint8 BGR). Deprecated for training; viz only."""
    rgba = _load_icon(png_path)
    if rgba is None:
        return None
    mask = _alpha_mask(rgba)

    # dark background (dashboard base color: near-black, slight noise)
    bg_val = random.randint(5, 35)
    canvas = np.full((size, size, 3), bg_val, np.uint8)
    canvas = canvas + np.random.randint(0, 12, canvas.shape, np.uint8)

    # scale icon to 50%-85% of the patch
    scale = random.uniform(0.5, 0.85)
    side = int(size * scale)
    icon = cv2.resize(rgba, (side, side), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (side, side)) > 0
    ic = icon[..., :3]

    if style == "lcd":
        ic = cv2.GaussianBlur(ic, (3, 3), 0)            # smooth / anti-alias
    elif style == "led":
        # pixelate + hard edges
        small = cv2.resize(ic, (side // 3, side // 3), interpolation=cv2.INTER_NEAREST)
        ic = cv2.resize(small, (side, side), interpolation=cv2.INTER_NEAREST)

    # paste at a random position
    oy = random.randint(0, size - side); ox = random.randint(0, size - side)
    roi = canvas[oy:oy + side, ox:ox + side]
    roi[m] = ic[m]
    canvas[oy:oy + side, ox:ox + side] = roi

    # glow (use the icon's dominant color)
    if m.any():
        color = ic[m].mean(axis=0)
        full_mask = np.zeros((size, size), bool)
        full_mask[oy:oy + side, ox:ox + side] = m
        canvas = _add_glow(canvas, full_mask, color, strength=random.uniform(0.3, 0.8))
    return canvas


def iter_icons():
    """Iterate knowledge-base icons, yield (kb_id, png_path)."""
    for p in sorted(Path(KB_DIR).glob("*.png")):
        head = p.name.split("-")[0]
        if head.isdigit():
            yield int(head), p


def make_negative(size=CLS_IMGSZ):
    """Synthesize an easy negative (background/glare/text bar) for not_a_light."""
    kind = random.choice(["dark", "glare", "text", "noise"])
    img = np.full((size, size, 3), random.randint(5, 40), np.uint8)
    if kind == "glare":
        cv2.circle(img, (random.randint(0, size), random.randint(0, size)),
                   random.randint(size // 4, size // 2), (200, 200, 200), -1)
        img = cv2.GaussianBlur(img, (0, 0), size // 8)
    elif kind == "text":
        cv2.putText(img, random.choice(["31C", "08:22", "SPN", "120", "0.1MPa"]),
                    (5, size // 2), cv2.FONT_HERSHEY_SIMPLEX, random.uniform(0.6, 1.2),
                    (230, 230, 230), 2)
    elif kind == "noise":
        img = np.random.randint(0, 80, (size, size, 3), np.uint8)
    return img


if __name__ == "__main__":
    out = Path(KB_DIR).parent / "_synth_preview"
    out.mkdir(exist_ok=True)
    n = 0
    for kb_id, png in iter_icons():
        for style in ("lcd", "led"):
            p = render_patch(png, style=style)
            if p is not None:
                cv2.imwrite(str(out / f"kb{kb_id}_{style}.jpg"), p); n += 1
        if kb_id > 20:
            break
    for i in range(6):
        cv2.imwrite(str(out / f"neg_{i}.jpg"), make_negative())
    print(f"preview samples written to {out} ({n} lights + 6 negatives)")
