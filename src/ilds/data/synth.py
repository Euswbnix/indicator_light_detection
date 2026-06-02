"""合成分类训练样本 —— 用知识库 123 个图标 PNG 生成灯 patch。

两阶段架构最大红利：分类器看到的是裁剪出的小 patch，与图标 PNG 同域，
加个背景 + 辉光就是训练样本，无需把图标贴回仪表盘。一次性覆盖 82 个零样本类。

两条渲染轨（同一图标同一类，质感不同，缺一不可）：
  - LCD 屏渲染：抗锯齿、平滑、可能带渐变（中央大图标）
  - LED 灯板渲染：像素化、背光硬边、辉光（顶部灯板小灯）
"""
import random
from pathlib import Path
import numpy as np
import cv2

from ..config import KB_DIR, CLS_IMGSZ


def _load_icon(png_path):
    """读 PNG，返回 RGBA。"""
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def _alpha_mask(rgba):
    """得到图标前景 mask（透明通道或非白）。"""
    a = rgba[..., 3]
    if a.min() < 250:
        return a > 30
    gray = cv2.cvtColor(rgba[..., :3], cv2.COLOR_BGR2GRAY)
    return gray < 220


def _add_glow(bgr, mask, color, strength=0.6):
    """对前景加辉光，模拟自发光。"""
    glow = np.zeros_like(bgr, np.float32)
    glow[mask] = color
    glow = cv2.GaussianBlur(glow, (0, 0), sigmaX=max(2, bgr.shape[0] // 20))
    out = bgr.astype(np.float32) + glow * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def render_patch(png_path, style="led", size=CLS_IMGSZ):
    """把一个图标渲染成一张灯 patch（uint8 BGR）。"""
    rgba = _load_icon(png_path)
    if rgba is None:
        return None
    mask = _alpha_mask(rgba)
    fg = rgba[..., :3].astype(np.float32)

    # 暗背景（仪表盘底色：近黑，略带噪点）
    bg_val = random.randint(5, 35)
    canvas = np.full((size, size, 3), bg_val, np.uint8)
    canvas = canvas + np.random.randint(0, 12, canvas.shape, np.uint8)

    # 缩放图标到 patch 的 50%-85%
    scale = random.uniform(0.5, 0.85)
    side = int(size * scale)
    icon = cv2.resize(rgba, (side, side), interpolation=cv2.INTER_AREA)
    m = cv2.resize(mask.astype(np.uint8), (side, side)) > 0
    ic = icon[..., :3]

    if style == "lcd":
        ic = cv2.GaussianBlur(ic, (3, 3), 0)            # 平滑抗锯齿
    elif style == "led":
        # 像素化 + 硬边
        small = cv2.resize(ic, (side // 3, side // 3), interpolation=cv2.INTER_NEAREST)
        ic = cv2.resize(small, (side, side), interpolation=cv2.INTER_NEAREST)

    # 随机位置贴上
    oy = random.randint(0, size - side); ox = random.randint(0, size - side)
    roi = canvas[oy:oy + side, ox:ox + side]
    roi[m] = ic[m]
    canvas[oy:oy + side, ox:ox + side] = roi

    # 辉光（取图标主色）
    if m.any():
        color = ic[m].mean(axis=0)
        full_mask = np.zeros((size, size), bool)
        full_mask[oy:oy + side, ox:ox + side] = m
        canvas = _add_glow(canvas, full_mask, color, strength=random.uniform(0.3, 0.8))
    return canvas


def iter_icons():
    """遍历知识库图标，返回 (kb_id, png_path)。"""
    for p in sorted(Path(KB_DIR).glob("*.png")):
        head = p.name.split("-")[0]
        if head.isdigit():
            yield int(head), p


def make_negative(size=CLS_IMGSZ):
    """合成易负样本（背景/反光/文字条），用于 not_a_light。"""
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
    print(f"预览样本写入 {out}（{n} 张灯 + 6 张负样本）")
