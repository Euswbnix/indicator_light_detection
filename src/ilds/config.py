"""全局配置：路径、超参、阈值。

约定：所有训练/推理用 ~/anaconda3/bin/python (py3.9, torch2.7 + ultralytics + cv2)。
"""
from pathlib import Path

# ---- 路径 ----
ROOT = Path(__file__).resolve().parents[2]          # indicator_light_detection/
KB_DIR = ROOT / "仪表灯知识库"                          # 知识库（图标 PNG + xls）
CLASSES_JSON = Path(__file__).resolve().parent / "classes.json"

# 原始数据
DONE_DIR = ROOT / "done"                            # 已标注（文件名=灯号列表）
RAW_DIRS = [                                         # 待标注/原始照片池
    ROOT / "仪表照片-重庆申通202507",
    ROOT / "全部照片_汇总",
]

# 产出
DATASETS = ROOT / "datasets"
DET_DS = DATASETS / "detector"                      # YOLO 格式（images/ + labels/）
CLS_DS = DATASETS / "classifier"                    # 分类 patch（按类分文件夹 + negatives/）
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"

# ---- Stage 1 检测器 ----
DET_IMGSZ = 1280            # 训练/推理输入边长（小灯需要高分辨率）
DET_MODEL_YAML = Path(__file__).resolve().parent / "models" / "yolov8s-p2.yaml"
DET_CONF = 0.15             # 低阈值，宁多勿漏，误检交给 Stage2 的 not_a_light 兜底
DET_IOU = 0.5               # 训练 NMS

# 推理切片 (SAHI 风格)
TILE = 640
TILE_OVERLAP = 0.25         # 重叠比例，须 > 最大灯尺寸占 tile 比，防止灯被切线劈开
DO_FULL_PASS = True         # 额外跑一次全图，保住中央大图标

# 合并去重用 DIoU-NMS
NMS_IOU = 0.55
NMS_KIND = "diou"           # diou / iou / soft

# ---- Stage 2 分类器 ----
CLS_IMGSZ = 128             # patch 输入（灯小，过大放大引入模糊）
CLS_CROP_EXPAND = 1.5       # 裁剪时 bbox 外扩倍数（保留光晕，过曝时外环仍有真色）
NOT_A_LIGHT = "not_a_light" # 背景/误检类
CLS_REJECT_THRESH = 0.50    # softmax 最高分低于此 → 判不可信 → 丢弃/转人工

# ---- 分类器数据（真实裁剪，弃用合成正样本）----
CROPS_DIR = CLS_DS / "crops"            # 按 kb_id 分文件夹的真实裁剪 patch
NEG_DIR = CLS_DS / "negatives"          # not_a_light（难负样本挖掘 + 随机非灯）
MANIFEST = CLS_DS / "manifest.csv"      # path,kb_id,split,source_image
ACTIVE_CLASSES = CLS_DS / "active_classes.json"  # 数据驱动：进模型的类（其余转人工）
MIN_SAMPLES = 20         # 类的“生存线”：训练集样本 >= 此值才进分类器；不足则转人工
VAL_RATIO = 0.15         # 按图划分验证集
SAMPLE_ALPHA = 0.5       # 采样权重 ∝ (1/count)^alpha：0=不均衡,1=完全均衡,0.5=平方根采样
NEG_FRACTION = 0.15      # 训练时 not_a_light 目标占比

# ---- 分维度增强（核心：抗光照但不破坏颜色语义）----
AUG = dict(
    brightness=(0.5, 1.6),  # V 放开抖
    saturation=(0.4, 1.3),  # S 放开抖（含降饱和，模拟阳光冲淡）
    hue_deg=10,             # H 只 ±10°！大幅旋转会摧毁红/黄/绿语义
    wb_shift=0.20,          # 白平衡偏移幅度（模拟手机自动白平衡）
    overexpose_p=0.25,      # 模拟灯芯过曝泛白概率
    blur_p=0.30,            # 模拟运动模糊/失焦
    jpeg_p=0.30,            # 模拟压缩伪影
)
