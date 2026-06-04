"""Global config: paths, hyperparameters, thresholds.

Convention: run all training/inference with a Python 3.9 env that has
torch 2.7 + ultralytics + opencv (e.g. ~/anaconda3/bin/python locally).
"""
from pathlib import Path

# ---- Paths ----
ROOT = Path(__file__).resolve().parents[2]          # indicator_light_detection/
KB_DIR = ROOT / "仪表灯知识库"                          # knowledge base (icon PNGs + xls)
CLASSES_JSON = Path(__file__).resolve().parent / "classes.json"

# Raw data
DONE_DIR = ROOT / "done"                            # annotated (filename = list of light ids)
RAW_DIRS = [                                         # pools of raw / to-be-annotated photos
    ROOT / "仪表照片-重庆申通202507",
    ROOT / "全部照片_汇总",
]

# Outputs
DATASETS = ROOT / "datasets"
DET_DS = DATASETS / "detector"                      # YOLO format (images/ + labels/)
CLS_DS = DATASETS / "classifier"                    # classifier patches (per-class dirs + negatives/)
WEIGHTS = ROOT / "weights"
RUNS = ROOT / "runs"

# ---- Stage 1 detector ----
DET_IMGSZ = 1280            # train/infer input size (small lights need high resolution)
DET_MODEL_YAML = Path(__file__).resolve().parent / "models" / "yolov8s-p2.yaml"
DET_CONF = 0.15             # low threshold: prefer recall; false positives are caught by Stage2 not_a_light
DET_IOU = 0.5               # training NMS

# Sliced inference (SAHI-style)
TILE = 640
TILE_OVERLAP = 0.25         # overlap ratio; must exceed largest-light/tile ratio so lights are not cut by seams
DO_FULL_PASS = True         # also run once on the full image to keep the large central icon

# DIoU-NMS for merging/deduplication
NMS_IOU = 0.55
NMS_KIND = "diou"           # diou / iou / soft

# ---- Stage 2 classifier ----
CLS_IMGSZ = 128             # patch input (lights are small; upscaling too much adds blur)
CLS_CROP_EXPAND = 1.5       # bbox expansion factor when cropping (keep the halo; outer ring keeps true color when overexposed)
NOT_A_LIGHT = "not_a_light" # background / false-positive class
CLS_REJECT_THRESH = 0.50    # if top softmax score is below this -> untrusted -> drop / route to human

# ---- Classifier data (real crops; synthetic positives dropped) ----
CROPS_DIR = CLS_DS / "crops"            # real crop patches in per-kb_id folders
NEG_DIR = CLS_DS / "negatives"          # not_a_light (hard-negative mining + random non-lights)
MANIFEST = CLS_DS / "manifest.csv"      # path,kb_id,split,source_image
ACTIVE_CLASSES = CLS_DS / "active_classes.json"  # data-driven: classes in the model (rest -> human)
MIN_SAMPLES = 20         # class "survival line": needs >= this many train samples to enter the classifier; else -> human
VAL_RATIO = 0.15         # image-level validation split
SAMPLE_ALPHA = 0.5       # sampling weight prop to (1/count)^alpha: 0=imbalanced, 1=fully balanced, 0.5=sqrt sampling
NEG_FRACTION = 0.15      # target fraction of not_a_light during training

# ---- Per-channel augmentation (key: robust to lighting, but preserve color semantics) ----
AUG = dict(
    brightness=(0.5, 1.6),  # V: jitter freely
    saturation=(0.4, 1.3),  # S: jitter freely (incl. desaturation, simulating sun washout)
    hue_deg=10,             # H: only +/-10 deg! Large rotation destroys the red/yellow/green semantics
    wb_shift=0.20,          # white-balance shift magnitude (simulate phone auto white balance)
    overexpose_p=0.25,      # probability of simulating overexposed (blown-out white) light core
    blur_p=0.30,            # simulate motion blur / out of focus
    jpeg_p=0.30,            # simulate compression artifacts
)
