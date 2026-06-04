"""Class taxonomy.

- Detector (Stage1): single class 'indicator_light' (only finds lights, no classification).
- Classifier (Stage2): 123 knowledge-base classes + not_a_light = 124 classes.

Design notes:
- Color is part of the semantics (same shape, different color = different class, e.g. the
  P-light red/yellow/green). The knowledge base already separates these with distinct kb_ids,
  so we use kb_id directly as the class and do NOT merge by color.
- Lookalike (merge) groups are provided for reporting/analysis only; not merged by default,
  to avoid losing information.
- index <-> kb_id bidirectional mapping; index 0 is fixed to not_a_light.
"""
import json
from .config import CLASSES_JSON, NOT_A_LIGHT

with open(CLASSES_JSON, encoding="utf-8") as f:
    _data = json.load(f)

CLASSES = _data["classes"]                              # [{kb_id,name,category,color}, ...]
KB_IDS = [c["kb_id"] for c in CLASSES]                   # the 123 kb_ids, ascending
KB_BY_ID = {c["kb_id"]: c for c in CLASSES}

# ---- Classifier label space: index 0 = not_a_light, then kb_ids ascending ----
IDX_TO_KB = [None] + KB_IDS                              # idx 0 -> None (background)
KB_TO_IDX = {kb: i for i, kb in enumerate(IDX_TO_KB) if kb is not None}
NUM_CLASSES = len(IDX_TO_KB)                             # 124

def idx_to_name(idx: int) -> str:
    if idx == 0:
        return NOT_A_LIGHT
    return KB_BY_ID[IDX_TO_KB[idx]]["name"]

def idx_to_color(idx: int) -> str:
    return "" if idx == 0 else KB_BY_ID[IDX_TO_KB[idx]]["color"]

def kb_to_idx(kb_id: int) -> int:
    return KB_TO_IDX[kb_id]

# ---- Lookalike groups (same shape, told apart by color/detail). For reporting & confusion analysis only. ----
# Each group lists kb_ids; when annotators cannot tell members apart they should flag for expert review, not guess.
LOOKALIKE_GROUPS = [
    {"name": "P-light(park/autohold/EPB)", "ids": [15, 16, 36, 120], "note": "circle-P, green/yellow/orange/red = 4 meanings"},
    {"name": "EBS",                "ids": [68, 89],          "note": "circle-EBS, yellow=minor red=severe"},
    {"name": "ECAS",               "ids": [37, 67],          "note": "circle-ECAS, yellow/red"},
    {"name": "brake(!)",           "ids": [121, 122],        "note": "circle-exclamation, yellow=aux brake red=system fault"},
    {"name": "emission(GB5/GB6)",  "ids": [115, 116],        "note": "near-identical, may merge by model"},
    {"name": "LDW-working",        "ids": [6, 7, 8, 9],      "note": "4 lane states, mergeable into 1"},
    {"name": "retarder-gear",      "ids": [53, 54, 55, 56, 57], "note": "different gear digits"},
    {"name": "charge-cable",       "ids": [21, 22],          "note": "1/2, near-identical"},
    {"name": "turn(tractor)",      "ids": [111, 112],        "note": "left/right"},
    {"name": "turn(trailer)",      "ids": [91, 92],          "note": "left/right"},
    {"name": "door-ajar",          "ids": [86, 87, 88],      "note": "main/passenger/both"},
    {"name": "abnormal-height",    "ids": [51, 52],          "note": "up/down"},
    {"name": "DCDC",               "ids": [40, 41],          "note": "DCDC/DCDC2"},
    {"name": "gas-leak(LNG/NG)",   "ids": [19, 45],          "note": "near-identical"},
]

# Business post-processing: left + right turn signals on together = hazard lights
TURN_LEFT, TURN_RIGHT = 112, 111
TURN_LEFT_TRAILER, TURN_RIGHT_TRAILER = 92, 91


# ============================================================
# Shape/color decoupling (final version; solves "same shape, different color")
# Idea: the classifier predicts shape-family + color, then we look up (shape, color) -> kb_id.
#   - shape-family: same-shape icons share a family (from LOOKALIKE_GROUPS); others are singletons.
#   - color: KB color normalized into coarse buckets red/yellow/green/blue/white.
#   - benefits: the shape head generalizes across colors (even zero-shot color variants may be
#     recovered by lookup); the color head has plenty of data and is robust;
#     (shape,color)->kb is an auditable KB lookup; unseen combos -> route to human.
# ============================================================
COLORS = ["red", "yellow", "green", "blue", "white"]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}

def norm_color(zh: str) -> str:
    """KB Chinese color string -> coarse bucket."""
    if "红" in zh: return "red"
    if "黄" in zh or "橙" in zh: return "yellow"
    if "绿" in zh: return "green"
    if "蓝" in zh: return "blue"
    return "white"   # white / black-white / other

# kb_id -> shape-family key
_FAMILY_OF = {}
for _g in LOOKALIKE_GROUPS:
    for _kb in _g["ids"]:
        _FAMILY_OF[_kb] = _g["name"]
for _kb in KB_IDS:                       # ungrouped icons: each its own family
    _FAMILY_OF.setdefault(_kb, f"solo:{_kb}")

SHAPE_FAMILIES = sorted(set(_FAMILY_OF.values()))
FAMILY_TO_IDX = {f: i for i, f in enumerate(SHAPE_FAMILIES)}
NUM_SHAPE_FAMILIES = len(SHAPE_FAMILIES)
NUM_COLORS = len(COLORS)

def shape_family_of(kb_id: int) -> str:
    return _FAMILY_OF[kb_id]

def color_of(kb_id: int) -> str:
    return norm_color(KB_BY_ID[kb_id]["color"])

# (shape-family idx, color idx) -> [kb_id, ...]
SHAPE_COLOR_TO_KB = {}
for _kb in KB_IDS:
    _k = (FAMILY_TO_IDX[shape_family_of(_kb)], COLOR_TO_IDX[color_of(_kb)])
    SHAPE_COLOR_TO_KB.setdefault(_k, []).append(_kb)

# Combos still ambiguous (same shape AND same color but multiple meanings, e.g. park-fault vs EPB-fault):
# need finer features or route to human.
AMBIGUOUS_COMBOS = {k: v for k, v in SHAPE_COLOR_TO_KB.items() if len(v) > 1}

def lookup_kb(family_idx: int, color_idx: int):
    """Return (kb_id or None, is_unique). None = unseen combo -> route to human."""
    cands = SHAPE_COLOR_TO_KB.get((family_idx, color_idx))
    if not cands:
        return None, False
    return cands[0], len(cands) == 1

if __name__ == "__main__":
    print(f"KB classes: {len(CLASSES)} | classifier label space (incl. not_a_light): {NUM_CLASSES}")
    print(f"idx 0 = {idx_to_name(0)} | idx 1 = kb{IDX_TO_KB[1]} {idx_to_name(1)}")
    print(f"lookalike groups: {len(LOOKALIKE_GROUPS)}")
