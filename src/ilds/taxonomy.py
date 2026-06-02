"""类别体系。

- 检测器(Stage1)：单类 'indicator_light'（只找灯，不分类）。
- 分类器(Stage2)：123 个知识库类 + not_a_light = 124 类。

设计要点：
- 颜色是语义的一部分（同形不同色 = 不同类，如 P 灯红/黄/绿），知识库已用不同 kb_id
  区分，因此这里直接用 kb_id 作为类，不做颜色合并。
- 提供 视觉近似组(merge groups) 仅作参考/评估用，默认不合并，避免信息丢失。
- index <-> kb_id 双向映射，index 0 固定为 not_a_light。
"""
import json
from .config import CLASSES_JSON, NOT_A_LIGHT

with open(CLASSES_JSON, encoding="utf-8") as f:
    _data = json.load(f)

CLASSES = _data["classes"]                              # [{kb_id,name,category,color}, ...]
KB_IDS = [c["kb_id"] for c in CLASSES]                   # 升序的 123 个 kb_id
KB_BY_ID = {c["kb_id"]: c for c in CLASSES}

# ---- 分类器标签空间：index 0 = not_a_light，之后按 kb_id 升序 ----
IDX_TO_KB = [None] + KB_IDS                              # idx 0 -> None(背景)
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

# ---- 视觉近似组（同形态，靠颜色/细节区分）。仅用于报告与混淆分析。----
# 每组列出 kb_id；标注遇到组内难分辨时应 flag 待专家确认，不可瞎猜。
LOOKALIKE_GROUPS = [
    {"name": "P灯(驻车/临停/EPB)", "ids": [15, 16, 36, 120], "note": "圆圈P，绿/黄/橙/红四色四义"},
    {"name": "EBS",              "ids": [68, 89],          "note": "圆圈EBS，黄=一般红=严重"},
    {"name": "ECAS",             "ids": [37, 67],          "note": "圆圈ECAS，黄/红"},
    {"name": "制动(!)",          "ids": [121, 122],        "note": "圆圈感叹号，黄=辅助制动红=系统故障"},
    {"name": "排放故障(国五/六)",  "ids": [115, 116],       "note": "形近，可按车型合并"},
    {"name": "LDW工作态",        "ids": [6, 7, 8, 9],      "note": "车道线4种，可合并为1类"},
    {"name": "缓速器档位",        "ids": [53, 54, 55, 56, 57], "note": "档位数字不同"},
    {"name": "充电线连接",        "ids": [21, 22],          "note": "1/2，形近"},
    {"name": "转向(主车)",        "ids": [111, 112],        "note": "左右"},
    {"name": "转向(挂车)",        "ids": [91, 92],          "note": "左右"},
    {"name": "车门未关",          "ids": [86, 87, 88],      "note": "主/副/主副"},
    {"name": "非正常高度",        "ids": [51, 52],          "note": "上/下"},
    {"name": "DCDC",             "ids": [40, 41],          "note": "DCDC/DCDC2"},
    {"name": "燃气泄漏(LNG/NG)",  "ids": [19, 45],          "note": "形近"},
]

# 业务后处理用：左右转向同亮 = 双闪
TURN_LEFT, TURN_RIGHT = 112, 111
TURN_LEFT_TRAILER, TURN_RIGHT_TRAILER = 92, 91


# ============================================================
# 形状/颜色解耦（最终版用，解决“同形不同色”）
# 思路：分类器出 形状族 + 颜色 两个预测，再查 (形状,颜色)->kb_id。
#   - 形状族：把同形图标归一族（来自 LOOKALIKE_GROUPS），其余各自成族。
#   - 颜色：KB 颜色归一到粗桶 red/yellow/green/blue/white。
#   - 好处：形状头可跨色泛化（连零样本色变体也可能查表识别）；颜色头数据极多很鲁棒；
#           (形状,颜色)->kb 是知识库查表，可审计；未见组合 -> 转人工。
# ============================================================
COLORS = ["red", "yellow", "green", "blue", "white"]
COLOR_TO_IDX = {c: i for i, c in enumerate(COLORS)}

def norm_color(zh: str) -> str:
    """KB 中文颜色 -> 粗桶。"""
    if "红" in zh: return "red"
    if "黄" in zh or "橙" in zh: return "yellow"
    if "绿" in zh: return "green"
    if "蓝" in zh: return "blue"
    return "white"   # 白色 / 黑白色 / 其它

# kb_id -> 形状族 key
_FAMILY_OF = {}
for _g in LOOKALIKE_GROUPS:
    for _kb in _g["ids"]:
        _FAMILY_OF[_kb] = _g["name"]
for _kb in KB_IDS:                       # 未分组的图标：各自成族
    _FAMILY_OF.setdefault(_kb, f"solo:{_kb}")

SHAPE_FAMILIES = sorted(set(_FAMILY_OF.values()))
FAMILY_TO_IDX = {f: i for i, f in enumerate(SHAPE_FAMILIES)}
NUM_SHAPE_FAMILIES = len(SHAPE_FAMILIES)
NUM_COLORS = len(COLORS)

def shape_family_of(kb_id: int) -> str:
    return _FAMILY_OF[kb_id]

def color_of(kb_id: int) -> str:
    return norm_color(KB_BY_ID[kb_id]["color"])

# (形状族 idx, 颜色 idx) -> [kb_id, ...]
SHAPE_COLOR_TO_KB = {}
for _kb in KB_IDS:
    _k = (FAMILY_TO_IDX[shape_family_of(_kb)], COLOR_TO_IDX[color_of(_kb)])
    SHAPE_COLOR_TO_KB.setdefault(_k, []).append(_kb)

# 仍然冲突的组合（同形同色仍多义，如驻车故障 vs EPB故障）：需更细特征或转人工
AMBIGUOUS_COMBOS = {k: v for k, v in SHAPE_COLOR_TO_KB.items() if len(v) > 1}

def lookup_kb(family_idx: int, color_idx: int):
    """返回 (kb_id 或 None, 是否唯一)。None=未见组合→转人工。"""
    cands = SHAPE_COLOR_TO_KB.get((family_idx, color_idx))
    if not cands:
        return None, False
    return cands[0], len(cands) == 1

if __name__ == "__main__":
    print(f"知识库类: {len(CLASSES)} | 分类器标签空间(含not_a_light): {NUM_CLASSES}")
    print(f"idx 0 = {idx_to_name(0)} | idx 1 = kb{IDX_TO_KB[1]} {idx_to_name(1)}")
    print(f"近似组: {len(LOOKALIKE_GROUPS)} 组")
