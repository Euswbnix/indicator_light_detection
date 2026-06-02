"""数据驱动的分类器标签空间。

无合成数据后，分类器只能识别“有足够真实样本”的类。
- active_classes.json 记录进模型的 kb_id（训练集样本 >= MIN_SAMPLES）。
- idx 0 固定 = not_a_light；其余按 kb_id 升序。
- 不在 active 里的类：检测器仍会框到 → 分类器低置信度/not_a_light → 转人工。

与 taxonomy.py 的区别：taxonomy 是全量 123 类字典（不变）；
class_space 是“这一版模型实际支持的类”，随数据增长而扩大。
"""
import json
from .config import ACTIVE_CLASSES, NOT_A_LIGHT
from .taxonomy import KB_BY_ID


class ClassSpace:
    def __init__(self, kb_ids):
        self.kb_ids = sorted(kb_ids)
        self.idx_to_kb = [None] + self.kb_ids          # idx 0 = not_a_light
        self.kb_to_idx = {kb: i for i, kb in enumerate(self.idx_to_kb) if kb is not None}

    @property
    def num_classes(self):
        return len(self.idx_to_kb)

    def is_active(self, kb_id):
        return kb_id in self.kb_to_idx

    def name(self, idx):
        if idx == 0:
            return NOT_A_LIGHT
        return KB_BY_ID[self.idx_to_kb[idx]]["name"]

    def to_idx(self, kb_id):
        return self.kb_to_idx[kb_id]

    def save(self, path=ACTIVE_CLASSES):
        json.dump({"kb_ids": self.kb_ids}, open(path, "w"), ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path=ACTIVE_CLASSES):
        return cls(json.load(open(path))["kb_ids"])

    @classmethod
    def from_counts(cls, counts: dict, min_samples: int):
        """counts: {kb_id: 训练集样本数} -> 取 >= min_samples 的类。"""
        return cls([kb for kb, c in counts.items() if c >= min_samples])
