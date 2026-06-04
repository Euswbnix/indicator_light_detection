"""Data-driven classifier label space.

Without synthetic data, the classifier can only recognize classes that have enough real samples.
- active_classes.json records the kb_ids in the model (train samples >= MIN_SAMPLES).
- idx 0 is fixed to not_a_light; the rest are kb_ids ascending.
- Classes not in `active`: the detector still boxes them -> classifier gives low confidence /
  not_a_light -> routed to human.

Difference from taxonomy.py: taxonomy is the full 123-class dictionary (fixed);
class_space is "the classes this model version actually supports", which grows with the data.
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
        """counts: {kb_id: train sample count} -> keep classes with >= min_samples."""
        return cls([kb for kb, c in counts.items() if c >= min_samples])
