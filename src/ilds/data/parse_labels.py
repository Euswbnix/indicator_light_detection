"""Parse the filename-based labels in the done/ folder.

Convention: filename = space-separated kb_ids of all lit lights in the image; a trailing
(n)/（n） is a duplicate counter.
e.g. "84 108 120 (2).jpg" -> lights [84, 108, 120]
Special: "未知" (unknown) / "无" (none) mean unidentifiable / no lit light.

Note: this is an "image-level multi-label" -- one weak-supervision source for the Stage2
classifier (alongside real bbox crops). Real bboxes need separate detection annotation
(see datasets/detector). This module only extracts "which classes are in this image".
"""
import re
from pathlib import Path

_COUNTER = re.compile(r"\s*[(（]\s*\d+\s*[)）]\s*$")

def parse_filename(stem: str):
    """Return (kb_ids:list[int], flags:list[str])."""
    while True:
        new = _COUNTER.sub("", stem)
        if new == stem:
            break
        stem = new
    toks = stem.split()
    ids = [int(t) for t in toks if t.isdigit()]
    flags = [t for t in toks if not t.isdigit()]   # non-numeric flag tokens: "未知"(unknown) / "无"(none)
    return ids, flags

def scan_done(done_dir):
    """Scan done/, return [(path, kb_ids, flags), ...]."""
    done_dir = Path(done_dir)
    out = []
    for p in sorted(done_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        ids, flags = parse_filename(p.stem)
        out.append((p, ids, flags))
    return out

if __name__ == "__main__":
    from collections import Counter
    from ..config import DONE_DIR
    rows = scan_done(DONE_DIR)
    cnt = Counter()
    for _, ids, _ in rows:
        cnt.update(ids)
    print(f"images {len(rows)}, instances {sum(cnt.values())}, classes {len(cnt)}")
    for kb, c in cnt.most_common(10):
        print(f"  kb{kb}: {c}")
