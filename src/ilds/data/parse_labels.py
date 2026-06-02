"""解析 done/ 文件夹的标注。

约定：文件名 = 该图中所有亮灯的 kb_id，空格分隔；结尾 (n)/（n） 为重复计数后缀。
例： "84 108 120 (2).jpg" -> 灯 [84, 108, 120]
特殊： "未知" / "无" 表示无法识别 / 无亮灯。

注意：这是“图级多标签”，给 Stage2 分类器提供 弱监督来源 之一（配合真实 bbox 裁剪）。
真正的 bbox 需要单独的检测标注（见 datasets/detector）。本模块只抽“这张图里有哪些类”。
"""
import re
from pathlib import Path

_COUNTER = re.compile(r"\s*[(（]\s*\d+\s*[)）]\s*$")

def parse_filename(stem: str):
    """返回 (kb_ids:list[int], flags:list[str])。"""
    while True:
        new = _COUNTER.sub("", stem)
        if new == stem:
            break
        stem = new
    toks = stem.split()
    ids = [int(t) for t in toks if t.isdigit()]
    flags = [t for t in toks if not t.isdigit()]   # 未知 / 无
    return ids, flags

def scan_done(done_dir):
    """扫描 done/，返回 [(path, kb_ids, flags), ...]。"""
    done_dir = Path(done_dir)
    out = []
    for p in sorted(done_dir.iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        ids, flags = parse_filename(p.stem)
        out.append((p, ids, flags))
    return out

if __name__ == "__main__":
    import sys
    from collections import Counter
    from ..config import DONE_DIR
    rows = scan_done(DONE_DIR)
    cnt = Counter()
    for _, ids, _ in rows:
        cnt.update(ids)
    print(f"图片 {len(rows)} 张，实例 {sum(cnt.values())}，类别 {len(cnt)}")
    for kb, c in cnt.most_common(10):
        print(f"  kb{kb}: {c}")
