"""读选图清单，变成 Seed 列表。"""

import csv
from pathlib import Path

from PIL import Image

from .records import Seed


def load_seeds(manifest, purpose=None):
    """读选图清单：CSV（列 path,group,page[,purpose]）或纯文本（每行一个路径）。

    purpose 给了且清单带 purpose 列时按用途过滤；逐张打开原图确认可读；
    同组按页序稳定排序（无页码的保持清单顺序，排在有页码的后面）。
    """
    manifest = Path(manifest)
    rows = []
    if manifest.suffix.lower() == ".csv":
        with manifest.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                raw = (row.get("path") or "").strip()
                if not raw:
                    continue
                listed = (row.get("purpose") or "").strip()
                if purpose and listed and listed != purpose:
                    continue
                page = None
                if row.get("page") not in (None, ""):
                    page = int(row["page"])
                group = (row.get("group") or "").strip() or Path(raw).parent.name
                rows.append((raw, group, page))
    else:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append((line, Path(line).parent.name, None))
    seeds = []
    for raw, group, page in rows:
        path = Path(raw)
        if not path.is_absolute():
            path = manifest.parent / path
        with Image.open(path) as im:  # 打开原图，确认可读
            im.verify()
        seeds.append(Seed(path=str(path), group=group, page=page))
    seeds.sort(key=lambda s: (s.group, s.page if s.page is not None else float("inf")))
    return seeds
