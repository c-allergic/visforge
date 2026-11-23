"""导出合格的题：规范化去重，写 data.jsonl 并复制图片。"""

import json
import shutil
from pathlib import Path

from .evaluation import normalize_text


def dedup_key(image, question):
    """同一图上题目去重的键：图片路径 + 规范化问句（casefold、去空白和标点）。"""
    return (str(image), normalize_text(question))


def normalize_question(text):
    """同一图上题目去重的键：casefold、去多余空白和标点后精确匹配。"""
    return normalize_text(text)


def export_verified(candidates, out_dir):
    """导出通过检查（status == accepted）的候选题。

    同一张图上规范化后相同的题只收一次。输出 data.jsonl（含 seed 出处和
    plan 轨迹，可查回构造记录）并复制图片。返回 {"written", "duplicates", "out"}。
    """
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    seen = set()
    written, duplicates, lines = 0, 0, []
    for cand in candidates:
        if cand.status != "accepted":
            continue
        src = Path(cand.seed.path)
        rel = f"images/{cand.seed.group}/{src.name}" if cand.seed.group else f"images/{src.name}"
        key = dedup_key(src, cand.question)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy2(src, dest)
        lines.append(
            {
                "image": rel,
                "question": cand.question,
                "answer": cand.answer,
                "seed": {"path": str(src), "group": cand.seed.group, "page": cand.seed.page},
                "plan": cand.plan,
                "evidence": cand.evidence,
            }
        )
        written += 1
    with (out / "data.jsonl").open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return {"written": written, "duplicates": duplicates, "out": str(out)}
