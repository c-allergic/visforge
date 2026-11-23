"""验题：先盲答，再核对答案与证据，给出 accept / revise / reject。"""

import json

from .evaluation import match_rule
from .models import extract_json


def answer_blind(view, model):
    """盲答：只给图和问题，不给任何参考答案或证据。"""
    prompt = f"Answer the following question using only the image; keep the answer brief.\nQuestion: {view.question}"
    return model({"text": prompt, "images": [view.image]})


def audit(candidate, blind_answer, model):
    """核对候选题：先规则比对盲答与参考答案，再让验题员结合证据确认。

    就算盲答碰巧答对，证据不足也要改。返回
    {"verdict", "reason", "rule_match", "blind_answer"}，并把 candidate.status
    改成 accepted / revised / rejected。
    """
    rule_ok = match_rule(str(blind_answer), str(candidate.answer))
    prompt = (
        "You are a question auditor. Below are a visual question, its reference answer, "
        "the blind answer from an independent solver, and the evidence kept during question construction.\n"
        f"Question: {candidate.question}\n"
        f"Reference answer: {candidate.answer}\n"
        f"Blind answer: {blind_answer}\n"
        f"Evidence: {json.dumps(candidate.evidence, ensure_ascii=False)}\n"
        "Using the image, judge whether the evidence is sufficient to uniquely determine the answer "
        "and whether the reference answer is correct. "
        "Return revise if the answer needs changing; return reject if the question cannot be "
        "answered uniquely from the image.\n"
        "Output only JSON: {\"verdict\": \"accept|revise|reject\", \"reason\": \"...\"}"
    )
    text, _meta = model({"text": prompt, "images": [candidate.seed.path]})
    data = extract_json(text)
    if not isinstance(data, dict) or data.get("verdict") not in ("accept", "revise", "reject"):
        candidate.status = "rejected"
        return {
            "verdict": "reject",
            "reason": "auditor did not return usable JSON",
            "rule_match": rule_ok,
            "blind_answer": blind_answer,
        }
    verdict = data["verdict"]
    candidate.status = {"accept": "accepted", "revise": "revised", "reject": "rejected"}[verdict]
    return {
        "verdict": verdict,
        "reason": data.get("reason", ""),
        "rule_match": rule_ok,
        "blind_answer": blind_answer,
    }
