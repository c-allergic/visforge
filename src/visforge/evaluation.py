"""答题与判分：两层判分，先规则匹配，不一致且给了判官才请模型判等效。"""

import math
import re
import unicodedata

from .models import extract_json


def normalize_text(text):
    """规范化文本：casefold、去掉全部空白和标点。"""
    s = str(text).strip().casefold()
    s = re.sub(r"\s+", "", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    return s.strip()


def _numbers(text):
    return re.findall(r"-?\d+(?:\.\d+)?", text)


def match_rule(answer, reference, rel_tol=0.01):
    """规则层判分：规范化后文本相等；或两边都是单个数值且允许小的相对误差。"""
    raw_a = re.sub(r"\s+", "", str(answer).casefold()).replace("−", "-")
    raw_r = re.sub(r"\s+", "", str(reference).casefold()).replace("−", "-")
    na, nr = _numbers(raw_a), _numbers(raw_r)
    a, r = normalize_text(raw_a), normalize_text(raw_r)
    if a and a == r and na == nr:
        return True
    if len(na) == 1 and len(nr) == 1:
        # 都是单个数值：去掉数字后的其余文字须一致或有一边为空
        rest_a = re.sub(r"-?\d+(?:\.\d+)?", "", raw_a)
        rest_r = re.sub(r"-?\d+(?:\.\d+)?", "", raw_r)
        if rest_a == rest_r or {rest_a, rest_r} == {"", "个"}:
            return math.isclose(float(na[0]), float(nr[0]), rel_tol=rel_tol)
    return False


def score(answer, reference, judge=None):
    """两层判分：先规则匹配；不一致且给了 judge 才调判官。返回 bool。"""
    if match_rule(answer, reference):
        return True
    if judge is None:
        return False
    return bool(judge(str(answer), str(reference)))


def make_judge(model):
    """用判官模型判两种说法是否等效，返回 (answer, reference) -> bool 的函数。"""

    def judge(answer, reference):
        prompt = (
            f"Reference answer: {reference}\nAnswer: {answer}\n"
            "As answers to a visual question, are these two statements equivalent? "
            "Output only JSON: {\"equivalent\": true|false}"
        )
        text, _meta = model({"text": prompt})
        data = extract_json(text)
        return isinstance(data, dict) and data.get("equivalent") is True

    return judge


def sample_answers(views, model, k=1):
    """每题让答题模型盲答 k 次（默认一次）。

    只返回 {"view", "answers"}，不接触参考答案。
    """
    from .verification import answer_blind  # 延迟导入，避免与 verification 循环

    records = []
    for view in views:
        answers = [answer_blind(view, model)[0] for _ in range(max(1, int(k)))]
        records.append({"view": view, "answers": answers})
    return records


def h_s(per_question):
    """难度分 h_s = 1 - 平均答对率；没有题时记 0。"""
    if not per_question:
        return 0.0
    hits = sum(1 for c in per_question if c)
    return 1.0 - hits / len(per_question)
