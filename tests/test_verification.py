"""验题：盲答入口不带答案；audit 的 verdict 与状态更新。"""

from fakes import BLUE_BOX, FakeModel, generation_model, make_scene
from visforge.records import Candidate, QuestionView, Seed
from visforge.verification import answer_blind, audit


def _candidate(scene):
    return Candidate(
        seed=Seed(str(scene), "g"),
        question="What is written on the object that supports the blue square?",
        answer="TEA",
        evidence=[{"step": "t", "object": "blue square", "bbox": BLUE_BOX}],
    )


def test_answer_blind_sends_only_image_and_question(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = FakeModel(default="TEA")
    view = QuestionView(image=str(scene), question="What is written on the object that supports the blue square?")
    text, meta = answer_blind(view, model)
    assert text == "TEA" and meta["model"] == "fake"
    sent = model.calls[0]
    assert "supports the blue square" in sent["text"]
    assert [str(p) for p in sent["images"]] == [str(scene)]  # 带的是图本身
    # 盲答请求里不允许出现参考答案或证据
    assert "TEA" not in sent["text"]
    assert "evidence" not in sent["text"] and "Reference answer" not in sent["text"]


def test_audit_accepts_when_verifier_confirms(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    cand = _candidate(scene)
    blind, _ = answer_blind(QuestionView(str(scene), cand.question), FakeModel(default="TEA"))
    result = audit(cand, blind, generation_model())
    assert result["verdict"] == "accept" and result["rule_match"] is True
    assert cand.status == "accepted"
    assert result["reason"]


def test_audit_records_rule_mismatch_but_can_still_accept(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    cand = _candidate(scene)
    # 盲答给个错误说法：规则不匹配，但验题员结合证据确认参考答案
    result = audit(cand, "DISH", generation_model())
    assert result["rule_match"] is False
    assert result["verdict"] == "accept" and cand.status == "accepted"


def test_audit_verifier_reject(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    cand = _candidate(scene)
    model = generation_model(verdict="reject")
    result = audit(cand, "TEA", model)
    assert result["verdict"] == "reject" and cand.status == "rejected"


def test_audit_verifier_revise(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    cand = _candidate(scene)
    result = audit(cand, "TEA", generation_model(verdict="revise"))
    assert result["verdict"] == "revise" and cand.status == "revised"


def test_audit_garbage_verdict_rejects(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    cand = _candidate(scene)
    model = FakeModel({"You are a question auditor": "I do not understand"})
    result = audit(cand, "TEA", model)
    assert result["verdict"] == "reject" and cand.status == "rejected"
    assert result["blind_answer"] == "TEA"  # 结果里带着盲答备查
