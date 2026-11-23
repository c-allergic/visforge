"""判分两层与难度分；盲答采样不接触参考答案。"""

from fakes import FakeJudge, FakeModel
from visforge.evaluation import h_s, make_judge, match_rule, sample_answers, score
from visforge.records import QuestionView


def test_rule_layer_matching():
    assert score("茶水。", "茶水") is True
    assert score("  TEA ", "tea") is True
    assert score("5个", "5") is True  # 数值 + 单位差异
    assert score("5.01", "5") is True  # 小的相对误差
    assert score("6", "5") is False
    assert match_rule("蓝杯子", "蓝 杯子") is True  # 多余空白


def test_judge_layer_only_used_when_rule_fails():
    judge = FakeJudge({"差不多": True, "差不多吧": True})
    assert score("茶水", "茶水", judge) is True
    assert judge.calls == []  # 规则层命中，判官没被调用
    assert score("差不多吧", "茶水", judge) is True
    assert judge.calls == [("差不多吧", "茶水")]
    assert score("完全不同", "茶水", judge) is False
    assert score("完全不同", "茶水") is False  # 没判官就是不对


def test_make_judge_uses_model():
    model = FakeModel({"equivalent": '{"equivalent": true}'})
    judge = make_judge(model)
    assert judge("the bluish cup", "blue cup") is True


def test_scoring_preserves_numeric_signs_decimal_places_and_units():
    for answer, reference in [("-5", "5"), ("1.25", "125"), ("5%", "5 kg"), ("", "")]:
        assert not match_rule(answer, reference)
    assert not make_judge(FakeModel(default='{"equivalent": "false"}'))("cat", "dog")


def test_h_s():
    assert h_s([True, True, False, False]) == 0.5
    assert h_s([True]) == 0.0
    assert h_s([False]) == 1.0
    assert h_s([]) == 0.0


def test_sample_answers_stays_blind():
    views = [QuestionView(image="a.png", question="有几个方块？"),
             QuestionView(image="b.png", question="写着什么？")]
    model = FakeModel({"有几个方块？": "2", "写着什么？": "TEA"})
    records = sample_answers(views, model, k=2)
    assert len(records) == 2
    assert all(len(r["answers"]) == 2 for r in records)
    assert records[0]["answers"] == ["2", "2"]
    for rec in records:
        assert set(rec["view"].__dataclass_fields__) == {"image", "question"}  # 盲答分离
        assert not hasattr(rec["view"], "answer")
