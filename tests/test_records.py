"""records 的盲答分离与字段约定。"""

from visforge.records import Candidate, QuestionView, Seed


def test_seed_has_only_three_fields():
    seed = Seed(path="p.png", group="g")
    assert seed.page is None
    assert set(Seed.__dataclass_fields__) == {"path", "group", "page"}


def test_question_view_carries_no_answer():
    view = QuestionView(image="a.png", question="图中有几个方块？")
    # 盲答分离：视图里绝不允许出现答案类字段
    assert set(QuestionView.__dataclass_fields__) == {"image", "question"}
    assert not hasattr(view, "answer")
    assert not hasattr(view, "reference")


def test_candidate_defaults():
    cand = Candidate(seed=Seed("p.png", "g"), question="q", answer="a")
    assert cand.status == "pending"
    assert cand.evidence == []
    assert cand.plan == []
    assert cand.seed.path == "p.png"
