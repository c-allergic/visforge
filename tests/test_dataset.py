"""导出：去重（规范化后精确匹配）与构造记录可回查。"""

import json

from fakes import make_scene
from visforge.dataset import export_verified
from visforge.records import Candidate, Seed


def _cand(scene, question, answer="blue", status="accepted", group="g1"):
    return Candidate(
        seed=Seed(str(scene), group), question=question, answer=answer, status=status,
        plan=[{"template": "t"}, {"id": "x", "op": "Select", "ok": True, "value": None,
                                  "evidence": [], "failure_reason": ""}],
        evidence=[{"bbox": [1, 2, 3, 4]}],
    )


def test_export_dedups_normalized_questions_on_same_image(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    cands = [
        _cand(scene, "What color is it?"),
        _cand(scene, "what   COLOR  is it?"),  # 规范化后同题
        _cand(scene, "How many?"),
        _cand(scene, "被拒绝的题", status="rejected"),
    ]
    result = export_verified(cands, tmp_path / "out")
    assert result["written"] == 2 and result["duplicates"] == 1
    lines = [json.loads(l) for l in (tmp_path / "out" / "data.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert {l["question"] for l in lines} == {"What color is it?", "How many?"}


def test_export_same_question_different_image_kept(tmp_path):
    a = make_scene(tmp_path / "a.png")
    b = make_scene(tmp_path / "b.png")
    result = export_verified([_cand(a, "What color?"), _cand(b, "What color?")], tmp_path / "out")
    assert result["written"] == 2 and result["duplicates"] == 0


def test_export_entries_trace_back_to_seed_and_plan(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    export_verified([_cand(scene, "What color?")], tmp_path / "out")
    entry = json.loads((tmp_path / "out" / "data.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert entry["seed"] == {"path": str(scene), "group": "g1", "page": None}
    assert entry["plan"][0] == {"template": "t"}
    assert entry["evidence"] == [{"bbox": [1, 2, 3, 4]}]
    # 图片副本放在 group 子目录里
    assert (tmp_path / "out" / "images" / "g1" / "scene.png").exists()
