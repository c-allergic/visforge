"""出题：两条固定模板、失败记录卡点、每图预算。"""

import json
from copy import deepcopy
import pytest
from PIL import Image

from fakes import FakeModel, generation_model, make_scene
from visforge.records import Seed
from visforge.synthesis import generate


def test_generate_runs_all_six_templates(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    report = generate(str(scene), _lib(), {"model": generation_model()})
    assert report["failures"] == [] and report["duplicates"] == 0
    questions = [c.question for c in report["candidates"]]
    assert len(questions) == 6
    assert any("What is written" in q for q in questions)
    assert any("How many" in q for q in questions)
    assert any("largest" in q for q in questions)
    assert any("more numerous" in q for q in questions)
    assert any("On which side" in q for q in questions)
    assert any("corresponds to" in q for q in questions)
    # 嵌套在 params.labels 里的 ${槽位} 也要被替换，答案才是描述文本而非字面量
    compare = next(c for c in report["candidates"] if "more numerous" in c.question)
    assert compare.answer == "blue squares"
    for name in ("extreme", "same-page-match"):
        item = next(c for c in report["candidates"] if c.plan[0]["template"] == name)
        assert item.answer == "the leftmost blue square" and item.plan[-1]["value"]["bbox"]
    for cand in report["candidates"]:
        assert cand.seed.path == str(scene)
        assert cand.plan and cand.plan[0]["template"]
        assert cand.evidence
    assert all(info["attempts"] == 1 and info["candidates"] == 1 and info["failures"] == 0
               for info in report["summary"].values())


def test_generate_accepts_seed_and_records_group(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    seed = Seed(path=str(scene), group="report1", page=3)
    report = generate(seed, _lib(), {"model": generation_model()})
    assert report["candidates"][0].seed.group == "report1"
    assert report["candidates"][0].seed.page == 3


def test_generate_records_where_it_got_stuck(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    report = generate(str(scene), _lib(), {"model": generation_model(invisible_on="")})
    # 支持物读标签模板卡在 partner 一步；其余 5 条模板不受影响
    assert not any(c.plan[0]["template"] == "support-read-label" for c in report["candidates"])
    assert len(report["candidates"]) == 5
    assert len(report["failures"]) == 1
    failure = report["failures"][0]
    assert failure["template"] == "support-read-label"
    assert failure["step"] == "partner"
    assert report["operator_attempts"]["FindPartner"] == 1
    assert report["operator_attempts"].get("Read", 0) == 0
    assert report["operator_attempts"]["Select"] == 9
    assert "bottom contact of the cup is not visible" in failure["reason"]


def test_failed_select_trace_retains_only_bounded_final_text_and_generation_metrics(tmp_path):
    def model(request):
        if "writing questions" in request["text"]:
            return json.dumps({"choices": [_choice("support-read-label", target="label")]}), {}
        assert request["generation_diagnostics"] is True
        text = ('<think>hidden analysis</think>start Bearer FAKESECRET api_key="private-key-123" '
                'data:image/png;base64,abcdef123= ' + ("x" * 1800) + " tail")
        meta = {"api_key": "meta-secret", "generation": {
            "input_tokens": 9, "generated_tokens": 32, "max_new_tokens": 32, "finish_reason": "length",
            "last_token_id": 4, "image_grid_thw": [[1, 6, 4]], "reasoning": "private",
        }}
        return text, meta

    scene = make_scene(tmp_path / "x.png")
    report = generate(scene, _lib(), {"model": model, "budget": 1})
    failure, = report["failures"]
    trace = failure["diagnostic"]
    assert failure["step"] == "target" and trace["subtype"] == "invalid_json"
    assert failure["operator"] == "Select"
    assert trace["source_image_size"] == [640, 480]
    assert trace["generation"]["finish_reason"] == "length"
    assert trace["generation"]["image_grid_thw"] == [[1, 6, 4]]
    preview = trace["final_text_excerpt"]
    assert preview.startswith("start ") and preview.endswith(" tail") and len(preview) <= 1000
    assert trace["excerpt_truncated"] and len(trace["final_text_sha256"]) == 64
    saved = json.dumps(trace)
    for hidden in ("hidden analysis", "FAKESECRET", "meta-secret", "private", "abcdef123"):
        assert hidden not in saved


def test_generate_budget_limits_templates(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    report = generate(str(scene), _lib(), {"model": generation_model(), "budget": 1})
    assert len(report["candidates"]) == 1
    assert report["candidates"][0].plan[0]["template"] == "support-read-label"


def test_generate_propose_garbage_recorded(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = FakeModel(default="Sorry, I cannot tell")  # 提议调用只会拿到非 JSON
    report = generate(str(scene), _lib(), {"model": model})
    assert report["candidates"] == []
    assert [f["step"] for f in report["failures"]] == ["selection"]
    assert report["summary"] == {} and report["selection"]["status"] == "invalid"
    assert len(model.calls) == 1


def _choice(template, **slots):
    return {"template": template, "slots": slots, "evidence": "Two blue squares are visible on the trays."}


def _plan(model, choices):
    model.add("You are writing questions", json.dumps({"choices": choices}))


def test_generate_requires_proposal_slots_to_be_text(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = generation_model()
    _plan(model, [_choice("range-count", targets=True)])
    report = generate(scene, _lib(), {"model": model})
    assert not report["candidates"] and report["selection"]["status"] == "invalid"
    assert report["selection"]["issues"][0]["reason"] == "invalid_slots"


def test_generate_runs_multiple_instances_per_template(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = generation_model()
    _plan(model, [_choice("range-count", targets="blue squares"), _choice("range-count", targets="trays")])
    report = generate(scene, _lib(), {"model": model, "instances_per_template": 3})
    questions = [c.question for c in report["candidates"]]
    assert "How many blue squares are in the image?" in questions
    assert "How many trays are in the image?" in questions
    assert len(questions) == 2
    assert report["summary"]["range-count"] == {"attempts": 2, "candidates": 2, "failures": 0}


def test_generate_dedups_duplicate_slot_fillings_within_a_template(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = generation_model()
    _plan(model, [_choice("range-count", targets=t) for t in ("blue squares", " Blue  Squares ", "trays")])
    report = generate(scene, _lib(), {"model": model, "instances_per_template": 3})
    # 相同槽位填充只跑一次，不计失败
    assert report["summary"]["range-count"]["attempts"] == 2
    assert len([f for f in report["failures"] if f["template"] == "range-count"]) == 0
    assert report["selection"]["issues"][0]["reason"] == "duplicate_binding"


def test_generate_budget_limits_template_instance_pairs(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = generation_model()
    _plan(model, [_choice("range-count", targets="blue squares"), _choice("range-count", targets="trays"),
                  _choice("support-read-label", target="left blue square")])
    report = generate(scene, _lib(), {"model": model, "budget": 2})
    # 模板 1 用 1 个实例、模板 2 只跑到第 1 个实例即达预算
    assert report["summary"]["support-read-label"]["attempts"] == 1
    assert report["summary"]["range-count"]["attempts"] == 1
    assert "extreme" not in report["summary"]
    assert len(report["candidates"]) == 2
    assert [r["rank"] for r in report["selection"]["choices"]] == [1, 3]


def test_generate_deduplicates_identical_questions_per_image(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    steps = [
        {"id": "targets", "op": "Select", "params": {"description": "${targets}"}, "inputs": []},
        {"id": "n", "op": "Count", "params": {}, "inputs": ["targets"]},
    ]
    lib = {
        "operators": _lib()["operators"],
        "templates": [
            {"name": "t1", "propose": "Pick a countable group. counted clearly.",
             "slots": ["targets"], "steps": steps,
             "question": "How many objects are in the image?", "answer_from": "n"},
            {"name": "t2", "propose": "Pick another countable group. counted clearly.",
             "slots": ["targets"], "steps": steps,
             "question": "How many objects are in the image?", "answer_from": "n"},
        ],
    }
    model = generation_model()
    _plan(model, [_choice("t1", targets="blue squares"), _choice("t2", targets="blue squares")])
    report = generate(scene, lib, {"model": model})
    assert len(report["candidates"]) == 1
    assert report["duplicates"] == 1  # 同图重复问句跳过，不算失败
    assert report["failures"] == []
    assert report["summary"]["t2"]["candidates"] == 0
    assert report["summary"]["t2"]["failures"] == 0


def _lib():
    from visforge.operators import seed_library

    return seed_library()


def test_planner_sees_all_rules_once_in_canonical_order_and_can_pick_last(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    lib = _lib()
    selected = lib["templates"][-1]["name"]
    choice = _choice(selected, entity_desc="left blue square", candidates_desc="blue squares")
    model = generation_model()
    _plan(model, [choice])
    prompts = []
    for templates in [lib["templates"], list(reversed(lib["templates"]))]:
        model.calls.clear()
        report = generate(scene, {**lib, "templates": templates}, {"model": model, "budget": 1})
        assert [c.plan[0]["template"] for c in report["candidates"]] == [selected]
        assert report["candidates"][0].plan[0]["selection_rank"] == 1
        assert report["selection"]["selected_templates"] == [selected]
        assert report["selection"]["choices"][0]["slots"] == choice["slots"]
        planners = [r for r in model.calls if "You are writing questions" in r["text"]]
        assert len(planners) == 1 and planners[0]["images"] == [str(scene)]
        assert planners[0]["generation_diagnostics"] and planners[0]["max_new_tokens"] >= 512
        prompts.append(planners[0]["text"])
        catalog = json.loads(prompts[-1].split("Template library:\n")[1])
        assert [r["name"] for r in catalog] == sorted(t["name"] for t in templates)
        for row in catalog:
            original = next(t for t in templates if t["name"] == row["name"])
            assert row["propose"] == original["propose"] and row["question"] == original["question"]
            assert row["slots"] == original["slots"] and row["operators"] == [s["op"] for s in original["steps"]]
    assert prompts[0] == prompts[1]


@pytest.mark.parametrize("reply,status,reason", [
    ("[]", "abstained", None), ('{"choices":[]}', "abstained", None),
    ('{"choices":[{"template":"bad"}]}', "invalid", "unknown_template"),
    ('{"choices":[{"template":"extreme","slots":{"targets":"squares"}}]}', "invalid", "missing_evidence"),
])
def test_no_speculative_operator_calls_when_planner_cannot_choose(tmp_path, reply, status, reason):
    model = FakeModel(default=reply)
    r = generate(make_scene(tmp_path / "x.png"), _lib(), {"model": model, "budget": 2})
    assert len(model.calls) == 1 and r["selection"]["status"] == status and not r["candidates"]
    if reason:
        assert r["selection"]["issues"][0]["reason"] == reason
    else:
        assert not r["failures"]


def test_freeze_route_before_operators_and_resume_without_new_selection(tmp_path):
    scene = make_scene(tmp_path / "x.png")
    model = generation_model()
    _plan(model, [_choice("range-count", targets="trays"), _choice("range-count", targets="blue squares")])
    saved = []
    def checkpoint(selection):
        assert len(model.calls) == 1
        saved.append(json.loads(json.dumps(selection)))
        raise InterruptedError
    with pytest.raises(InterruptedError):
        generate(scene, _lib(), {"model": model, "on_selection": checkpoint, "instances_per_template": 1})
    route, = saved
    assert route["issues"][0]["reason"] == "instance_cap"
    model.calls.clear()
    report = generate(scene, _lib(), {"model": model, "frozen_selection": route, "instances_per_template": 1})
    assert report["selection"] == route
    assert not any("You are writing questions" in req["text"] for req in model.calls)
    assert [c.question for c in report["candidates"]] == ["How many trays are in the image?"]


def test_explicit_budget_is_reported_even_for_one_rule_and_exists_answers_are_text(tmp_path):
    lib = _lib()
    tpl = lib["templates"][1]
    tpl.update(steps=[tpl["steps"][0], {"id": "exists", "op": "Exists", "params": {}, "inputs": ["targets"]}],
               question="Are there {targets}?", answer_from="exists")
    lib["templates"] = [tpl]
    report = generate(make_scene(tmp_path / "x.png"), lib, {"model": generation_model(), "budget": 8})
    assert report["selection"]["budget"] == 8
    assert [(c.answer, c.plan[-1]["value"]) for c in report["candidates"]] == [("yes", True)]


def test_zero_budget_and_invalid_frozen_routes_never_call_model(tmp_path):
    scene = make_scene(tmp_path / "x.png")
    model = FakeModel()
    result = generate(scene, _lib(), {"model": model, "budget": 0})
    assert result["selection"]["status"] == "skipped" and not model.calls
    invalid = {**result["selection"], "budget": 1, "choices": [_choice("absent", target="x")]}
    with pytest.raises(ValueError, match="frozen"):
        generate(scene, _lib(), {"model": model, "budget": 1, "frozen_selection": invalid})
    assert not model.calls


def test_same_library_plans_differently_for_scene_and_blank_image(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    blank = tmp_path / "blank.png"
    Image.new("RGB", (640, 480), "white").save(blank)
    model = generation_model()

    def plan_by_pixels(request):
        with Image.open(request["images"][0]) as image:
            visible = image.getpixel((150, 200)) == (40, 80, 200)
        return json.dumps({"choices": [_choice("range-count", targets="blue squares")] if visible else []})

    model.add("You are writing questions", plan_by_pixels)
    first = generate(scene, _lib(), {"model": model, "budget": 1})
    second = generate(blank, _lib(), {"model": model, "budget": 1})
    assert first["selection"]["selected_templates"] == ["range-count"]
    assert second["selection"]["status"] == "abstained" and not second["candidates"]


def test_planner_catalog_exposes_changed_guards_parameters_and_input_order(tmp_path):
    scene = make_scene(tmp_path / "x.png")
    lib = _lib()
    ordinary = next(t for t in lib["templates"] if t["name"] == "locate")
    changed = deepcopy(ordinary)
    changed["name"] = "locate-vertical"
    changed["steps"][0]["params"]["require_unique"] = False
    changed["steps"][-1]["params"]["axis"] = "vertical"
    changed["steps"][-1]["inputs"].reverse()
    lib["templates"] = [changed, ordinary]
    locate = next(op for op in lib["operators"] if op["name"] == "Locate")
    locate["prompt"] = "Compare only the vertical boxes if both labels are legible."
    model = FakeModel(default='{"choices":[]}')
    generate(scene, lib, {"model": model, "budget": 1})
    prefix, catalog = model.calls[0]["text"].split("Template library:\n")
    definitions = json.loads(prefix.split("Operator definitions:\n")[1])
    assert [op["name"] for op in definitions] == ["Locate", "Select"]
    assert definitions[0]["prompt"] == locate["prompt"] and definitions[0]["description"] == locate["description"]
    base, vertical = json.loads(catalog)
    assert base["operators"] == vertical["operators"]
    assert base["steps"] == ordinary["steps"] and vertical["steps"] == changed["steps"]
    assert vertical["answer_from"] == changed["answer_from"]


def test_invalid_planner_response_keeps_only_sanitized_excerpt_and_stop_evidence(tmp_path):
    def cutoff(request):
        visible = 'BAD Bearer TOPSECRET api_key="secret-field" data:image/jpeg;base64,SGVsbG8= '
        return "<think>private chain</think>" + visible + ("z" * 1900) + " END", {
            "model": "local", "api_key": "ignored", "generation": {"finish_reason": "length", "generated_tokens": 512},
        }
    trace = generate(make_scene(tmp_path / "x.png"), _lib(), {"model": cutoff})["selection"]
    assert trace["status"] == "invalid" and trace["generation"]["finish_reason"] == "length"
    sample = trace["response_excerpt"]
    assert sample["excerpt_truncated"] and len(sample["final_text_sha256"]) == 64
    assert sample["final_text_excerpt"].startswith("BAD ") and sample["final_text_excerpt"].endswith(" END")
    assert len(sample["final_text_excerpt"]) <= 1000
    assert all(value not in json.dumps(trace) for value in ("TOPSECRET", "secret-field", "SGVsbG8", "private chain", "ignored"))


def test_object_answer_with_indistinguishable_names_and_overlapping_boxes_fails_composition(tmp_path):
    model = generation_model()
    _plan(model, [_choice("same-page-match", entity_desc="left blue square", candidates_desc="blue squares")])
    model.add("Description: blue squares", json.dumps({"objects": [
        {"name": "square", "bbox": [10, 20, 30, 40]}, {"name": "square", "bbox": [11, 21, 31, 41]},
    ]}))
    report = generate(make_scene(tmp_path / "scene.png"), _lib(), {"model": model})
    assert not report["candidates"]
    assert [(f["step"], f["selection_rank"]) for f in report["failures"]] == [("compose", 1)]
