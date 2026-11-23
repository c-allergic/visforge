"""进化：observer 提改动、套改动、两版比较的两个硬门槛。"""

import json
from copy import deepcopy

import pytest

from fakes import FakeModel, generation_model, make_scene, solver_model
from visforge.evolution import apply_edit, compare_libraries, evaluate_library, propose_edit
from visforge.operators import seed_library
from visforge.records import Seed
from visforge.synthesis import generate


def test_propose_edit_modifies_operator():
    lib = seed_library()
    model_response = json.dumps(
        {"action": "modify", "name": "FindPartner",
         "description": "find the supporting object; the bottom-contact evidence must be kept",
         "prompt": "Confirm the contact is visible, then give the supporting object."},
        ensure_ascii=False,
    )
    model = FakeModel({"You are the rule-library observer": model_response})
    edit = propose_edit(lib, [{"question": "q", "blind": "wrong", "reference": "right"}], model)
    assert edit and edit["action"] == "modify" and edit["name"] == "FindPartner"
    new = apply_edit(lib, edit)
    assert new is not lib
    old_op = next(o for o in lib["operators"] if o["name"] == "FindPartner")
    new_op = next(o for o in new["operators"] if o["name"] == "FindPartner")
    assert new_op["description"] == "find the supporting object; the bottom-contact evidence must be kept"
    assert "bottom-contact evidence" not in old_op["description"]  # 旧库不动


def test_propose_edit_returns_none_on_garbage():
    model = FakeModel(default="I am not JSON")
    assert propose_edit(seed_library(), [], model) is None


def test_propose_edit_rejects_modify_of_missing_operator():
    model = FakeModel(default=json.dumps({"action": "modify", "name": "Nope", "description": "x"}))
    assert propose_edit(seed_library(), [], model) is None


def test_observer_can_repair_one_invalid_schema_into_a_typed_composition():
    answers = iter([
        '{"action":"add_template","template":{}}',
        json.dumps({"action": "read_object", "name": "read-single-label",
                    "propose": "Select a uniquely identifiable labeled object.",
                    "what": "the label text", "question": "What label is printed on {target}?"}),
    ])
    prompts, log = [], []
    def model(request):
        prompts.append(request["text"])
        return next(answers), {"model": "observer-test"}

    old = seed_library()
    edit = propose_edit(old, [{"template": "range-count", "reason": "nonunique items"}], model,
                        budget=1, on_response=log.append, repair=True)
    assert [row["outcome"] for row in log] == ["invalid", "edit"]
    assert [row["index"] for row in log] == [1, 2]
    assert log[0]["validation_error"] in prompts[1] and "nonunique items" in prompts[1]
    assert log[0]["raw"] == '{"action":"add_template","template":{}}'
    assert log[1]["model"] == "observer-test" and not log[1]["validation_error"]
    tpl = next(t for t in apply_edit(old, edit, budget=1)["templates"] if t["name"] == "read-single-label")
    assert [s["op"] for s in tpl["steps"]] == ["Select", "Read"]
    assert tpl["steps"][-1]["params"]["what"] == "the label text"
    assert tpl["question"] == "What label is printed on {target}?"


@pytest.mark.parametrize("reply", ["NO_EDIT", "No edit: too little evidence.", '{"action":"no_edit"}'])
def test_explicit_observer_decline_is_final(reply):
    model, log = FakeModel(default=reply), []
    assert propose_edit(seed_library(), [], model, repair=True, on_response=log.append) is None
    assert [r["outcome"] for r in log] == ["declined"]
    assert len(model.calls) == 1


def test_select_prompt_short_form_and_bounded_invalid_replies():
    revised = "Find one clearly visible object matching the description and provide its full bounds."
    old = seed_library()
    edit = propose_edit(old, [], FakeModel(default="SELECT_PROMPT: " + revised), repair=True)
    assert apply_edit(old, edit)["operators"][0]["prompt"] == revised
    log = []
    model = FakeModel(default="unsupported action " * 2000)
    assert propose_edit(old, [], model, repair=True, on_response=log.append) is None
    assert len(model.calls) == 2 and all(r["outcome"] == "invalid" and r["validation_error"] for r in log)
    assert all(r["raw_truncated"] and len(r["raw"]) <= 8192 for r in log)


def _libs():
    """旧库只留计数模板（易），新库只留读标签模板（难）。"""
    lib = seed_library()
    old = {"name": "old", "operators": lib["operators"], "templates": [lib["templates"][1]]}
    new = {"name": "new", "operators": lib["operators"], "templates": [lib["templates"][0]]}
    return old, new


def _holdout(tmp_path):
    easy = make_scene(tmp_path / "easy.png")
    # 注意：标记串不能出现在测试自身的临时目录名里，所以用 "blocked"
    hard = make_scene(tmp_path / "blocked.png")  # 新版在这张图上找不到接触证据
    return [Seed(str(easy), "hold"), Seed(str(hard), "hold")]


def test_compare_adopts_when_harder_and_yield_ok(tmp_path):
    old, new = _libs()
    settings = {"model": generation_model(invisible_on="blocked"), "solver": solver_model(),
                "min_yield_per_image": 0.4}
    decision = compare_libraries(old, new, _holdout(tmp_path), settings)
    assert decision["adopted"] is True
    assert decision["new"]["h_s"] > decision["old"]["h_s"]
    # 失败图计入产量：2 张留出图只出 1 道合格题
    assert decision["new"]["failed_images"] == 1
    assert decision["new"]["yield_per_image"] == 0.5
    assert decision["old"]["yield_per_image"] == 1.0  # 每图 1 道计数题，两张都成功
    assert decision["old"]["failed_images"] == 0
    assert decision["new"]["coverage"] == {"support-read-label": 1}


def test_compare_rejects_when_yield_below_floor(tmp_path):
    old, new = _libs()
    settings = {"model": generation_model(invisible_on="blocked"), "solver": solver_model(),
                "min_yield_per_image": 0.75}
    decision = compare_libraries(old, new, _holdout(tmp_path), settings)
    assert decision["adopted"] is False
    assert "below floor" in decision["reason"]  # 更难但产量不达标，仍不采纳
    assert decision["new"]["h_s"] > decision["old"]["h_s"]


def test_compare_rejects_when_not_harder(tmp_path):
    old, new = _libs()
    # solver 对读标签题也答对 → 新版不再更难
    solver = solver_model()
    solver.default = "TEA"
    settings = {"model": generation_model(invisible_on=None), "solver": solver,
                "min_yield_per_image": 0.4}
    decision = compare_libraries(old, new, _holdout(tmp_path), settings)
    assert decision["adopted"] is False
    assert "h_s not improved" in decision["reason"]


def _template_edit():
    return {"action": "add_template", "template": {
        "name": "read-one-object",
        "propose": "Pick a concrete object whose support is visible and whose own text is legible.",
        "slots": ["target"],
        "steps": [
            {"id": "obj", "op": "Select", "inputs": [],
             "params": {"description": "${target}", "require_unique": True}},
            {"id": "label", "op": "Read", "inputs": ["obj"], "params": {"what": "object label"}},
        ],
        "question": "What is the label on {target}?",
        "answer_from": "label",
    }}


def test_observer_template_executes_when_chosen_under_same_budget(tmp_path):
    old = seed_library()
    proposed = _template_edit()
    observer = FakeModel(default=json.dumps(proposed))
    edit = propose_edit(old, [{"question": "Which label?"}], observer, budget=1)
    assert edit == proposed
    new = apply_edit(old, edit, budget=1)
    scene = make_scene(tmp_path / "scene.png")
    selected = generate(scene, new, {"model": generation_model(), "budget": 1})["candidates"]
    assert [(c.question, c.answer) for c in selected] == [("What is the label on the left blue square?", "TEA")]
    assert [s["op"] for s in selected[0].plan[1:]] == ["Select", "Read"]
    assert old == seed_library()


@pytest.mark.parametrize("change", [
    {"name": "support-read-label"}, {"answer_from": "absent"},
    {"question": "Read {target.missing}"}, {"question": "Read {unbound}"},
    {"slots": ["target", "target"]}, {"slots": ["target[0]"]},
    {"steps": [{"id": "x", "op": "BrandNew", "params": {}, "inputs": []}]},
    {"steps": [{"id": "x", "op": "Read", "params": {}, "inputs": ["later"]}]},
])
def test_observer_declines_unusable_templates(change):
    edit = _template_edit()
    edit["template"].update(change)
    assert propose_edit(seed_library(), [], FakeModel(default=json.dumps(edit))) is None
    with pytest.raises(ValueError):
        apply_edit(seed_library(), edit)


@pytest.mark.parametrize("variant", ["rename_only", "wrong_slot", "wrong_input_type", "unused_step", "wrong_scale"])
def test_template_must_change_an_executable_chain(variant):
    lib = seed_library()
    if variant == "rename_only":
        template = deepcopy(lib["templates"][0])
        template["name"] = "renamed"
    else:
        template = _template_edit()["template"]
        if variant == "wrong_slot":
            template["steps"][0]["params"]["description"] = "object ${target}"
        if variant == "wrong_input_type":
            template["steps"][0]["params"]["require_unique"] = False
        if variant == "unused_step":
            template["steps"].append(dict(template["steps"][1], id="ignored"))
        if variant == "wrong_scale":
            template["steps"][1]["params"]["scale"] = "large"
    with pytest.raises(ValueError):
        apply_edit(lib, {"action": "add_template", "template": template})


@pytest.mark.parametrize("edit", [
    {"action": "modify", "name": "Read", "description": "only notes change"},
    {"action": "modify", "name": "Count", "prompt": "Count carefully"},
    {"action": "modify", "name": "FindPartner", "prompt": "Check visible contact"},
    {"action": "modify", "name": "Select", "prompt": seed_library()["operators"][0]["prompt"]},
    {"action": "add", "name": "Invented", "prompt": "Does not have a handler"},
])
def test_only_changes_to_active_executable_prompts_are_eligible(edit):
    old, _ = _libs()  # only Select and Count are scheduled
    model = FakeModel(default=json.dumps(edit))
    assert propose_edit(old, [], model, budget=1) is None
    with pytest.raises(ValueError):
        apply_edit(old, edit, budget=1)


def test_compare_measures_target_and_reference_after_independent_audit(tmp_path):
    old, new = _libs()
    seed = Seed(str(make_scene(tmp_path / "hold.png")), "whole-document", page=5)
    verifier = FakeModel({"You are a question auditor": '{"verdict":"accept","reason":"visible"}'},
                         default="blind verifier response")
    target = FakeModel({"How many": "2"}, default="target guess")
    reference = FakeModel({"How many": "2"}, default="TEA")
    decision = compare_libraries(old, new, [seed], {
        "model": generation_model(verdict="reject"), "verifier": verifier, "solver": target,
        "reference_solver": reference, "min_reference_accuracy": 1, "k": 2,
    })
    assert decision["adopted"]
    assert decision["new"]["reference_accuracy"] == 1
    record, = decision["new"]["records"]
    assert record["seed"] == {"path": seed.path, "group": "whole-document", "page": 5}
    assert record["audit"]["blind_answer"] == "blind verifier response"
    assert record["target_correct"] == [False, False]
    assert record["reference_correct"] == [True, True]
    assert len(target.calls) == len(reference.calls) == 4
    assert all("Reference answer" not in req["text"] for req in target.calls + reference.calls)


def test_compare_deduplicates_verified_questions_before_scoring(tmp_path):
    old, new = _libs()
    for lib in (old, new):
        clone = deepcopy(lib["templates"][0])
        clone["name"] += "-alias"
        clone["question"] = clone["question"].replace("?", "??")
        lib["templates"].append(clone)
    decision = compare_libraries(old, new, _holdout(tmp_path), {
        "model": generation_model(), "solver": solver_model(), "k": 2,
        "min_coverage": 2,
    })
    assert not decision["adopted"]
    # generate() 已在同图上把规范化后相同的问句去重，evaluate 层不再见到重复
    assert decision["new"]["accepted"] == decision["new"]["questions"] == 2
    assert decision["new"]["duplicates"] == 0
    assert len(decision["new"]["per_question_correct"]) == 4
    assert decision["new"]["yield_per_image"] == 1


def test_generate_level_dedup_keeps_alias_questions_out_of_the_export(tmp_path):
    # 同图上“??”结尾的别名题与原题规范化后相同 → generate 层跳过
    lib = seed_library()
    clone = deepcopy(lib["templates"][0])
    clone["name"] += "-alias"
    clone["question"] = clone["question"].replace("?", "??")
    lib["templates"].append(clone)
    report = generate(make_scene(tmp_path / "scene.png"), lib, {"model": generation_model()})
    assert len(report["candidates"]) == 6
    assert report["duplicates"] == 1
    assert report["failures"] == []


def test_zero_verified_parent_cannot_establish_difficulty_improvement(tmp_path):
    old, new = _libs()
    generator = generation_model()
    generator.add("counted clearly", "not JSON")
    decision = compare_libraries(old, new, _holdout(tmp_path), {"model": generator, "solver": solver_model()})
    assert decision["new"]["h_s"] == 1
    assert not decision["adopted"]
    assert len(decision["old"]["failures"]) == 2
    assert decision["old"]["failed_images"] == 2


def test_reference_floor_can_reject_harder_verified_questions(tmp_path):
    old, new = _libs()
    decision = compare_libraries(old, new, _holdout(tmp_path), {
        "model": generation_model(), "solver": solver_model(), "reference_solver": solver_model(),
        "min_reference_accuracy": 0.9,
    })
    assert not decision["adopted"]
    assert decision["new"]["reference_accuracy"] == 0
    assert decision["new"]["h_s"] == 1


def test_compare_assigns_equal_template_attempts_with_different_library_sizes(tmp_path):
    old = seed_library()
    new = apply_edit(old, _template_edit())
    decision = compare_libraries(old, new, _holdout(tmp_path), {"model": generation_model(), "solver": solver_model()})
    assert decision["budget"] == 8
    for arm, library in (("old", old), ("new", new)):
        metrics = decision[arm]
        assert metrics["available_templates"] == sorted(t["name"] for t in library["templates"])
        assert len(metrics["selections"]) == 2
        for item in metrics["selections"]:
            assert item["selection"]["budget"] == 8
            assert set(item["selection"]["selected_templates"]) == set(item["summary"])
            assert len(item["selection"]["choices"]) <= 8


def test_an_empty_target_response_cannot_be_scored_as_a_harder_question(tmp_path):
    lib, _ = _libs()
    verifier = FakeModel({"You are a question auditor": '{"verdict":"accept","reason":"visible"}'}, default="2")
    with pytest.raises(ValueError, match="empty"):
        evaluate_library(lib, _holdout(tmp_path), {
            "model": generation_model(), "verifier": verifier, "solver": FakeModel(default=" "),
        })


def test_renaming_internal_steps_does_not_make_an_edit(tmp_path):
    lib = seed_library()
    clone = deepcopy(lib["templates"][1])
    clone["name"] = "same-count"
    clone["steps"][0]["id"] = "renamed"
    clone["steps"][1]["inputs"] = ["renamed"]
    with pytest.raises(ValueError):
        apply_edit(lib, {"action": "add_template", "template": clone})


def test_validator_accepts_available_branching_numeric_operators_and_all_seed_shapes():
    lib = seed_library()
    for source in lib["templates"]:
        template = dict(deepcopy(source), name="changed-" + source["name"],
                        propose=source["propose"] + " Choose labels that are fully legible.")
        changed = apply_edit(lib, {"action": "add_template", "template": template}, budget=1)
        assert changed["templates"][:-1] == lib["templates"]
        assert changed["templates"][-1] == template
    calc = deepcopy(next(t for t in lib["templates"] if t["name"] == "count-compare"))
    calc.update(name="count-difference", question="How many more {a_desc} than {b_desc} are there?")
    calc["steps"][-1].update(op="Calculate", params={"op": "-"})
    assert apply_edit(lib, {"action": "add_template", "template": calc})["templates"][-1] == calc
    with pytest.raises(ValueError):
        calc["steps"][-1]["inputs"] = ["a_sel", "b_count"]
        apply_edit(lib, {"action": "add_template", "template": calc})


def test_observer_can_modify_later_reachable_model_operator_without_position_bias():
    old = seed_library()
    edit = {"action": "modify", "name": "Match", "prompt": "Read the legend before comparing each candidate."}
    assert propose_edit(old, [], FakeModel(default=json.dumps(edit)), budget=1) == edit


def test_explicit_attempt_budget_applies_to_both_arms_with_only_one_template(tmp_path, monkeypatch):
    from visforge import evolution
    old, new = _libs()
    calls = []
    def abstain(seed, library, settings):
        calls.append((library["name"], settings["budget"], settings["instances_per_template"]))
        sel = {"status": "abstained", "selected_templates": [], "choices": [], "issues": ["no evidence"]}
        return {"candidates": [], "summary": {}, "duplicates": 0, "failures": [], "selection": sel}
    monkeypatch.setattr(evolution, "generate", abstain)
    decision = compare_libraries(old, new, _holdout(tmp_path),
                                 {"model": None, "solver": None, "instances_per_template": 3}, budget=8)
    assert calls == [("old", 8, 3)] * 2 + [("new", 8, 3)] * 2
    for label in ("old", "new"):
        assert [s["selection"]["issues"] for s in decision[label]["selections"]] == [["no evidence"]] * 2
        assert decision[label]["selection_counts"] == {"abstained": 2}
        assert decision[label]["attempts"] == 0


def test_observer_summary_counts_every_example_without_copying_all_audit_text():
    samples = [{"template": "locate", "question": "Which side?", "answer": "left", "status": "accepted",
                "target_correct": [False], "reference_correct": [True],
                "audit": {"reason": "clear position " * 300}, "seed": {"group": f"doc-{i % 16}"}}
               for i in range(128)]
    observer = FakeModel(default="NO_EDIT")
    propose_edit(seed_library(), samples, observer)
    request = observer.calls[0]
    summary = json.loads(request["text"].split("Development results: ")[1])
    assert summary["by_template"]["locate"]["accepted"] == 128
    assert summary["by_template"]["locate"]["target_samples"] == 128
    assert summary["by_template"]["locate"]["reference_correct"] == 128
    assert len(request["text"]) < 32000
    assert request["max_new_tokens"] >= 1024


def test_observer_distinguishes_truncated_replies_from_other_operator_failures():
    records = [
        {"selection": {"status": "selected", "selected_templates": [], "issues": [], "choices": []},
         "operator_attempts": {"Select": 5, "Read": 2}},
        {"template": "count", "step": "objects", "operator": "Select",
         "diagnostic": {"subtype": "invalid_json", "generation": {"finish_reason": "length"}}},
        {"template": "heading", "step": "region", "operator": "Select",
         "diagnostic": {"subtype": "wrong_schema"}},
        {"template": "heading", "step": "text", "operator": "Read",
         "diagnostic": {"subtype": "insufficient_evidence"}},
    ]
    model = FakeModel(default="NO_EDIT")
    propose_edit(seed_library(), records, model)
    summary = json.loads(model.calls[0]["text"].split("Development results: ")[1])
    assert summary["operator_attempts"] == {"Select": 5, "Read": 2}
    assert summary["operator_failures"] == {
        "Select": {"truncated_json": 1, "wrong_schema": 1}, "Read": {"insufficient_evidence": 1}}
    assert summary["by_template"]["heading"]["target_samples"] == 0
