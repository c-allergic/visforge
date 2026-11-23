"""Exercise edited components through the planner and the normal synthesis entry point."""

import copy
import json

import pytest

from fakes import generation_model, make_scene
from visforge.evolution import apply_edit, propose_edit
from visforge.operators import seed_library
from visforge.synthesis import generate


def composition_edit():
    return {
        "action": "batch",
        "hypothesis": "Use the supporting object as the region for reading its printed label.",
        "edits": [
            {"action": "add_composition", "operator": {
                "name": "read_support", "description": "Find the support and read its visible label.",
                "inputs": [{"id": "target", "type": "object"}], "slots": ["field"],
                "steps": [
                    {"id": "base", "op": "FindPartner", "inputs": ["target"]},
                    {"id": "text", "op": "Read", "inputs": ["base"], "params": {"what": "${field}"}},
                ],
                "answer_from": "text", "result": "read_value",
            }},
            {"action": "update_template", "name": "support-read-label", "changes": {
                "steps": [
                    {"id": "target", "op": "Select", "params": {"description": "${target}", "require_unique": True}},
                    {"id": "reading", "op": "read_support", "inputs": ["target"], "params": {"field": "label"}},
                ],
                "answer_from": "reading",
            }},
            {"action": "set_planner_prompt", "prompt": "Prioritize relations with an observable supporting surface."},
        ],
    }


def test_compound_edit_executes_through_the_planner(tmp_path):
    model = generation_model()
    scene = make_scene(tmp_path / "scene.png")
    original = seed_library()
    edit = composition_edit()
    candidate = apply_edit(original, edit)
    report = generate(scene, candidate, {"model": model})
    request = model.calls[0]["text"]
    assert "Prioritize relations with an observable supporting surface." in request
    assert "Respect operator requirements" in request
    assert '"read_support"' in request
    assert '"op":"FindPartner"' in request and '"op":"Read"' in request
    question = next(row for row in report["candidates"] if row.plan[0]["template"] == "support-read-label")
    assert question.answer == "TEA"
    assert [step["op"] for step in question.plan[1:]] == ["Select", "FindPartner", "Read"]
    assert report["operator_attempts"]["read_support"] == 1
    assert original == seed_library()


@pytest.mark.parametrize("failure", ["type", "unbound", "cycle", "unused", "no-op"])
def test_bad_component_edits_cannot_change_the_parent(failure):
    edit = composition_edit()
    composition = edit["edits"][0]["operator"]
    if failure == "type":
        composition["inputs"][0]["type"] = "numeric"
    elif failure == "unbound":
        composition["steps"][1]["params"]["what"] = "${unbound}"
    elif failure == "cycle":
        composition["steps"][0]["op"] = "read_support"
    elif failure == "unused":
        edit["edits"].pop(1)
    else:
        edit = {"action": "update_template", "name": "support-read-label",
                "changes": {"answer_from": seed_library()["templates"][0]["answer_from"]}}
    original = seed_library()
    snapshot = copy.deepcopy(original)
    with pytest.raises(ValueError):
        apply_edit(original, edit)
    assert original == snapshot


def test_observer_lists_real_edit_actions_without_a_ready_made_template():
    seen = []

    def observe(request):
        seen.append(request["text"])
        return json.dumps(composition_edit()), {}

    assert propose_edit(seed_library(), [], observe, budget=8) == composition_edit()
    assert all(action in seen[0] for action in ("set_planner_prompt", "update_template", "add_composition"))
    assert "heading-read" not in seen[0]


def test_adding_a_template_with_the_same_expanded_plan_is_a_duplicate():
    edit = composition_edit()
    original = seed_library()
    previous = original["templates"][0]
    revised = {"name": "another-support-read", **{key: value for key, value in previous.items() if key != "name"}}
    revised.update(edit["edits"][1]["changes"])
    revised["steps"][1]["params"]["field"] = previous["steps"][-1]["params"]["what"]
    edit["edits"] = [edit["edits"][0], {"action": "add_template", "template": revised}]
    with pytest.raises(ValueError, match="does not change generation"):
        apply_edit(original, edit)
