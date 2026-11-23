"""算子增广：Identify / Locate / Exists / Compare / Extreme / Calculate / Match 与 Select.allow_empty。"""

import json

import pytest

from fakes import BLUE_BOX, BLUE_BOX_2, TRAY_A, TRAY_B, generation_model, make_scene
from visforge.operators import OperatorResult, apply_operator, parse_value, seed_library


def _op(library, name):
    return next(o for o in library["operators"] if o["name"] == name)


def _rejecting_model(request):
    raise AssertionError("this operator must not call the model")


def _ctx(tmp_path, steps, step, model=_rejecting_model):
    return {
        "image": str(make_scene(tmp_path / "scene.png")),
        "model": model,
        "steps": steps,
        "step": step,
    }


def _obj(bbox):
    return {"name": "blue square", "bbox": list(bbox)}


def test_parse_value_extracts_number_and_unit():
    assert parse_value(3) == (3.0, None)
    assert parse_value("5 cm") == (5.0, "cm")
    assert parse_value("about -2.5mm") == (-2.5, "mm")
    assert parse_value("80 pixels") == (80.0, "pixels")
    assert parse_value("no digits") is None
    assert parse_value(None) is None
    assert parse_value(True) is None


def test_select_allow_empty_returns_ok_empty_list(tmp_path):
    model = generation_model()
    model.add("list all objects matching", '{"objects": []}')
    params = {"description": "dinosaur", "allow_empty": True}
    r = apply_operator(_op(seed_library(), "Select"),
                       _ctx(tmp_path, {}, {"id": "t", "params": params, "inputs": []}, model))
    assert r.ok and r.value == [] and r.evidence == []


def test_select_default_still_fails_on_empty(tmp_path):
    model = generation_model()
    model.add("list all objects matching", '{"objects": []}')
    r = apply_operator(_op(seed_library(), "Select"),
                       _ctx(tmp_path, {}, {"id": "t", "params": {"description": "dinosaur"}, "inputs": []}, model))
    assert not r.ok and "no object matching" in r.failure_reason


def test_identify_keeps_single_match(tmp_path):
    steps = {"t": OperatorResult(True, [_obj(BLUE_BOX)])}
    r = apply_operator(_op(seed_library(), "Identify"),
                       _ctx(tmp_path, steps, {"id": "i", "params": {"distinguisher": "the left one"}, "inputs": ["t"]}))
    assert r.ok and r.value["bbox"] == BLUE_BOX
    assert r.evidence[0]["distinguisher"] == "the left one"


def test_identify_fails_on_multiple_or_empty(tmp_path):
    lib = seed_library()
    steps = {"t": OperatorResult(True, [_obj(BLUE_BOX), _obj(BLUE_BOX_2)])}
    r = apply_operator(_op(lib, "Identify"),
                       _ctx(tmp_path, steps, {"id": "i", "params": {}, "inputs": ["t"]}))
    assert not r.ok and "not unique, cannot ask which one" in r.failure_reason
    steps = {"t": OperatorResult(True, [])}
    r = apply_operator(_op(lib, "Identify"),
                       _ctx(tmp_path, steps, {"id": "i", "params": {}, "inputs": ["t"]}))
    assert not r.ok and "no object matching" in r.failure_reason


def test_locate_is_pure_geometry_without_model_calls(tmp_path):
    steps = {"e": OperatorResult(True, _obj(BLUE_BOX)), "a": OperatorResult(True, {"name": "tray", "bbox": TRAY_B})}
    r = apply_operator(_op(seed_library(), "Locate"),
                       _ctx(tmp_path, steps, {"id": "side", "params": {"axis": "horizontal"}, "inputs": ["e", "a"]}))
    assert r.ok and r.value == "left"  # 蓝方块中心 x=180 < 托盘 B 中心 x=460
    assert r.evidence[0]["role"] == "entity" and r.evidence[1]["role"] == "anchor"
    assert r.evidence[2] == {"axis": "horizontal", "offset": -280.0}


def test_locate_vertical_axis(tmp_path):
    steps = {"e": OperatorResult(True, _obj(BLUE_BOX)), "a": OperatorResult(True, {"name": "tray", "bbox": TRAY_A})}
    r = apply_operator(_op(seed_library(), "Locate"),
                       _ctx(tmp_path, steps, {"id": "side", "params": {"axis": "vertical"}, "inputs": ["e", "a"]}))
    assert r.ok and r.value == "above"  # 中心 y=220 < 托盘 A 中心 y=350


def test_locate_fails_when_positions_are_indistinguishable(tmp_path):
    near = [150, 200, 230, 280]  # 中心与蓝方块几乎重合
    steps = {"e": OperatorResult(True, _obj(BLUE_BOX)), "a": OperatorResult(True, _obj(near))}
    r = apply_operator(_op(seed_library(), "Locate"),
                       _ctx(tmp_path, steps, {"id": "side", "params": {"axis": "horizontal"}, "inputs": ["e", "a"]}))
    assert not r.ok and "positions not distinguishable along the axis" in r.failure_reason


def test_exists_true_from_selection_without_model(tmp_path):
    calls = []
    def model(request):
        calls.append(request)
        return "{}", {}
    steps = {"t": OperatorResult(True, [_obj(BLUE_BOX), _obj(BLUE_BOX_2)])}
    r = apply_operator(_op(seed_library(), "Exists"),
                       _ctx(tmp_path, steps, {"id": "x", "params": {}, "inputs": ["t"]}, model))
    assert r.ok and r.value is True and calls == []


def test_exists_false_after_confirmed_full_scope_search(tmp_path):
    model = generation_model()
    model.add("ENTIRE image", '{"found": false, "full_scope_checked": true}')
    steps = {"t": OperatorResult(True, [])}
    r = apply_operator(_op(seed_library(), "Exists"),
                       _ctx(tmp_path, steps, {"id": "x", "params": {"description": "dinosaur"}, "inputs": ["t"]}, model))
    assert r.ok and r.value is False


def test_exists_fails_without_full_scope_confirmation(tmp_path):
    model = generation_model()
    model.add("ENTIRE image", '{"found": false, "full_scope_checked": false}')
    steps = {"t": OperatorResult(True, [])}
    r = apply_operator(_op(seed_library(), "Exists"),
                       _ctx(tmp_path, steps, {"id": "x", "params": {}, "inputs": ["t"]}, model))
    assert not r.ok and "cannot confirm absence without a full-scope search" in r.failure_reason


def test_compare_names_the_larger_operand(tmp_path):
    steps = {
        "a": OperatorResult(True, 2, [{"counted": 2}]),
        "b": OperatorResult(True, 1, [{"counted": 1}]),
    }
    r = apply_operator(_op(seed_library(), "Compare"),
                       _ctx(tmp_path, steps, {"id": "c", "params": {"labels": ["blue squares", "trays"]},
                                              "inputs": ["a", "b"]}))
    assert r.ok and r.value == "blue squares"
    assert r.evidence[0]["parsed"] == 2.0 and r.evidence[0]["provenance"] == [{"counted": 2}]


def test_compare_fails_on_units_or_equal_values(tmp_path):
    lib = seed_library()
    steps = {"a": OperatorResult(True, "5 cm"), "b": OperatorResult(True, "3 in")}
    r = apply_operator(_op(lib, "Compare"),
                       _ctx(tmp_path, steps, {"id": "c", "params": {"labels": ["a", "b"]}, "inputs": ["a", "b"]}))
    assert not r.ok and "units differ, not on a common scale" in r.failure_reason
    steps = {"a": OperatorResult(True, "5 cm"), "b": OperatorResult(True, "5.04 cm")}
    r = apply_operator(_op(lib, "Compare"),
                       _ctx(tmp_path, steps, {"id": "c", "params": {"labels": ["a", "b"]}, "inputs": ["a", "b"]}))
    assert not r.ok and "values are equal, no distinguishable order" in r.failure_reason


def test_extreme_picks_unique_max_from_model_readings(tmp_path):
    model = generation_model()
    steps = {"t": OperatorResult(True, [_obj(BLUE_BOX), _obj(BLUE_BOX_2)])}
    r = apply_operator(_op(seed_library(), "Extreme"),
                       _ctx(tmp_path, steps, {"id": "x", "params": {"field": "size", "direction": "max"},
                                              "inputs": ["t"]}, model))
    assert r.ok and r.value["bbox"] == BLUE_BOX
    assert [e.get("value") for e in r.evidence[:-1]] == [80.0, 50.0]
    assert r.evidence[-1]["winner"] == "blue square"


def _extreme_readings(tmp_path, readings):
    model = generation_model()
    model.add("Output only a JSON array", json.dumps(readings))
    candidates = OperatorResult(True, [_obj(BLUE_BOX), _obj(BLUE_BOX_2)])
    return apply_operator(_op(seed_library(), "Extreme"),
                          _ctx(tmp_path, {"t": candidates},
                               {"id": "x", "params": {"field": "height"}, "inputs": ["t"]}, model))


@pytest.mark.parametrize("readings", [
    [{"name": "blue square", "bbox": BLUE_BOX_2, "value": "50 pixels"},
     {"name": "blue square", "bbox": BLUE_BOX, "value": "80 pixels"}],
    [{"candidate_id": 1, "value": "50 pixels"}, {"candidate_id": 0, "value": "80 pixels"}],
    {"readings": [{"candidate_id": 1, "value": "50 pixels"}, {"candidate_id": 0, "value": "80 pixels"}]},
])
def test_extreme_attributes_shuffled_readings_to_their_objects(tmp_path, readings):
    result = _extreme_readings(tmp_path, readings)
    assert result.ok and result.value["bbox"] == BLUE_BOX
    assert [(e["bbox"], e["value"]) for e in result.evidence[:-1]] == [(BLUE_BOX, 80.), (BLUE_BOX_2, 50.)]


@pytest.mark.parametrize("readings", [
    [{"candidate_id": 0, "value": 3}, {"candidate_id": 0, "value": 9}],
    [{"name": "blue square", "value": 3}, {"name": "blue square", "value": 9}],
    [{"candidate_id": 0, "bbox": BLUE_BOX_2, "value": 3}, {"candidate_id": 1, "value": 9}],
    [{"candidate_id": 0, "value": "3 m"}, {"candidate_id": 1, "value": "9 cm"}],
    [{"candidate_id": 0, "value": "3 m"}, {"candidate_id": 1, "value": 9}],
    [{"candidate_id": 0, "value": "$3"}, {"candidate_id": 1, "value": "€9"}],
    [{"candidate_id": 0, "value": "3 MW"}, {"candidate_id": 1, "value": "9 mW"}],
    [{"candidate_id": 0, "value": 10**400}, {"candidate_id": 1, "value": 9}],
])
def test_extreme_rejects_unidentified_or_incomparable_readings(tmp_path, readings):
    result = _extreme_readings(tmp_path, readings)
    assert not result.ok


def test_extreme_fails_on_tie_or_unreadable_value(tmp_path):
    lib = seed_library()
    model = generation_model()
    model.add("Output only a JSON array", json.dumps([
        {"name": "blue square", "bbox": BLUE_BOX, "value": "80 pixels"},
        {"name": "blue square", "bbox": BLUE_BOX_2, "value": "80.4 pixels"},
    ]))
    steps = {"t": OperatorResult(True, [_obj(BLUE_BOX), _obj(BLUE_BOX_2)])}
    r = apply_operator(_op(lib, "Extreme"),
                       _ctx(tmp_path, steps, {"id": "x", "params": {"field": "size"}, "inputs": ["t"]}, model))
    assert not r.ok and "there is a tie, no unique extremum" in r.failure_reason
    model = generation_model()
    model.add("Output only a JSON array", json.dumps([
        {"name": "blue square", "bbox": BLUE_BOX, "value": "80 pixels"},
        {"name": "blue square", "bbox": BLUE_BOX_2, "value": "large-ish"},
    ]))
    r = apply_operator(_op(lib, "Extreme"),
                       _ctx(tmp_path, steps, {"id": "x", "params": {"field": "size"}, "inputs": ["t"]}, model))
    assert not r.ok and "cannot read field size for all candidates" in r.failure_reason


def test_calculate_runs_compute_tool_without_the_model(tmp_path):
    calls = []
    def model(request):
        calls.append(request)
        return "{}", {}
    steps = {"a": OperatorResult(True, "5 cm", [{"step": "a"}]), "b": OperatorResult(True, "3 cm", [{"step": "b"}])}
    r = apply_operator(_op(seed_library(), "Calculate"),
                       _ctx(tmp_path, steps, {"id": "c", "params": {"op": "-"}, "inputs": ["a", "b"]}, model))
    assert r.ok and r.value == "2.0 cm" and calls == []  # 算术不经过模型
    assert r.evidence[-1] == {"operation": "-", "result": 2.0}


def test_calculate_rejects_mixed_units_and_bad_operators(tmp_path):
    lib = seed_library()
    steps = {"a": OperatorResult(True, "5 cm"), "b": OperatorResult(True, "3 in")}
    r = apply_operator(_op(lib, "Calculate"),
                       _ctx(tmp_path, steps, {"id": "c", "params": {"op": "+"}, "inputs": ["a", "b"]}))
    assert not r.ok and "units differ" in r.failure_reason
    steps = {"a": OperatorResult(True, 5), "b": OperatorResult(True, 3)}
    r = apply_operator(_op(lib, "Calculate"),
                       _ctx(tmp_path, steps, {"id": "c", "params": {"op": "**"}, "inputs": ["a", "b"]}))
    assert not r.ok and "unsupported arithmetic operator" in r.failure_reason


def test_match_returns_the_matched_candidate(tmp_path):
    model = generation_model()
    steps = {
        "e": OperatorResult(True, _obj(BLUE_BOX)),
        "c": OperatorResult(True, [_obj(BLUE_BOX), _obj(BLUE_BOX_2)]),
    }
    r = apply_operator(_op(seed_library(), "Match"),
                       _ctx(tmp_path, steps, {"id": "m", "params": {}, "inputs": ["e", "c"]}, model))
    assert r.ok and r.value["bbox"] == BLUE_BOX
    assert r.evidence[0]["role"] == "target" and r.evidence[1]["role"] == "match"
    assert r.evidence[2]["cue"] == "same printed label"


def test_match_fails_without_distinguishing_cue(tmp_path):
    model = generation_model()
    model.add("match_index", '{"match_index": 0, "cue": "", "distinct": false}')
    steps = {"e": OperatorResult(True, _obj(BLUE_BOX)), "c": OperatorResult(True, [_obj(BLUE_BOX)])}
    r = apply_operator(_op(seed_library(), "Match"),
                       _ctx(tmp_path, steps, {"id": "m", "params": {}, "inputs": ["e", "c"]}, model))
    assert not r.ok and "no distinguishing cue to match" in r.failure_reason


def test_match_rejects_out_of_range_index(tmp_path):
    model = generation_model()
    model.add("match_index", '{"match_index": 5, "cue": "x", "distinct": true}')
    steps = {"e": OperatorResult(True, _obj(BLUE_BOX)), "c": OperatorResult(True, [_obj(BLUE_BOX)])}
    r = apply_operator(_op(seed_library(), "Match"),
                       _ctx(tmp_path, steps, {"id": "m", "params": {}, "inputs": ["e", "c"]}, model))
    assert not r.ok and "valid match index" in r.failure_reason
