"""算子链：Select→Count 与 Select→FindPartner→Read 在合成图上可走通。"""

import json

import pytest
from PIL import Image

from fakes import BLUE_BOX, LABEL_A, TRAY_A, generation_model, make_scene
from visforge.operators import OperatorResult, apply_operator, seed_library
from visforge.synthesis import generate


def _op(library, name):
    return next(o for o in library["operators"] if o["name"] == name)


@pytest.mark.parametrize("reply,expected_box,source_format", [
    ([{"label": "label }", "bbox_2d": [100, 200, 300, 600]}], [200, 160, 600, 480], "normalized_xyxy_0_1000"),
    ({"objects": [{"label": "label }", "bbox_2d": [100, 200, 300, 600]}]}, [200, 160, 600, 480], "normalized_xyxy_0_1000"),
    ([{"name": "label }", "bbox": [100, 200, 300, 600], "label": "printed sign"}], [100, 200, 300, 600], "original_pixel_xyxy"),
    ({"objects": [{"name": "label }", "bbox": [100, 200, 300, 600]}]}, [100, 200, 300, 600], "original_pixel_xyxy"),
])
def test_select_preserves_source_coordinates_and_outputs_pixel_evidence(reply, expected_box, source_format):
    result = apply_operator(seed_library()["operators"][0], {
        "image": Image.new("RGB", (2000, 800)), "model": lambda req: (json.dumps(reply), {}),
        "step": {"id": "t", "params": {"description": "label", "require_unique": True}},
    })
    assert result.ok and result.value["bbox"] == expected_box
    assert result.evidence == [{
        "step": "t", "object": "label }", "bbox": expected_box, "source_bbox": [100, 200, 300, 600],
        "source_image_size": [2000, 800], "source_bbox_format": source_format,
    }]


@pytest.mark.parametrize("reply,subtype", [
    ('[{"label":"t","bbox_2d":[0,0,1000,1000]}', "invalid_json"),
    ('{"something_else":[]}', "wrong_schema"),
    ('[{"label":"t","bbox_2d":[0,0,1001,500]}]', "invalid_box"),
    ('[{"label":"t","bbox_2d":[0,0,900,500],"bbox":[0,0,2,2]}]', "wrong_schema"),
    ('[{"label":"t","bbox":[0,0,2,2]}]', "wrong_schema"),
    ('[{"name":"a","bbox":[0,0,2,2]},{"label":"b","bbox_2d":[0,0,2,2]}]', "wrong_schema"),
    ('[]', "empty_grounding"),
])
def test_select_rejects_bad_replies_with_specific_diagnostics(reply, subtype):
    result = apply_operator(seed_library()["operators"][0], {
        "image": Image.new("RGB", (64, 48)), "model": lambda req: (reply, {}),
        "step": {"id": "t", "params": {"description": "label"}},
    })
    assert not result.ok and result.diagnostic["subtype"] == subtype


def test_seed_library_has_eleven_operators_and_six_templates():
    lib = seed_library()
    assert [o["name"] for o in lib["operators"]] == [
        "Select", "Identify", "Count", "Exists", "Locate", "FindPartner",
        "Read", "Compare", "Extreme", "Calculate", "Match",
    ]
    assert len(lib["templates"]) == 6


def test_select_then_count_chain(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    lib = seed_library()
    model = generation_model()
    steps = {}
    r1 = apply_operator(
        _op(lib, "Select"),
        {"image": str(scene), "model": model, "steps": steps,
         "step": {"id": "t", "params": {"description": "blue square"}, "inputs": []}},
    )
    assert r1.ok and len(r1.value) == 2 and r1.evidence
    steps["t"] = r1
    r2 = apply_operator(
        _op(lib, "Count"),
        {"image": str(scene), "model": model, "steps": steps,
         "step": {"id": "n", "params": {}, "inputs": ["t"]}},
    )
    assert r2.ok and r2.value == 2


def test_select_requires_unique_when_asked(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = generation_model()
    r = apply_operator(
        seed_library()["operators"][0],
        {"image": str(scene), "model": model, "steps": {},
         "step": {"id": "t", "params": {"description": "blue square", "require_unique": True}, "inputs": []}},
    )
    assert not r.ok and "not unique" in r.failure_reason


def test_select_no_match_reports_missing_evidence(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = generation_model()
    model.add("list all objects matching", '{"objects": []}')
    r = apply_operator(
        seed_library()["operators"][0],
        {"image": str(scene), "model": model, "steps": {},
         "step": {"id": "t", "params": {"description": "dinosaur"}, "inputs": []}},
    )
    assert not r.ok and "no object matching" in r.failure_reason


def test_select_find_partner_read_chain(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    lib = seed_library()
    model = generation_model()
    steps = {}
    r1 = apply_operator(
        _op(lib, "Select"),
        {"image": str(scene), "model": model, "steps": steps,
         "step": {"id": "t", "params": {"description": "the left blue square", "require_unique": True}, "inputs": []}},
    )
    assert r1.ok and r1.value["bbox"] == BLUE_BOX
    steps["t"] = r1
    r2 = apply_operator(
        _op(lib, "FindPartner"),
        {"image": str(scene), "model": model, "steps": steps,
         "step": {"id": "p", "params": {}, "inputs": ["t"]}},
    )
    assert r2.ok and r2.value["bbox"] == TRAY_A
    assert [e["role"] for e in r2.evidence] == ["target", "partner", "contact"]
    steps["p"] = r2
    r3 = apply_operator(
        _op(lib, "Read"),
        {"image": str(scene), "model": model, "steps": steps,
         "step": {"id": "l", "params": {"what": "text on the supporting object"}, "inputs": ["p"]}},
    )
    assert r3.ok and r3.value == LABEL_A
    assert r3.evidence[0]["tool"] == "zoom"  # 留下放大区域的 provenance


def test_find_partner_without_contact_reports_what_is_missing(tmp_path):
    scene = make_scene(tmp_path / "scene.png")
    model = generation_model(invisible_on="")  # 一律看不到接触处
    steps = {"t": OperatorResult(True, {"name": "blue square", "bbox": BLUE_BOX})}
    r = apply_operator(
        _op(seed_library(), "FindPartner"),
        {"image": str(scene), "model": model, "steps": steps,
         "step": {"id": "p", "params": {}, "inputs": ["t"]}},
    )
    assert not r.ok and "bottom contact of the cup is not visible" in r.failure_reason


def test_unknown_operator_fails_clearly():
    r = apply_operator({"name": "Nope"}, {"step": {}})
    assert not r.ok and "unknown operator" in r.failure_reason


def test_library_json_roundtrip(tmp_path):
    lib = seed_library()
    import json

    text = json.dumps(lib, ensure_ascii=False)
    assert json.loads(text)["templates"][0]["steps"][1]["op"] == "FindPartner"
    # 出题链路在合成图上整体可走通（顺带确认 6 条模板的库可序列化后仍可用）
    scene = make_scene(tmp_path / "scene.png")
    report = generate(str(scene), json.loads(text), {"model": generation_model()})
    assert len(report["candidates"]) == 6


def _first_two_templates():
    lib = seed_library()
    return {"operators": lib["operators"], "templates": lib["templates"][:2]}


@pytest.mark.parametrize("objects", [
    [None], ["a square"], [{"name": "bad", "bbox": [0, 0, 1000, 500]}],
    [{"name": "bad", "bbox": [5, 2, 5, 8]}], [{"name": "bad", "bbox": [0, 1, float("inf"), 4]}],
])
def test_invalid_model_detections_fail_the_current_template(tmp_path, objects):
    model = generation_model()
    model.add("list all objects matching", json.dumps({"objects": objects}))
    report = generate(make_scene(tmp_path / "scene.png"), _first_two_templates(), {"model": model})
    assert report["candidates"] == []
    assert [failure["step"] for failure in report["failures"]] == ["target", "targets"]


@pytest.mark.parametrize("reply", [
    {"legible": "false", "text": "TEA"}, {"legible": True, "text": None},
])
def test_read_requires_positive_boolean_and_visible_text(tmp_path, reply):
    model = generation_model()
    model.add("What to read:", json.dumps(reply))
    report = generate(make_scene(tmp_path / "scene.png"), seed_library(), {"model": model, "budget": 1})
    assert report["candidates"] == []
    assert [failure["step"] for failure in report["failures"]] == ["label"]


def test_read_fails_locally_for_empty_crop(tmp_path):
    previous = OperatorResult(True, {"name": "tray", "bbox": [800, 900, 820, 910]})
    result = apply_operator(_op(seed_library(), "Read"), {
        "image": make_scene(tmp_path / "scene.png"), "model": generation_model(), "steps": {"p": previous},
        "step": {"id": "label", "params": {}, "inputs": ["p"]},
    })
    assert not result.ok and result.failure_reason


def test_partner_requires_visible_contact_in_original_image(tmp_path):
    model = generation_model()
    model.add("Target object:", json.dumps({
        "partner": {"name": "tray", "bbox": TRAY_A}, "contact": [0, 0, 0, 0],
    }))
    report = generate(make_scene(tmp_path / "scene.png"), seed_library(), {"model": model, "budget": 1})
    assert [failure["step"] for failure in report["failures"]] == ["partner"]


_BOXED = {"name": "tray", "bbox": TRAY_A}


@pytest.mark.parametrize("operator,inputs,payload,subtype", [
    ("Exists", [[]], {"found": 0, "full_scope_checked": True}, "wrong_schema"),
    ("Exists", [[]], {"found": False, "full_scope_checked": False}, "insufficient_evidence"),
    ("Match", [_BOXED, [_BOXED]], {"distinct": True, "match_index": 0, "cue": " "}, "wrong_schema"),
    ("Match", [_BOXED, [_BOXED]], {"distinct": False}, "insufficient_evidence"),
    ("Read", [_BOXED], {"legible": True, "text": None}, "wrong_schema"),
    ("Read", [_BOXED], {"legible": False}, "insufficient_evidence"),
    ("FindPartner", [_BOXED], {"visible": 0}, "wrong_schema"),
    ("FindPartner", [_BOXED], {"visible": False}, "insufficient_evidence"),
    ("Extreme", [[_BOXED]], [{"candidate_id": 0}], "wrong_schema"),
    ("Extreme", [[{"name": "tag", "bbox": [0, 0, 1, 1]}]],
     [{"bbox": [False, False, True, True], "value": 2}], "wrong_schema"),
    ("Extreme", [[_BOXED]], [{"candidate_id": 0, "value": "unreadable"}], "insufficient_evidence"),
])
def test_operator_failures_distinguish_format_from_evidence(operator, inputs, payload, subtype):
    steps = {str(i): OperatorResult(True, value) for i, value in enumerate(inputs)}
    result = apply_operator(_op(seed_library(), operator), {
        "image": Image.new("RGB", (640, 480)),
        "model": lambda req: ("<think>private draft</think>" + json.dumps(payload), {"secret": "private"}),
        "steps": steps, "step": {"id": "step", "inputs": list(steps)},
    })
    assert not result.ok
    assert result.diagnostic["subtype"] == subtype
    assert result.diagnostic["final_text_excerpt"] == json.dumps(payload)
    assert result.diagnostic["source_image_size"] == [640, 480]
    assert "private" not in json.dumps(result.diagnostic)


def test_read_uses_only_the_final_visible_ocr():
    result = apply_operator(_op(seed_library(), "Read"), {
        "image": Image.new("RGB", (640, 480)),
        "model": lambda req: ('<think>{"legible":true,"text":"DRAFT"}</think>\n'
                              '{"legible":true,"text":"TEA"}', {}),
        "steps": {"p": OperatorResult(True, _BOXED)}, "step": {"id": "label", "inputs": ["p"]},
    })
    assert result.ok and result.value == "TEA"
