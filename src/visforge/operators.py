"""算子：dict + 函数，不做类继承和注册器。

apply_operator 同时做检查和执行（合在一层）；用不了时 failure_reason 写清缺哪条证据。
"""

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from .models import response_excerpt
from .structured import READ_OUTPUT, OutputContract, OutputError, readable_text
from .tools import run_tool


_SELECT_OUTPUT = OutputContract({"objects": [{"name": "object", "bbox": [0, 0, 1, 1]}]}, list_key="objects")
_EXISTS_OUTPUT = OutputContract({"found": False, "full_scope_checked": True})
_EXTREME_OUTPUT = OutputContract([{"candidate_id": 0, "value": "2 cm"}], list_key="readings")
_MATCH_OUTPUT = OutputContract({"match_index": 0, "cue": "shared visible marking", "distinct": True},
                               fields={"distinct": bool})
_PARTNER_OUTPUT = OutputContract({"partner": {"name": "support", "bbox": [0, 0, 1, 1]},
                                  "contact": [0, 0, 1, 1]}, fields={})


@dataclass
class OperatorResult:
    """一步算子的结果：ok、值、留下的证据、失败原因。"""

    ok: bool
    value: Any = None
    evidence: list = field(default_factory=list)
    failure_reason: str = ""
    diagnostic: dict | None = None


def _prior(ctx):
    """取当前步骤第一个输入步骤的结果。"""
    inputs = ctx["step"].get("inputs") or []
    if not inputs:
        return None
    return (ctx.get("steps") or {}).get(inputs[0])


def _priors(ctx):
    """按 inputs 顺序取全部输入步骤的结果。"""
    steps = ctx.get("steps") or {}
    return [steps.get(name) for name in (ctx["step"].get("inputs") or [])]


def _image_size(image):
    if isinstance(image, Image.Image):
        return image.size
    with Image.open(image) as source:
        return source.size


def _valid_box(box, size, *, pixels=True):
    if not isinstance(box, (tuple, list)) or len(box) != 4 or not all(type(n) in (int, float) for n in box):
        return False
    x1, y1, x2, y2 = box
    width, height = size
    return (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height
            and (not pixels or (int(x1) < int(x2) and int(y1) < int(y2))))


def _output_diagnostic(subtype, size, text, meta):
    """Retain only sanitized visible output. The digest covers the full sanitized text."""
    diagnostic = {"subtype": subtype, "source_image_size": list(size), **response_excerpt(text)}
    generation = (meta or {}).get("generation")
    if isinstance(generation, dict):
        metrics = {key: generation[key] for key in ("input_tokens", "generated_tokens", "max_new_tokens", "last_token_id")
                   if type(generation.get(key)) is int}
        if generation.get("finish_reason") in ("eos", "length", "other"):
            metrics["finish_reason"] = generation["finish_reason"]
        grid = generation.get("image_grid_thw")
        if isinstance(grid, list) and len(grid) == 1 and isinstance(grid[0], list) and len(grid[0]) == 3:
            if all(type(n) is int and n > 0 for n in grid[0]):
                metrics["image_grid_thw"] = grid
        diagnostic["generation"] = metrics
    return diagnostic


def _model_result(op, ctx, prompt, contract, evaluate, *, view=None):
    """Decode the response, then check the operator's visual evidence."""
    text, meta = ctx["model"]({
        "text": prompt + "\n" + contract.instruction,
        "images": [ctx["image"] if view is None else view],
        "generation_diagnostics": True,
    })
    try:
        return evaluate(contract.parse(text))
    except OutputError as error:
        diagnostic = _output_diagnostic(error.subtype, _image_size(ctx["image"]), text, meta)
        return OperatorResult(False, failure_reason=f"{op['name']}: {error}", diagnostic=diagnostic)


def _select(op, ctx):
    step, params = ctx["step"], ctx["step"].get("params", {})
    desc = params.get("description")
    if not desc:
        return OperatorResult(False, failure_reason="Select is missing the object description")
    size = _image_size(ctx["image"])
    prompt = (
        op.get("prompt", "Look at the image and list all objects matching the description.")
        + f"\nDescription: {desc}\n"
        + "Give each name and bbox [x1, y1, x2, y2] in original-image pixel coordinates "
        f"(width {size[0]}, height {size[1]})."
    )
    return _model_result(op, ctx, prompt, _SELECT_OUTPUT,
                         lambda items: _selected_objects(items, params, step["id"], size))


def _selected_objects(items, params, step_id, size):
    if not items:
        if params.get("allow_empty"):
            return OperatorResult(True, [], [])
        raise OutputError("empty_grounding", f"no object matching '{params['description']}' found")
    objects, evidence = [], []
    first_box_key = None
    for item in items:
        if not isinstance(item, dict) or ("bbox" in item) == ("bbox_2d" in item):
            raise OutputError("wrong_schema", "each object needs exactly one bbox coordinate field")
        normalized = "bbox_2d" in item
        label_key, box_key = ("label", "bbox_2d") if normalized else ("name", "bbox")
        if first_box_key is not None and box_key != first_box_key:
            raise OutputError("wrong_schema", "objects use mixed coordinate conventions")
        first_box_key = box_key
        if not isinstance(item.get(label_key), str) or not item[label_key].strip():
            raise OutputError("wrong_schema", f"each {box_key} needs a nonempty {label_key}")
        raw_box = item[box_key]
        if not _valid_box(raw_box, (1000, 1000) if normalized else size, pixels=not normalized):
            raise OutputError("invalid_box", "object has an invalid box in its declared coordinates")
        box = raw_box
        if normalized:
            x1, y1, x2, y2 = raw_box
            width, height = size
            box = [math.floor(x1 * width / 1000), math.floor(y1 * height / 1000),
                   math.ceil(x2 * width / 1000), math.ceil(y2 * height / 1000)]
        name = item[label_key]
        objects.append({"name": name, "bbox": box} if normalized else item)
        evidence.append({"step": step_id, "object": name, "bbox": box, "source_bbox": raw_box,
                         "source_image_size": list(size),
                         "source_bbox_format": "normalized_xyxy_0_1000" if normalized else "original_pixel_xyxy"})
    if params.get("require_unique") and len(objects) > 1:
        raise OutputError("non_unique", f"{len(objects)} objects match '{params['description']}', "
                          "not unique enough to ask 'which one'")
    value = objects[0] if params.get("require_unique") else objects
    return OperatorResult(True, value, evidence)


def parse_value(text_or_num):
    """数字或含数字的文本 -> (数值, 单位或 None)；读不出返回 None。"""
    if isinstance(text_or_num, bool):
        return None
    if isinstance(text_or_num, (int, float)):
        return (float(text_or_num), None)
    if not isinstance(text_or_num, str):
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text_or_num)
    if match is None:
        return None
    value = float(match.group())
    unit = text_or_num[match.end():].strip(" \t.:;!,")
    return (value, unit.casefold() or None)


def _count(op, ctx):
    prior = _prior(ctx)
    if prior is None:
        return OperatorResult(False, failure_reason="Count needs a range selected by Select first")
    items = prior.value
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return OperatorResult(False, failure_reason="Count's previous step did not produce an object list")
    if not items:
        return OperatorResult(False, failure_reason="no objects found in the range, cannot form a counting question")
    # 第一版范围就是全图：给模型看的是整张原图
    return OperatorResult(True, len(items), [{"range": "full image", "counted": len(items)}])


def _identify(op, ctx):
    prior = _prior(ctx)
    if prior is None or not isinstance(prior.value, list):
        return OperatorResult(False, failure_reason="Identify needs the object list from a Select step")
    params = ctx["step"].get("params", {})
    if not prior.value:
        return OperatorResult(False, failure_reason="no object matching the description found")
    if len(prior.value) > 1:
        return OperatorResult(
            False,
            failure_reason=f"{len(prior.value)} matching objects: not unique, cannot ask which one",
        )
    obj = prior.value[0]
    evidence = [{"step": ctx["step"]["id"], "object": obj.get("name"), "bbox": obj.get("bbox"),
                 "distinguisher": params.get("distinguisher", "")}]
    return OperatorResult(True, obj, evidence)


def _locate(op, ctx):
    """纯几何定位：两个 bbox 的中心沿轴比较，零模型调用。"""
    results = _priors(ctx)
    params = ctx["step"].get("params", {})
    if len(results) < 2 or any(r is None for r in results[:2]):
        return OperatorResult(False, failure_reason="Locate needs two completed inputs: entity and anchor")
    entity, anchor = results[0], results[1]
    for role, result in (("entity", entity), ("anchor", anchor)):
        if not isinstance(result.value, dict) or not result.value.get("bbox"):
            return OperatorResult(False, failure_reason=f"Locate needs a single located object as the {role}")
    axis = params.get("axis", "horizontal")
    if axis not in ("horizontal", "vertical"):
        return OperatorResult(False, failure_reason="Locate axis must be 'horizontal' or 'vertical'")
    eb, ab = entity.value["bbox"], anchor.value["bbox"]
    if axis == "horizontal":
        offset = (eb[0] + eb[2]) / 2 - (ab[0] + ab[2]) / 2
        tol = 0.25 * max(eb[2] - eb[0], ab[2] - ab[0])
        direction = "right" if offset > 0 else "left"
    else:
        offset = (eb[1] + eb[3]) / 2 - (ab[1] + ab[3]) / 2
        tol = 0.25 * max(eb[3] - eb[1], ab[3] - ab[1])
        direction = "below" if offset > 0 else "above"
    if abs(offset) < tol:
        return OperatorResult(False, failure_reason="positions not distinguishable along the axis")
    evidence = [
        {"role": "entity", "bbox": list(eb)},
        {"role": "anchor", "bbox": list(ab)},
        {"axis": axis, "offset": offset},
    ]
    return OperatorResult(True, direction, evidence)


def _exists(op, ctx):
    prior = _prior(ctx)
    if prior is None or not isinstance(prior.value, list):
        return OperatorResult(False, failure_reason="Exists needs the object list from a Select step")
    params = ctx["step"].get("params", {})
    if prior.value:
        # 选到即存在，不调模型
        return OperatorResult(True, True, [{"found_in_selection": len(prior.value)}])
    desc = params.get("description", "the described objects")
    prompt = (
        op.get("prompt", "Check whether any object matching the description exists in the image.")
        + f"\nDescription: {desc}\n"
        + "Look over the ENTIRE image carefully before answering. Report both fields as booleans."
    )

    def evaluate(data):
        if not data["full_scope_checked"]:
            raise OutputError("insufficient_evidence", "cannot confirm absence without a full-scope search")
        if data["found"]:
            return OperatorResult(True, True, [{"found_by_full_scope_search": True}])
        return OperatorResult(True, False, [{"full_scope_search": "no match found in the entire image"}])

    return _model_result(op, ctx, prompt, _EXISTS_OUTPUT, evaluate)


def _operands(ctx, name):
    """取两个操作数步骤的结果；不完整时返回 (None, 错误信息)。"""
    results = _priors(ctx)
    if len(results) < 2 or any(r is None for r in results[:2]):
        return None, f"{name} needs two completed operand steps"
    return results[:2], None


def _compare(op, ctx):
    results, error = _operands(ctx, "Compare")
    if error:
        return OperatorResult(False, failure_reason=error)
    params = ctx["step"].get("params", {})
    a, b = parse_value(results[0].value), parse_value(results[1].value)
    if a is None or b is None:
        return OperatorResult(False, failure_reason="Compare: cannot read a numeric value from both operands")
    (va, ua), (vb, ub) = a, b
    if ua != ub:
        return OperatorResult(False, failure_reason="units differ, not on a common scale")
    labels = params.get("labels") or ["the first", "the second"]
    if math.isclose(va, vb, rel_tol=0.01):
        return OperatorResult(False, failure_reason="values are equal, no distinguishable order")
    winner = labels[0] if va > vb else labels[1]
    evidence = [
        {"operand": labels[0], "value": results[0].value, "parsed": va, "unit": ua,
         "provenance": results[0].evidence},
        {"operand": labels[1], "value": results[1].value, "parsed": vb, "unit": ub,
         "provenance": results[1].evidence},
    ]
    return OperatorResult(True, winner, evidence)


def _extreme_measurement(value):
    if type(value) in (int, float):
        try:
            amount = float(value)
        except OverflowError:
            return None
        return (amount, None) if math.isfinite(amount) else None
    if not isinstance(value, str):
        return None
    # Preserve prefix units such as currency and suffix units such as pixels.
    match = re.fullmatch(
        r"([^\d+.,-]*?)([+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:[eE][+-]?\d+)?)([^\d]*)",
        value.strip(),
    )
    if match is None:
        return None
    prefix, number, suffix = match.groups()
    amount = float(number.replace(",", ""))
    # Unit symbols are case-sensitive (e.g. MW and mW differ by nine orders of magnitude).
    unit = " ".join((prefix.strip() + " " + suffix.strip()).split())
    return (amount, unit or None) if math.isfinite(amount) else None


def _extreme_candidate_id(reading, candidates):
    if not isinstance(reading, dict):
        return None
    identity = {k: reading[k] for k in ("candidate_id", "name", "bbox") if k in reading}
    if not identity or ("candidate_id" in identity and type(identity["candidate_id"]) is not int):
        return None
    if "bbox" in identity:
        box = identity["bbox"]
        if not isinstance(box, list) or any(type(n) not in (int, float) for n in box):
            return None
    matches = [candidate["candidate_id"] for candidate in candidates
               if all(candidate[key] == value for key, value in identity.items())]
    return matches[0] if len(matches) == 1 else None


def _extreme(op, ctx):
    prior = _prior(ctx)
    if prior is None or not isinstance(prior.value, list) or not prior.value:
        return OperatorResult(False, failure_reason="Extreme needs a nonempty selected object list")
    params = ctx["step"].get("params", {})
    field = params.get("field", "size")
    direction = params.get("direction", "max")
    if direction not in ("max", "min"):
        return OperatorResult(False, failure_reason="Extreme direction must be 'max' or 'min'")
    candidates = prior.value
    listing = []
    for index, cand in enumerate(candidates):
        if not isinstance(cand, dict) or not cand.get("bbox"):
            return OperatorResult(False, failure_reason="Extreme: every candidate needs a name and a bbox")
        listing.append({"candidate_id": index, "name": cand.get("name"), "bbox": list(cand["bbox"])})
    prompt = (
        op.get("prompt", f"Read the {field} of each listed object.")
        + f"\nField: {field}\nCandidates: {json.dumps(listing, ensure_ascii=False)}\n"
        + "Use each given candidate_id exactly once and keep all readings on the same scale."
    )
    return _model_result(op, ctx, prompt, _EXTREME_OUTPUT,
                         lambda rows: _rank_readings(rows, candidates, listing, field, direction))


def _rank_readings(data, candidates, listing, field, direction):
    if len(data) != len(candidates):
        raise OutputError("wrong_schema", "expected one reading for each listed object")
    readings = {}
    for entry in data:
        index = _extreme_candidate_id(entry, listing)
        if index is None or index in readings:
            raise OutputError("wrong_schema", "each reading must identify a different listed object")
        if type(entry.get("value")) not in (int, float, str):
            raise OutputError("wrong_schema", "each reading needs a numeric or textual value")
        parsed = _extreme_measurement(entry["value"])
        if parsed is None:
            raise OutputError("insufficient_evidence", f"cannot read field {field} for all candidates")
        readings[index] = parsed
    if len({unit for _, unit in readings.values()}) != 1:
        raise OutputError("insufficient_evidence", "units differ, not on a common scale")
    values = [(cand, readings[index]) for index, cand in enumerate(candidates)]
    best = max(p[0] for _c, p in values) if direction == "max" else min(p[0] for _c, p in values)
    winners = [(c, p) for c, p in values if math.isclose(p[0], best, rel_tol=0.01)]
    if len(winners) > 1:
        raise OutputError("insufficient_evidence", "there is a tie, no unique extremum")
    winner, _ = winners[0]
    evidence = [{"name": c.get("name"), "bbox": list(c["bbox"]), "value": p[0], "unit": p[1]} for c, p in values]
    evidence.append({"winner": winner.get("name"), "bbox": list(winner["bbox"])})
    return OperatorResult(True, winner, evidence)


def _calculate(op, ctx):
    """算术运算：数值全部来自上一步，绝不经过模型。"""
    results, error = _operands(ctx, "Calculate")
    if error:
        return OperatorResult(False, failure_reason=error)
    params = ctx["step"].get("params", {})
    a, b = parse_value(results[0].value), parse_value(results[1].value)
    if a is None or b is None:
        return OperatorResult(False, failure_reason="Calculate: cannot read a numeric value from both operands")
    (va, ua), (vb, ub) = a, b
    if ua != ub:
        return OperatorResult(False, failure_reason="units differ, not on a common scale")
    symbol = params.get("op", "+")
    if symbol not in ("+", "-", "*", "/"):
        return OperatorResult(False, failure_reason=f"unsupported arithmetic operator: {symbol}")
    try:
        computed = run_tool("compute", None, {"expression": f"({va}){symbol}({vb})"})["value"]
    except ZeroDivisionError:
        return OperatorResult(False, failure_reason="cannot compute: division by zero")
    answer = f"{computed} {ua}".strip() if ua else computed
    evidence = [
        {"operand": results[0].value, "parsed": va, "unit": ua, "provenance": results[0].evidence},
        {"operand": results[1].value, "parsed": vb, "unit": ub, "provenance": results[1].evidence},
        {"operation": symbol, "result": computed},
    ]
    return OperatorResult(True, answer, evidence)


def _match(op, ctx):
    """同一张图内的对应物匹配：按共享标签、外观或地标选。"""
    results = _priors(ctx)
    if len(results) < 2 or any(r is None for r in results[:2]):
        return OperatorResult(False, failure_reason="Match needs the target step and the candidate-list step")
    target, pool_res = results[0], results[1]
    if not isinstance(target.value, dict) or not target.value.get("bbox"):
        return OperatorResult(False, failure_reason="Match needs a single located target object")
    pool = pool_res.value
    if not isinstance(pool, list) or not pool:
        return OperatorResult(False, failure_reason="Match needs a nonempty candidate list from Select")
    options = []
    for cand in pool:
        if not isinstance(cand, dict) or not cand.get("bbox"):
            return OperatorResult(False, failure_reason="Match: every candidate needs a name and a bbox")
        options.append({"name": cand.get("name"), "bbox": list(cand["bbox"])})
    target_box = list(target.value["bbox"])
    prompt = (
        op.get("prompt", "Decide which candidate corresponds to the target object, "
        "using shared labels, appearance, or landmarks.")
        + f"\nTarget: {json.dumps({'name': target.value.get('name'), 'bbox': target_box}, ensure_ascii=False)}\n"
        + f"Candidates: {json.dumps(options, ensure_ascii=False)}\n"
        + "Give the 0-based match_index and describe the distinguishing visual cue. "
        "Set distinct=false when no distinguishing cue exists."
    )

    def evaluate(data):
        if not data["distinct"]:
            raise OutputError("insufficient_evidence", "no distinguishing cue to match")
        index = data.get("match_index")
        if type(index) is not int or not 0 <= index < len(pool):
            raise OutputError("wrong_schema", "model did not return a valid match index")
        cue = data.get("cue")
        if not isinstance(cue, str) or not cue.strip():
            raise OutputError("wrong_schema", "a distinct match needs a nonempty visual cue")
        matched = pool[index]
        evidence = [
            {"role": "target", "bbox": target_box},
            {"role": "match", "name": matched.get("name"), "bbox": list(matched["bbox"])},
            {"cue": cue},
        ]
        return OperatorResult(True, matched, evidence)

    return _model_result(op, ctx, prompt, _MATCH_OUTPUT, evaluate)


def _find_partner(op, ctx):
    prior = _prior(ctx)
    if prior is None or not isinstance(prior.value, dict) or not prior.value.get("bbox"):
        return OperatorResult(
            False, failure_reason="FindPartner needs a single object uniquely selected first (a single Select result)"
        )
    target = prior.value
    size = _image_size(ctx["image"])
    prompt = (
        op.get("prompt", "Find the object that supports the target object.")
        + f"\nTarget object: {target.get('name')}, location: {target.get('bbox')}\n"
        + "Identify the partner and their contact or containment region. When it is not visible, "
        'return {"visible": false, "reason": "what evidence is missing"}. '
        f"Use original-image pixel coordinates (width {size[0]}, height {size[1]})."
    )

    def evaluate(data):
        visible = data.get("visible", True)
        if type(visible) is not bool:
            raise OutputError("wrong_schema", "visible must be a boolean")
        if not visible:
            reason = data.get("reason", "")
            if not isinstance(reason, str):
                raise OutputError("wrong_schema", "reason must be text")
            excerpt = response_excerpt(reason)["final_text_excerpt"]
            raise OutputError("insufficient_evidence", excerpt or "cannot see the support or contact relation")
        partner = data.get("partner")
        if (not isinstance(partner, dict) or not isinstance(partner.get("name"), str)
                or not partner["name"].strip() or "bbox" not in partner or "contact" not in data):
            raise OutputError("wrong_schema", "a visible partner needs a name, bbox and contact box")
        if not _valid_box(partner["bbox"], size) or not _valid_box(data["contact"], size):
            raise OutputError("invalid_box", "invalid partner or contact box in original-image coordinates")
        evidence = [
            {"role": "target", "bbox": target["bbox"]},
            {"role": "partner", "name": partner["name"], "bbox": partner["bbox"]},
            {"role": "contact", "bbox": data["contact"]},
        ]
        return OperatorResult(True, partner, evidence)

    return _model_result(op, ctx, prompt, _PARTNER_OUTPUT, evaluate)


def _read(op, ctx):
    prior = _prior(ctx)
    if prior is None or not isinstance(prior.value, dict) or not prior.value.get("bbox"):
        return OperatorResult(
            False, failure_reason="Read needs the bbox of the region to read from the previous step"
        )
    params = ctx["step"].get("params", {})
    bbox = prior.value["bbox"]
    if not _valid_box(bbox, _image_size(ctx["image"])):
        return OperatorResult(False, failure_reason="Read: invalid or empty region in original-image coordinates")
    # 从原图放大裁剪，避免整图缩小后看不清
    zoom = run_tool("zoom", ctx["image"], {"bbox": bbox, "scale": params.get("scale", 3)})
    what = params.get("what", "text or number")
    prompt = (
        op.get("prompt", "Read the text or number in the image.")
        + f"\nWhat to read: {what}\n"
        + "Set legible=false when the text is unclear."
    )

    def evaluate(data):
        text = readable_text(data)
        if text is None:
            raise OutputError("insufficient_evidence", f"cannot read {what} clearly even after zooming")
        return OperatorResult(True, text, [zoom["provenance"], {"region": list(bbox), "text": text}])

    return _model_result(op, ctx, prompt, READ_OUTPUT, evaluate, view=zoom["image"])


# 名字到函数的分派表；检查与执行合在每个函数里
_APPLY = {
    "Select": _select,
    "Identify": _identify,
    "Count": _count,
    "Exists": _exists,
    "Locate": _locate,
    "FindPartner": _find_partner,
    "Read": _read,
    "Compare": _compare,
    "Extreme": _extreme,
    "Calculate": _calculate,
    "Match": _match,
}


def apply_operator(op, context):
    """检查并执行一个算子步骤。

    op 是规则 dict（name/description/prompt）；context 含 image、model、
    steps（已完成步骤的结果）和 step（当前步骤的 id/params/inputs）。
    """
    fn = _APPLY.get(op.get("name"))
    if fn is None:
        return OperatorResult(False, failure_reason=f"unknown operator: {op.get('name')}")
    return fn(op, context)


# 初始计划模板：出题模型只负责填槽位短语，问句由模板给出
TEMPLATES = [
    {
        "name": "support-read-label",
        "propose": "Pick a concrete object in the image whose support (or contact) is clearly visible, as the target.",
        "slots": ["target"],
        "steps": [
            {
                "id": "target",
                "op": "Select",
                "params": {"description": "${target}", "require_unique": True},
                "inputs": [],
            },
            {"id": "partner", "op": "FindPartner", "params": {}, "inputs": ["target"]},
            {
                "id": "label",
                "op": "Read",
                "params": {"what": "text or number on the supporting object"},
                "inputs": ["partner"],
            },
        ],
        "question": "What is written on the object that supports {target}?",
        "answer_from": "label",
    },
    {
        "name": "range-count",
        "propose": "Pick a group of same-kind objects in the image that can be counted clearly, "
        "and give the category description.",
        "slots": ["targets"],
        "steps": [
            {
                "id": "targets",
                "op": "Select",
                "params": {"description": "${targets}", "require_unique": False},
                "inputs": [],
            },
            {"id": "n", "op": "Count", "params": {}, "inputs": ["targets"]},
        ],
        "question": "How many {targets} are in the image?",
        "answer_from": "n",
    },
    {
        "name": "extreme",
        "propose": "Pick a group of same-kind objects in the image that differ visibly in size, "
        "and give the category description.",
        "slots": ["targets"],
        "steps": [
            {
                "id": "targets",
                "op": "Select",
                "params": {"description": "${targets}", "require_unique": False},
                "inputs": [],
            },
            {
                "id": "extreme",
                "op": "Extreme",
                "params": {"field": "size", "direction": "max"},
                "inputs": ["targets"],
            },
        ],
        "question": "Which {targets} is the largest?",
        "answer_from": "extreme",
    },
    {
        "name": "count-compare",
        "propose": "Pick two kinds of countable objects in the image, giving a distinct category description "
        "for each.",
        "slots": ["a_desc", "b_desc"],
        "steps": [
            {"id": "a_sel", "op": "Select", "params": {"description": "${a_desc}", "require_unique": False}, "inputs": []},
            {"id": "b_sel", "op": "Select", "params": {"description": "${b_desc}", "require_unique": False}, "inputs": []},
            {"id": "a_count", "op": "Count", "params": {}, "inputs": ["a_sel"]},
            {"id": "b_count", "op": "Count", "params": {}, "inputs": ["b_sel"]},
            {
                "id": "more",
                "op": "Compare",
                "params": {"labels": ["${a_desc}", "${b_desc}"]},
                "inputs": ["a_count", "b_count"],
            },
        ],
        "question": "Which is more numerous, {a_desc} or {b_desc}?",
        "answer_from": "more",
    },
    {
        "name": "locate",
        "propose": "Pick one clearly visible object as the entity and another distinct object as the reference; "
        "both must be uniquely identifiable.",
        "slots": ["entity_desc", "anchor_desc"],
        "steps": [
            {
                "id": "entity",
                "op": "Select",
                "params": {"description": "${entity_desc}", "require_unique": True},
                "inputs": [],
            },
            {
                "id": "anchor",
                "op": "Select",
                "params": {"description": "${anchor_desc}", "require_unique": True},
                "inputs": [],
            },
            {"id": "side", "op": "Locate", "params": {"axis": "horizontal"}, "inputs": ["entity", "anchor"]},
        ],
        "question": "On which side of {anchor_desc} is the {entity_desc}?",
        "answer_from": "side",
    },
    {
        "name": "same-page-match",
        "propose": "Pick one uniquely identifiable object as the target and a group of same-kind objects "
        "as the candidates it should be matched against.",
        "slots": ["entity_desc", "candidates_desc"],
        "steps": [
            {
                "id": "entity",
                "op": "Select",
                "params": {"description": "${entity_desc}", "require_unique": True},
                "inputs": [],
            },
            {
                "id": "candidates",
                "op": "Select",
                "params": {"description": "${candidates_desc}", "require_unique": False},
                "inputs": [],
            },
            {"id": "match", "op": "Match", "params": {}, "inputs": ["entity", "candidates"]},
        ],
        "question": "Which {candidates_desc} corresponds to the {entity_desc} (same label or appearance)?",
        "answer_from": "match",
    },
]


def seed_library():
    """初始规则库：11 个算子的描述、提示词和 6 条计划模板，可直接 JSON 序列化。"""
    return {
        "name": "seed",
        "operators": [
            {
                "name": "Select",
                "description": "Look at objects in the specified range; when asking 'which one', "
                "exactly one must remain.",
                "prompt": "Look at the image, list all objects matching the description, "
                "and give each object's name and bbox.",
            },
            {
                "name": "Identify",
                "description": "Tell which object matches a distinguishing description; the selection "
                "must reduce to exactly one match.",
                "prompt": "Check the selected objects and keep only the one matching the distinguishing "
                "description.",
            },
            {
                "name": "Count",
                "description": "See the full range specified by the question before reporting a total.",
                "prompt": "Check the range, then give the count.",
            },
            {
                "name": "Exists",
                "description": "Answer 'yes' from a single sighting; answer 'no' or report a total only "
                "after checking the full specified range.",
                "prompt": "Check the entire image for the described object before answering.",
            },
            {
                "name": "Locate",
                "description": "Give the direction of one object relative to another; both bounding boxes "
                "must come from earlier steps and be comparable along one axis.",
                "prompt": "Compare the two bounding boxes geometrically along the requested axis.",
            },
            {
                "name": "FindPartner",
                "description": "Find the supporting or paired object; contact or containment must be visible.",
                "prompt": "Find the object supporting the target object and give their contact region.",
            },
            {
                "name": "Read",
                "description": "Read a value or text; it must be legible and confirmed to belong to "
                "the object the question refers to.",
                "prompt": "Here is the zoomed image; read the text or number in it.",
            },
            {
                "name": "Compare",
                "description": "Compare two read values only on a common scale with readable units; "
                "equal values cannot answer 'which is more'.",
                "prompt": "Compare the two values from the earlier steps and name the larger one.",
            },
            {
                "name": "Extreme",
                "description": "Pick the unique maximum or minimum over a fully enumerated candidate list; "
                "ties fail and every candidate must be read.",
                "prompt": "Read the requested field for every listed candidate, then pick the extreme.",
            },
            {
                "name": "Calculate",
                "description": "Arithmetic over already-read values with one shared unit; the computation "
                "never goes through the model.",
                "prompt": "Combine the two earlier values with the requested arithmetic operation.",
            },
            {
                "name": "Match",
                "description": "Within one image, match a target object to a candidate by a shared label, "
                "appearance, or landmark; without a distinguishing cue the match fails.",
                "prompt": "Decide which candidate corresponds to the target object and name the cue.",
            },
        ],
        "templates": copy.deepcopy(TEMPLATES),
    }
