"""Interfaces, validation, and expansion for executable synthesis components."""

import re
from string import Formatter


DEFAULT_PLANNER_PROMPT = "Rank concrete bindings by the visible evidence that their full operator chains can succeed."
MAX_TEMPLATE_STEPS = 24
MAX_COMPOSITION_DEPTH = 6


def _fill_slots(params, proposal):
    """把参数里的 "${槽位}" 换成出题模型的提议（含嵌套 list/dict）。"""
    def fill(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return proposal.get(value[2:-1], value)
        if isinstance(value, list):
            return [fill(item) for item in value]
        if isinstance(value, dict):
            return {key: fill(item) for key, item in value.items()}
        return value

    return {key: fill(value) for key, value in params.items()}


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _symbol(value):
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", value) is not None


def _template_body(template):
    """Compare execution order and dependencies independent of bookkeeping names."""
    steps = template["steps"]
    positions = {step["id"]: i for i, step in enumerate(steps)}
    plan = [(step["op"], step.get("params", {}), [positions[s] for s in step.get("inputs", [])])
            for step in steps]
    return template["propose"], template["slots"], plan, template["question"], positions[template["answer_from"]]


# Runtime input and result types for the implemented operators. Read values may
# be numeric text; the numeric operators additionally check units at execution.
_STEP_TYPES = {
    "Select": ((), "list"),
    "Identify": (({"list"},), "object"),
    "Count": (({"object", "list"},), "numeric"),
    "Exists": (({"list"},), "boolean"),
    "Locate": (({"object"}, {"object"}), "text"),
    "FindPartner": (({"object"},), "object"),
    "Read": (({"object"},), "read_value"),
    "Compare": (({"numeric", "read_value"}, {"numeric", "read_value"}), "text"),
    "Extreme": (({"list"},), "object"),
    "Calculate": (({"numeric", "read_value"}, {"numeric", "read_value"}), "numeric"),
    "Match": (({"object"}, {"list"}), "object"),
}
_MODEL_OPERATORS = {"Select", "Exists", "FindPartner", "Read", "Extreme", "Match"}
_TEXT_PARAMS = {"Select": {"description"}, "Identify": {"distinguisher"}, "Read": {"what"},
                "Extreme": {"field"}, "Exists": {"description"}}
_ENUM_PARAMS = {"Locate": ("axis", ("horizontal", "vertical")), "Extreme": ("direction", ("max", "min")),
                "Calculate": ("op", ("+", "-", "*", "/"))}
_EXTRA_PARAMS = {"Select": {"require_unique", "allow_empty"}, "Read": {"scale"}, "Compare": {"labels"}}
_TYPES = {result for _, result in _STEP_TYPES.values()}


def _parameter_slots(op, params, slots, composition=None):
    if composition is not None and (not isinstance(params, dict) or set(params) != set(composition["slots"])):
        raise ValueError(f"{op} parameters must bind exactly its declared slots")
    text_keys = set(composition["slots"]) if composition is not None else _TEXT_PARAMS.get(op, set())
    extra = _EXTRA_PARAMS.get(op, set())
    enum = _ENUM_PARAMS.get(op)
    if not isinstance(params, dict) or params.keys() - (text_keys | extra | ({enum[0]} if enum else set())):
        raise ValueError(f"unsupported {op} parameters")
    if enum and enum[0] in params and params[enum[0]] not in enum[1]:
        raise ValueError(f"unsupported {op} {enum[0]}")
    if op == "Select":
        if "description" not in params or any(type(params.get(k, False)) is not bool for k in extra):
            raise ValueError("Select needs a description and boolean selection controls")
        if params.get("require_unique") and params.get("allow_empty"):
            raise ValueError("a unique object selection cannot also allow an empty list")
    if op == "Read":
        scale = params.get("scale", 3)
        if type(scale) not in (float, int) or not 0 < scale <= 4:
            raise ValueError("Read scale must be positive and at most 4")
    values = [params[k] for k in text_keys if k in params]
    if op == "Compare" and "labels" in params:
        labels = params["labels"]
        if not isinstance(labels, list) or len(labels) != 2:
            raise ValueError("Compare labels must be a pair of strings")
        values.extend(labels)
    used = set()
    for value in values:
        if not _nonempty(value):
            raise ValueError(f"{op} text parameters must be nonempty")
        if "{" in value or "}" in value:
            if value not in {"${" + slot + "}" for slot in slots}:
                raise ValueError("operator parameter placeholders must have the form ${slot}")
            used.add(value[2:-1])
    return used


def _check_template(template, library):
    required = {"name", "propose", "slots", "steps", "question", "answer_from"}
    if not isinstance(template, dict) or set(template) != required:
        raise ValueError("template fields must be " + ", ".join(sorted(required)))
    name = template["name"]
    if not _nonempty(name) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
        raise ValueError("invalid template name")
    for other in library.get("templates", []):
        if name == other["name"]:
            raise ValueError("template name already exists")
    if not _nonempty(template["propose"]) or not _nonempty(template["question"]):
        raise ValueError("template needs a proposal instruction and question")
    slots = template["slots"]
    if not isinstance(slots, list) or not slots or not all(map(_symbol, slots)) or len(slots) != len(set(slots)):
        raise ValueError("slots must be unique identifiers")
    used_slots = set()
    for literal, field, spec, conversion in Formatter().parse(template["question"]):
        if field is not None:
            if field not in slots or spec or conversion or literal.endswith("$"):
                raise ValueError("question placeholders must have the form {slot}")
            used_slots.add(field)
    if not used_slots:
        raise ValueError("question must reference a declared proposal slot")
    compositions = composition_definitions(library)
    used_slots.update(_check_plan(template["steps"], template["answer_from"], library, slots,
                                  compositions=compositions))
    if used_slots != set(slots):
        raise ValueError("every declared slot must be used")
    body = _template_body(expand_template(template, compositions))
    if any(body == _template_body(expand_template(other, compositions)) for other in library.get("templates", [])):
        raise ValueError("renaming an existing template or its steps does not change generation")


def _check_plan(steps, answer_from, library, slots, *, compositions, ports=None, result_type=None):
    if not isinstance(steps, list) or not steps:
        raise ValueError("execution steps are required")
    available = {op["name"] for op in library["operators"]}
    types, consumed, used_slots = dict(ports or {}), set(), set()
    for step in steps:
        if not isinstance(step, dict) or not _symbol(step.get("id")) or step["id"] in types:
            raise ValueError("step requires a unique identifier")
        if set(step) - {"id", "op", "inputs", "params"}:
            raise ValueError("unsupported step field")
        op = step.get("op")
        if not isinstance(op, str) or op not in available or (op not in _STEP_TYPES and op not in compositions):
            raise ValueError("unknown or unavailable operator")
        inputs = step.get("inputs", [])
        composition = compositions.get(op)
        if composition is None:
            expected, result = _STEP_TYPES[op]
        else:
            expected = tuple({port["type"]} for port in composition["inputs"])
            result = composition["result"]
        if (not isinstance(inputs, list) or len(inputs) != len(expected)
                or any(not _symbol(ref) or types.get(ref) not in allowed for ref, allowed in zip(inputs, expected))):
            raise ValueError(f"{op} inputs must reference compatible earlier steps")
        if len(set(inputs)) != len(inputs):
            raise ValueError("binary operations require two distinct inputs")
        params = step.get("params", {})
        used_slots.update(_parameter_slots(op, params, slots, composition))
        types[step["id"]] = "object" if op == "Select" and params.get("require_unique") else result
        consumed.update(inputs)
    if (not _symbol(answer_from) or answer_from in (ports or {}) or types.keys() - consumed != {answer_from}):
        raise ValueError("every step must contribute to a single answer_from result")
    if result_type is not None and types[answer_from] != result_type:
        raise ValueError("composition output does not match its declared result type")
    if result_type is None and types[answer_from] == "list":
        raise ValueError("every step must contribute to a single answer_from result")
    return used_slots


def composition_definitions(library):
    """Read the executable composition definitions in declaration order."""
    compositions = {}
    for operator in library.get("operators", []):
        if "steps" not in operator:
            continue
        name = operator.get("name")
        required = {"name", "description", "inputs", "slots", "steps", "answer_from", "result"}
        if (set(operator) != required or not _symbol(name) or not _nonempty(operator["description"])
                or not isinstance(operator["result"], str) or operator["result"] not in _TYPES):
            raise ValueError("invalid composition definition")
        if sum(other["name"] == name for other in library["operators"]) != 1 or name in _STEP_TYPES:
            raise ValueError("composition operator name already exists")
        slots, inputs = operator["slots"], operator["inputs"]
        if not isinstance(slots, list) or not all(map(_symbol, slots)) or len(slots) != len(set(slots)):
            raise ValueError("composition slots must be unique identifiers")
        if not isinstance(inputs, list) or any(
            not isinstance(port, dict) or set(port) != {"id", "type"} or not _symbol(port["id"])
            or not isinstance(port["type"], str) or port["type"] not in _TYPES for port in inputs
        ):
            raise ValueError("composition needs a list of typed input ports")
        ports = {port["id"]: port["type"] for port in inputs}
        if len(ports) != len(inputs):
            raise ValueError("composition input port names must be unique")
        used = _check_plan(operator["steps"], operator["answer_from"], library, slots,
                           compositions=compositions, ports=ports, result_type=operator["result"])
        if used != set(slots):
            raise ValueError("every composition slot must be used")
        compositions[name] = operator
    return compositions


def expand_template(template, compositions):
    """Inline a composition's checked DAG with locally scoped identifiers and arguments."""
    if not compositions:
        return template
    steps = []

    def expand(plan, refs, prefix, bindings, parents):
        for step in plan["steps"]:
            key = prefix + step["id"]
            inputs = [refs[ref] for ref in step.get("inputs", [])]
            params = _fill_slots(step.get("params", {}), bindings)
            operator = compositions.get(step["op"])
            if operator is not None:
                if len(parents) >= MAX_COMPOSITION_DEPTH:
                    raise ValueError(f"composition nesting exceeds {MAX_COMPOSITION_DEPTH} levels")
                ports = {port["id"]: value for port, value in zip(operator["inputs"], inputs, strict=True)}
                chain = [*parents, {"id": key, "op": step["op"]}]
                refs[step["id"]] = expand(operator, ports, key + "/", params, chain)
            else:
                item = {**step, "id": key, "inputs": inputs, "params": params}
                if parents:
                    item["compositions"] = parents
                steps.append(item)
                if len(steps) > MAX_TEMPLATE_STEPS:
                    raise ValueError(f"template exceeds {MAX_TEMPLATE_STEPS} executable steps")
                refs[step["id"]] = key
        return refs[plan["answer_from"]]

    answer_from = expand(template, {}, "", {}, [])
    return {**template, "steps": steps, "answer_from": answer_from}


def operator_interfaces(library):
    """Expose the same primitives and composition bodies used at execution."""
    definitions = {}
    for operator in library["operators"]:
        name = operator["name"]
        if "steps" in operator:
            definitions[name] = operator
            continue
        if name not in _STEP_TYPES:
            continue
        inputs, result = _STEP_TYPES[name]
        enum = _ENUM_PARAMS.get(name)
        definitions[name] = {
            "description": operator.get("description", ""),
            "editable_prompt": operator.get("prompt") if name in _MODEL_OPERATORS else None,
            "inputs": [sorted(types) for types in inputs],
            "result": result,
            "text_params": sorted(_TEXT_PARAMS.get(name, ())),
            "other_params": sorted(_EXTRA_PARAMS.get(name, ())),
        }
        if enum:
            definitions[name]["enum_param"] = {enum[0]: enum[1]}
        if name == "Select":
            definitions[name]["result_when_require_unique"] = "object"
    return definitions
