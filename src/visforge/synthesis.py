"""出题：一张图按规则库里的计划模板走，产出候选题；走不通记录卡在哪一步。"""

import json
from collections import Counter
from pathlib import Path

from .dataset import dedup_key
from .models import extract_json, response_excerpt
from .operators import OperatorResult, apply_operator
from .records import Candidate, Seed


def _find_operator(library, name):
    for op in library.get("operators", []):
        if op.get("name") == name:
            return op
    return None


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


def _binding_error(choice, templates):
    name = choice.get("template") if isinstance(choice, dict) else None
    if not isinstance(name, str) or name not in templates:
        return "unknown_template"
    values = choice.get("slots")
    if (not isinstance(values, dict) or set(values) != set(templates[name]["slots"])
            or not all(isinstance(v, str) and v.strip() for v in values.values())):
        return "invalid_slots"
    if not isinstance(choice.get("evidence"), str) or not choice["evidence"].strip():
        return "missing_evidence"
    return None


def _binding_key(choice):
    return choice["template"], tuple(sorted(
        (slot, " ".join(value.casefold().split())) for slot, value in choice["slots"].items()
    ))


def _select_bindings(model, image, templates, library, budget, instances):
    names = sorted(templates)
    trace = {"status": "skipped", "available_templates": names, "budget": budget,
             "instances_per_template": instances, "selected_templates": [], "choices": [], "issues": []}
    if budget == 0 or not names:
        return trace
    catalog = []
    used_operators = set()
    for name in names:
        tpl = templates[name]
        row = {key: tpl[key] for key in ("name", "propose", "slots", "question", "answer_from")}
        row["steps"] = [{key: step[key] for key in ("id", "op", "inputs", "params") if key in step}
                        for step in tpl["steps"]]
        row["operators"] = [step["op"] for step in tpl["steps"]]
        used_operators.update(row["operators"])
        catalog.append(row)
    definitions = []
    for name in sorted(used_operators):
        op = _find_operator(library, name)
        if op is not None:
            definitions.append({key: op[key] for key in ("name", "description", "guard", "guards", "prompt") if key in op})
    prompt = (
        "You are writing questions from an image. Examine the image and choose from the entire template library. "
        "Rank concrete bindings by the visible evidence that their full operator chains can succeed. "
        "Respect operator requirements, step parameters and the order of each step's inputs. "
        "Choose different applicable templates and targets before repeating a template. Use distinct objects, "
        "groups, comparisons or text regions for additional bindings. Do not guess answers or invent evidence. "
        f"Return up to {budget} bindings total, at most {instances} per template. When the image supports them, "
        "use the available attempts to cover different questions. Return fewer or an empty choices list if warranted. "
        "Use exact template names and slot keys. Each slot is a short descriptive string, and evidence is one "
        "short visible cue. Only output JSON of the form "
        '{"choices":[{"template":"NAME","slots":{"SLOT":"DESCRIPTION"},"evidence":"VISIBLE CUE"}]}.\n'
        "Operator definitions:\n" + json.dumps(definitions, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\nTemplate library:\n" + json.dumps(catalog, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    )
    text, meta = model({"text": prompt, "images": [image], "generation_diagnostics": True,
                        "max_new_tokens": min(4096, max(512, 256 + 128 * budget))})
    generation = (meta or {}).get("generation") or {}
    trace["planner_model"] = (meta or {}).get("model")
    trace["generation"] = {key: generation[key] for key in ("input_tokens", "generated_tokens", "max_new_tokens")
                           if type(generation.get(key)) is int}
    if generation.get("finish_reason") in ("length", "eos", "other"):
        trace["generation"]["finish_reason"] = generation["finish_reason"]
    parsed = extract_json(text, allow_array=True)
    rows = parsed.get("choices") if isinstance(parsed, dict) else parsed
    if not isinstance(rows, list):
        trace["issues"].append({"reason": "invalid_json" if parsed is None else "missing_choices"})
        trace["status"] = "invalid"
        trace["response_excerpt"] = response_excerpt(text)
        return trace
    counts, seen, ranked = Counter(), set(), []
    for rank, raw in enumerate(rows, 1):
        error = _binding_error(raw, templates)
        if not error:
            name = raw["template"]
            key = _binding_key(raw)
            if key in seen:
                error = "duplicate_binding"
            elif counts[name] >= instances:
                error = "instance_cap"
        if error:
            trace["issues"].append({"rank": rank, "reason": error})
            continue
        seen.add(key)
        counts[name] += 1
        choice = {"rank": rank, "template": name, "slots": {k: v.strip() for k, v in raw["slots"].items()},
                  "evidence": raw["evidence"].strip()}
        ranked.append((counts[name], choice))
    # Among model-selected bindings only, give each family one opportunity before its next instance.
    for _, choice in sorted(ranked, key=lambda pair: pair[0]):
        if len(trace["choices"]) < budget:
            trace["choices"].append(choice)
        else:
            trace["issues"].append({"rank": choice["rank"], "reason": "budget"})
    trace["selected_templates"] = list(dict.fromkeys(choice["template"] for choice in trace["choices"]))
    trace["status"] = "selected" if trace["choices"] else ("invalid" if rows else "abstained")
    if trace["status"] == "invalid":
        trace["response_excerpt"] = response_excerpt(text)
    return trace


def _check_frozen(trace, templates, budget, instances):
    if (not isinstance(trace, dict) or trace.get("available_templates") != sorted(templates)
            or trace.get("budget") != budget or trace.get("instances_per_template") != instances):
        raise ValueError("frozen selection does not match this template catalog or budget")
    choices = trace.get("choices")
    if not isinstance(choices, list) or len(choices) > budget or any(_binding_error(c, templates) for c in choices):
        raise ValueError("invalid bindings in frozen selection")
    ranks = [c.get("rank") for c in choices]
    names = [c["template"] for c in choices]
    if (any(type(rank) is not int or rank < 1 for rank in ranks) or len(set(ranks)) != len(ranks)
            or len({_binding_key(c) for c in choices}) != len(choices)
            or any(n > instances for n in Counter(names).values())
            or trace.get("selected_templates") != list(dict.fromkeys(names))
            or trace.get("status") not in (("selected",) if choices else ("abstained", "invalid", "skipped"))
            or not isinstance(trace.get("issues"), list)):
        raise ValueError("inconsistent frozen selection")


def _object_answer(obj, inputs):
    """Give a human-readable referent and distinguish repeated names using the recorded boxes."""
    name, box = obj.get("name"), obj.get("bbox")
    if not isinstance(name, str) or not name.strip() or not box:
        return None
    peers = [item["bbox"] for value in inputs if isinstance(value, list) for item in value
             if isinstance(item, dict) and str(item.get("name", "")).casefold() == name.casefold() and item.get("bbox")]
    if len(peers) < 2:
        return name
    if box not in peers:
        return None
    peers.remove(box)
    for axis, first, last in ((0, "leftmost", "rightmost"), (1, "topmost", "bottommost")):
        if all(box[axis + 2] <= peer[axis] for peer in peers):
            return f"the {first} {name}"
        if all(box[axis] >= peer[axis + 2] for peer in peers):
            return f"the {last} {name}"
    return None


def _run_template(tpl, image, library, model, proposal):
    """按顺序执行模板步骤。

    返回组合结果、执行计划和可选的失败记录。
    """
    steps, plan, evidence = {}, [{"template": tpl["name"]}], []
    for step in tpl["steps"]:
        op = _find_operator(library, step["op"])
        params = _fill_slots(step.get("params", {}), proposal)
        ctx = {
            "image": image,
            "model": model,
            "steps": steps,
            "step": {"id": step["id"], "params": params, "inputs": step.get("inputs", [])},
        }
        if op is None:
            result = OperatorResult(False, failure_reason=f"operator {step['op']} not in the library")
        else:
            result = apply_operator(op, ctx)
        steps[step["id"]] = result
        plan.append(
            {
                "id": step["id"],
                "op": step["op"],
                "ok": result.ok,
                "value": result.value,
                "evidence": result.evidence,
                "failure_reason": "" if result.ok else result.failure_reason,
            }
        )
        if not result.ok:
            failure = {"step": step["id"], "operator": step["op"], "reason": result.failure_reason}
            if result.diagnostic is not None:
                failure["diagnostic"] = result.diagnostic
            return None, plan, failure
        evidence.extend(result.evidence)
    try:
        question = tpl["question"].format(**proposal)
    except (KeyError, IndexError):
        return None, plan, {"step": "compose", "reason": "question template is missing slots"}
    answer = steps[tpl["answer_from"]].value
    if isinstance(answer, dict):
        terminal = next(step for step in tpl["steps"] if step["id"] == tpl["answer_from"])
        answer = _object_answer(answer, [steps[i].value for i in terminal.get("inputs", [])])
        if answer is None:
            return None, plan, {"step": "compose", "reason": "the selected object has no unique textual description"}
    elif isinstance(answer, bool):
        answer = "yes" if answer else "no"
    return (question, answer, evidence), plan, None


def generate(image_path, library, settings):
    """Plan grounded template bindings for one image, then execute the selected chains.

    budget bounds total attempted bindings and instances_per_template caps any one template.
    on_selection can checkpoint the route before execution. Pass it back as frozen_selection
    to replay the same route for the same image and catalog without another planner call.
    """
    seed = image_path if isinstance(image_path, Seed) else Seed(
        path=str(image_path), group=Path(str(image_path)).parent.name
    )
    image = str(seed.path)
    model = settings["model"]
    catalog = library.get("templates") or []
    templates = {tpl["name"]: tpl for tpl in catalog}
    if len(templates) != len(catalog):
        raise ValueError("template names must be unique for image planning")
    instances = max(1, int(settings.get("instances_per_template", 3)))
    budget = len(templates) * instances
    if settings.get("budget") is not None:
        budget = max(0, int(settings["budget"]))
    if settings.get("frozen_selection") is not None:
        selection = settings["frozen_selection"]
        _check_frozen(selection, templates, budget, instances)
    else:
        selection = _select_bindings(model, image, templates, library, budget, instances)
        if settings.get("on_selection") is not None:
            settings["on_selection"](selection)
    candidates, failures, summary = [], [], {}
    operator_attempts = Counter()
    duplicates = 0
    seen_questions = set()
    if selection["status"] == "invalid":
        failures.append({"template": None, "step": "selection", "reason": "planner returned no usable binding",
                         "issues": selection["issues"]})
    for choice in selection["choices"]:
        name = choice["template"]
        counts = summary.setdefault(name, {"attempts": 0, "candidates": 0, "failures": 0})
        counts["attempts"] += 1
        composed, plan, failure = _run_template(templates[name], image, library, model, choice["slots"])
        operator_attempts.update(step["op"] for step in plan[1:])
        if composed is None:
            failures.append({"template": name, "selection_rank": choice["rank"], **failure})
            counts["failures"] += 1
            continue
        question, answer, evidence = composed
        key = dedup_key(image, question)
        if key in seen_questions:
            duplicates += 1
            continue
        seen_questions.add(key)
        plan[0]["selection_rank"] = choice["rank"]
        candidates.append(Candidate(seed=seed, question=question, answer=answer, evidence=evidence, plan=plan))
        counts["candidates"] += 1
    return {"image": image, "candidates": candidates, "failures": failures, "duplicates": duplicates,
            "summary": summary, "selection": selection, "operator_attempts": dict(operator_attempts)}
