"""Observer edits and paired evaluation on fresh, identically budgeted images."""

import copy
import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from .components import (DEFAULT_PLANNER_PROMPT, _MODEL_OPERATORS, _check_template, _nonempty, _template_body,
                         composition_definitions, expand_template, operator_interfaces)
from .dataset import normalize_question
from .evaluation import h_s, score
from .history import library_version, observer_history
from .records import Candidate, QuestionView
from .structured import OutputContract
from .synthesis import generate
from .verification import answer_blind, audit


_EDIT_FORMS = {
    "add_template": {"template": "name, propose, slots, steps, question, answer_from"},
    "update_template": {"name": "existing template", "changes": "object with changed template fields"},
    "remove_template": {"name": "existing template"},
    "set_planner_prompt": {"prompt": "replacement instruction for how to rank and choose image bindings"},
    "modify": {"name": "active model operator", "prompt": "replacement instruction", "description": "optional"},
    "add_composition": {"operator": "name, description, inputs, slots, steps, answer_from, result"},
    "batch": {"edits": "ordered list of 1 to 6 dependent edits"},
}
_OBSERVER_OUTPUT = OutputContract({"action": "batch"}, fields={"action": str})


def _apply_one(new, edit):
    action = edit.get("action") if isinstance(edit, dict) else None
    if not isinstance(action, str) or action not in _EDIT_FORMS or action == "batch":
        raise ValueError("edit must use a supported action")
    if edit.keys() - ({"action", "hypothesis"} | _EDIT_FORMS[action].keys()):
        raise ValueError(f"unsupported fields for {action}")
    templates = new.setdefault("templates", [])
    if action == "add_composition":
        operator = edit.get("operator")
        if not isinstance(operator, dict) or "steps" not in operator:
            raise ValueError("add_composition requires an executable operator definition")
        new["operators"].append(operator)
        composition_definitions(new)
        return
    if action == "set_planner_prompt":
        prompt = edit.get("prompt")
        if not _nonempty(prompt) or len(prompt.strip()) > 2800:
            raise ValueError("planner prompt must have 1 to 2800 characters")
        if prompt.strip() == new.get("planner_prompt", DEFAULT_PLANNER_PROMPT):
            raise ValueError("edit must change the planner prompt")
        new["planner_prompt"] = prompt.strip()
        return
    if edit.get("action") == "add_template":
        template = edit.get("template")
        _check_template(template, new)
        templates.append(template)
        return
    name = edit.get("name")
    if action in ("update_template", "remove_template"):
        index = next((index for index, item in enumerate(templates) if item["name"] == name), None)
        if index is None:
            raise ValueError("edit must name an existing template")
        original = templates.pop(index)
        if action == "remove_template":
            if not templates:
                raise ValueError("the planner needs at least one template")
            return
        changes = edit.get("changes")
        if not isinstance(changes, dict) or not changes or changes.keys() - (original.keys() - {"name"}):
            raise ValueError("update_template needs changes to existing fields other than name")
        revised = {**original, **changes}
        _check_template(revised, new)
        if _template_body(revised) == _template_body(original):
            raise ValueError("edit must change the template")
        templates.insert(index, revised)
        return
    compositions = composition_definitions(new)
    active = {step["op"] for tpl in templates for step in expand_template(tpl, compositions)["steps"]}
    if not isinstance(name, str) or name not in _MODEL_OPERATORS or name not in active:
        raise ValueError("operator must use its prompt in a library template")
    op = next((op for op in new["operators"] if op["name"] == name), None)
    prompt = edit.get("prompt")
    if op is None or not _nonempty(prompt) or prompt.strip() == op.get("prompt", "").strip():
        raise ValueError("edit must change the executable prompt")
    op["prompt"] = prompt.strip()
    if "description" in edit:
        if not isinstance(edit["description"], str):
            raise ValueError("operator description must be text")
        if edit["description"].strip():
            op["description"] = edit["description"].strip()


def apply_edit(library, edit, *, budget=None):
    """Apply an edit or ordered batch atomically, then check that new operators can be executed."""
    if not isinstance(edit, dict) or (budget is not None and budget < 1):
        raise ValueError("an executable edit and positive attempt budget are required")
    if "hypothesis" in edit and not _nonempty(edit["hypothesis"]):
        raise ValueError("hypothesis must be nonempty text")
    edits = edit.get("edits") if edit.get("action") == "batch" else [edit]
    if (not isinstance(edits, list) or not 1 <= len(edits) <= 6
            or (edit.get("action") == "batch" and edit.keys() - {"action", "edits", "hypothesis"})):
        raise ValueError("batch requires an ordered list of 1 to 6 edits")
    new = copy.deepcopy(library)
    for change in copy.deepcopy(edits):
        _apply_one(new, change)
    compositions = composition_definitions(new)
    added = set(compositions) - set(composition_definitions(library))
    if added:
        used = {parent["op"] for template in new["templates"]
                for step in expand_template(template, compositions)["steps"]
                for parent in step.get("compositions", [])}
        if added - used:
            raise ValueError(f"new compositions need a calling template: {', '.join(sorted(added - used))}")
    if new == library:
        raise ValueError("edit must change the executable library")
    return new


def _observer_edit(text, library, budget):
    """Compile an explicit observer choice, then validate the resulting executable edit."""
    if not _nonempty(text):
        raise ValueError("empty observer response")
    reply = text.strip()
    if re.fullmatch(r"no[ _-]edit(?:[.!:]\s*.*)?", reply, re.IGNORECASE | re.DOTALL):
        return None
    short = re.fullmatch(r"SELECT_PROMPT:[ \t]*(.+)", reply, re.IGNORECASE)
    edit = {"action": "modify", "name": "Select", "prompt": short[1]} if short else _OBSERVER_OUTPUT.parse(reply)
    if edit.get("action") == "no_edit":
        return None
    if edit.get("action") == "read_object":
        fields = ("name", "propose", "what", "question")
        if not all(_nonempty(edit.get(key)) for key in fields):
            raise ValueError("read_object requires text fields name, propose, what, question")
        template = {key: edit[key] for key in ("name", "propose", "question")}
        template.update(slots=["target"], answer_from="value", steps=[
            {"id": "obj", "op": "Select", "inputs": [],
             "params": {"description": "${target}", "require_unique": True}},
            {"id": "value", "op": "Read", "inputs": ["obj"], "params": {"what": edit["what"]}},
        ])
        edit = {"action": "add_template", "template": template}
    apply_edit(library, edit, budget=budget)
    return edit


def _observer_summary(records):
    """Count all outcomes, then show representative grounded cases in a bounded context."""
    by_template, planner_issues, operator_failures = {}, Counter(), {}
    operator_attempts = Counter()
    image_routes, samples = [], {}
    for record in records:
        source = (record.get("seed") or {}).get("group", "")
        if "selection" in record:
            trace = record["selection"]
            operator_attempts.update(record.get("operator_attempts", {}))
            issues = Counter(str(i.get("reason", "invalid") if isinstance(i, dict) else i)[:80]
                             for i in trace["issues"])
            planner_issues.update(issues)
            if len(image_routes) < 32:
                image_routes.append({"source": source, "status": trace["status"],
                                     "selected": trace["selected_templates"], "issues": dict(issues)})
            for name, counts in record.get("summary", {}).items():
                stats = by_template.setdefault(name, Counter())
                stats["attempts"] += counts["attempts"]
                stats["candidates"] += counts["candidates"]
            for choice in trace["choices"]:
                key = ("selection", source, choice["template"])
                samples.setdefault(key, {"source": source, "template": choice["template"],
                                         "visible_evidence": str(choice["evidence"])[:180]})
            continue
        name = record.get("template") or "planner"
        stats = by_template.setdefault(name, Counter())
        status = record.get("status") or "construction_failure"
        stats[status] += 1
        for role in ("target", "reference"):
            correct = record.get(f"{role}_correct", [])
            stats[f"{role}_samples"] += len(correct)
            stats[f"{role}_correct"] += sum(correct)
        failure = record.get("diagnostic") or {}
        category = failure.get("subtype", record.get("step", status))
        if record.get("operator"):
            subtype = failure.get("subtype", "operator_rejected")
            if subtype == "invalid_json" and failure.get("generation", {}).get("finish_reason") == "length":
                subtype = "truncated_json"
            operator_failures.setdefault(record["operator"], Counter())[subtype] += 1
        key = ("execution", source, name, category, tuple(record.get("target_correct", [])))
        if key not in samples:
            sample = {"source": source, "template": name, "status": status}
            for field in ("question", "answer", "reason", "operator", "step", "blind", "reference",
                          "target_answers", "reference_answers", "target_correct", "reference_correct"):
                if field in record:
                    sample[field] = str(record[field])[:200]
            if record.get("audit"):
                sample["audit_reason"] = str(record["audit"].get("reason", ""))[:240]
            if failure:
                sample["diagnostic"] = failure.get("subtype")
            samples[key] = sample
    # Round robin over sources so domains appearing late in the manifest have a voice.
    queues = {}
    for key, sample in samples.items():
        queues.setdefault((sample["source"], key[0]), []).append(sample)
    picked = []
    while queues and len(picked) < 32:
        for key in list(queues):
            picked.append(queues[key].pop(0))
            if not queues[key]:
                del queues[key]
            if len(picked) == 32:
                break
    return json.dumps({"by_template": by_template, "planner_issues": planner_issues,
                       "operator_attempts": operator_attempts, "operator_failures": operator_failures,
                       "source_routes": image_routes, "examples": picked}, ensure_ascii=False)


def propose_edit(library, records, model, *, budget=None, on_response=None, repair=False,
                 history=(), rejected=None):
    """Record each local response and optionally request one correction of invalid output."""
    templates = sorted(library["templates"], key=lambda x: x["name"])
    operators = operator_interfaces(library)
    prompt = (
        "You are the rule-library observer. Study the development results below. Design one reusable "
        "change to make valid questions harder for the fixed target solver while preserving visual "
        "answerability and verified yield. Correct observed failure modes using concrete evidence. "
        "Return one JSON object with an action and a short hypothesis tied to the observed evidence, or reply NO_EDIT. "
        "Use a batch when component changes depend on each other. Declare a composition before its caller. "
        "Composition inputs are an ordered list of {id, type} ports, slots are textual parameters, "
        "and steps have id, op, optional inputs and params. Bind parameter references as ${slot} "
        "and question references as {slot}. Connect each input and step to answer_from. "
        "Use a calling template to make a new composition available to the planner.\n"
        f"Edit actions and their fields: {json.dumps(_EDIT_FORMS, ensure_ascii=False)}\n"
        f"All candidate templates (the image planner may select any, up to {budget} bindings): {json.dumps(templates, ensure_ascii=False)}\n"
        f"Current planner strategy: {json.dumps(library.get('planner_prompt', DEFAULT_PLANNER_PROMPT), ensure_ascii=False)}\n"
        f"Current operator prompts: {json.dumps(operators, ensure_ascii=False)}"
        + (observer_history(history, library) if history else "")
        + "\n"
        f"Development results: {_observer_summary(records)}"
    )
    request = prompt
    for index in range(1, 3 if repair else 2):
        text, meta = model({"text": request, "generation_diagnostics": True, "max_new_tokens": 2048})
        raw = text if isinstance(text, str) else ""
        try:
            edit = _observer_edit(raw, library, budget)
            outcome, error = ("edit" if edit is not None else "declined"), None
            if edit is not None and rejected:
                version = library_version(apply_edit(library, edit, budget=budget))
                if version in rejected:
                    edit, outcome = None, "repeated"
                    error = f"This candidate was already evaluated on the same protocol and rejected: {rejected[version]}"
        except ValueError as exc:
            edit, outcome, error = None, "invalid", str(exc)
        if on_response is not None:
            on_response({"index": index, "raw": raw[:8192], "raw_truncated": len(raw) > 8192,
                         "model": (meta or {}).get("model"), "outcome": outcome, "validation_error": error})
        if outcome not in ("invalid", "repeated"):
            return edit
        request = (f"{prompt}\nYour previous proposal could not proceed: {error}. "
                   f"Previous reply: {raw[:1000]}\nReturn one corrected edit now.")
    return None


def _sample(view, model, k):
    responses = [answer_blind(view, model) for _ in range(k)]
    answers = [text for text, _ in responses]
    if not all(map(_nonempty, answers)):
        raise ValueError("cannot score an empty or non-text blind response")
    models = [(meta or {}).get("model") for _, meta in responses]
    return answers, models


SELECTION_POLICY = "image_planner_v1"


def generation_limits(settings, budget):
    budget = settings.get("budget", 8) if budget is None else budget
    instances = settings.get("instances_per_template", 3)
    if type(budget) is not int or budget < 0 or type(instances) is not int or instances < 1:
        raise ValueError("budget must be nonnegative and instances_per_template must be positive")
    return budget, instances


def evaluate_library(library, seeds, settings, budget=None, *, drafts=None, on_event=None):
    """Audit, deduplicate accepted questions per image, then score fresh blind samples.

    An explicit verifier performs both its own blind solve and the evidence audit. Legacy
    model/solver callers retain the solver for the audit's blind step and the model for
    the evidence step. The scored target answers are sampled only after acceptance.
    """
    k = int(settings.get("k", 1))
    budget, instances = generation_limits(settings, budget)
    if k < 1:
        raise ValueError("k must be positive")
    model, solver = settings["model"], settings["solver"]
    verifier = settings.get("verifier")
    reference, judge = settings.get("reference_solver"), settings.get("judge")
    seeds = list(seeds)
    emit = on_event or (lambda event, row: None)
    records, failures, correct, ref_correct = [], [], [], []
    selections = []
    selection_counts = Counter()
    coverage, family_coverage = Counter(), Counter()
    seen, failed_images, duplicates = set(), 0, 0
    for index, seed in enumerate(seeds):
        provenance = {**asdict(seed), "path": str(seed.path)}
        if drafts is None:
            report = generate(seed, library, {"model": model, "budget": budget, "instances_per_template": instances})
            for cand in report["candidates"]:
                emit("candidate", asdict(cand))
            for failure in report["failures"]:
                emit("generation_failure", {"seed": provenance, **failure})
            candidates = [(cand, None) for cand in report["candidates"]]
        else:
            report = drafts[index]
            if report["seed"] != provenance or not report.get("generated"):
                raise ValueError("incomplete or mismatched per-image generation checkpoint")
            candidates = [(Candidate(seed, r["question"], r["answer"], r["evidence"], r["plan"]), r)
                          for r in report["candidates"]]
        selection_row = {"seed": provenance, "selection": report["selection"], "summary": report["summary"],
                         "generation_duplicates": report["duplicates"]}
        if "operator_attempts" in report:
            selection_row["operator_attempts"] = report["operator_attempts"]
        if drafts is None:
            emit("selection", selection_row)
        selections.append(selection_row)
        selection_counts[report["selection"]["status"]] += 1
        failures.extend({"seed": provenance, **failure} for failure in report["failures"])
        verified = 0
        for cand, raw in candidates:
            view = QuestionView(str(seed.path), cand.question)
            identity = {"seed": provenance, "template": cand.plan[0]["template"], "question": cand.question,
                        "selection_rank": cand.plan[0].get("selection_rank")}
            blind, meta = answer_blind(view, verifier if verifier is not None else solver)
            emit("verifier_blind", {**identity, "answer": blind})
            verdict = audit(cand, blind, verifier if verifier is not None else model)
            emit("audit", {**identity, **verdict})
            record = {
                **identity, "answer": cand.answer, "evidence": cand.evidence, "plan": cand.plan, "status": cand.status,
                "audit": verdict, "verifier_blind_model": (meta or {}).get("model"),
                "target_answers": [], "target_correct": [], "reference_answers": [], "reference_correct": [],
            }
            records.append(record)
            if cand.status != "accepted":
                continue
            key = (str(Path(seed.path).resolve()), normalize_question(cand.question))
            if key in seen:
                record["status"] = "duplicate"
                duplicates += 1
                continue
            seen.add(key)
            verified += 1
            coverage[record["template"]] += 1
            family = "->".join(step["op"] for step in cand.plan[1:])
            family_coverage[family] += 1
            if raw is None:
                answers, model_ids = _sample(view, solver, k)
                emit("target_answer", {**identity, "answers": answers, "models": model_ids})
            else:
                answers, model_ids = raw["target_answers"], raw["target_models"]
                if len(answers) != k or not all(map(_nonempty, answers)):
                    raise ValueError("incomplete frozen target answers")
            record.update(target_answers=answers, target_models=model_ids)
            hits = [score(text, str(cand.answer), judge) for text in answers]
            record["target_correct"] = hits
            correct.extend(hits)
            if reference is not None:
                answers, models = _sample(view, reference, k)
                emit("reference_answer", {**identity, "answers": answers, "models": models})
                hits = [score(text, str(cand.answer), judge) for text in answers]
                record.update(reference_answers=answers, reference_correct=hits, reference_models=models)
                ref_correct.extend(hits)
            emit("scored", record)
        if not verified:
            failed_images += 1
    return {
        "h_s": h_s(correct), "questions": len(seen), "samples": len(correct), "accepted": len(seen),
        "images": len(seeds), "failed_images": failed_images, "duplicates": duplicates,
        "yield_per_image": len(seen) / len(seeds) if seeds else 0.0,
        "coverage": dict(coverage), "family_coverage": dict(family_coverage),
        "reference_accuracy": sum(ref_correct) / len(ref_correct) if ref_correct else None,
        "per_question_correct": correct, "records": records, "failures": failures,
        "selection_policy": SELECTION_POLICY, "budget": budget, "instances_per_template": instances,
        "available_templates": sorted(t["name"] for t in library.get("templates", [])),
        "selections": selections, "selection_counts": dict(selection_counts),
        "attempts": sum(len(row["selection"]["choices"]) for row in selections),
        "generation_duplicates": sum(row["generation_duplicates"] for row in selections),
    }


def compare_libraries(old, new, seeds, settings, budget=None, *, baseline=None, on_event=None,
                      evaluate_arm=None):
    """Score both libraries on identical ordered sources with equal template attempts per image.

    Adoption requires two nonempty verified datasets, strictly higher target difficulty,
    and the configured candidate yield, distinct operator-chain and reference accuracy floors.
    The reference/coverage gates are optional. No evidence from the holdout goes to the observer.
    """
    floor = float(settings.get("min_yield_per_image", 0.0))
    min_coverage = settings.get("min_coverage", 0)
    ref_floor = settings.get("min_reference_accuracy")
    if not math.isfinite(floor) or floor < 0 or type(min_coverage) is not int or min_coverage < 0:
        raise ValueError("yield and coverage floors must be finite and nonnegative")
    if ref_floor is not None:
        ref_floor = float(ref_floor)
        if not 0 <= ref_floor <= 1 or settings.get("reference_solver") is None:
            raise ValueError("reference accuracy floor requires a reference solver and a value in [0, 1]")
    common_budget, instances = generation_limits(settings, budget)
    seeds = list(seeds)
    if baseline is not None and (baseline.get("selection_policy") != SELECTION_POLICY
                                 or baseline.get("budget") != common_budget
                                 or baseline.get("instances_per_template") != instances):
        raise ValueError("the frozen baseline used different image selection settings")
    def live_arm(label, library):
        callback = None if on_event is None else lambda event, row: on_event(label, event, row)
        return evaluate_library(library, seeds, settings, common_budget, on_event=callback)

    evaluate_arm = evaluate_arm or live_arm
    before = baseline if baseline is not None else evaluate_arm("old", old)
    after = evaluate_arm("new", new)
    reasons = []
    if not before["accepted"] or not after["accepted"]:
        reasons.append("both arms require verified questions")
    if after["h_s"] <= before["h_s"]:
        reasons.append(f"h_s not improved (new {after['h_s']:.3f} <= old {before['h_s']:.3f})")
    if after["yield_per_image"] < floor:
        reasons.append(f"per-image yield {after['yield_per_image']:.2f} below floor {floor:.2f}")
    if len(after["family_coverage"]) < min_coverage:
        reasons.append(f"distinct operator-chain coverage below floor {min_coverage}")
    if ref_floor is not None and (after["reference_accuracy"] is None or after["reference_accuracy"] < ref_floor):
        reasons.append(f"reference accuracy below floor {ref_floor:.3f}")
    return {
        "adopted": not reasons, "reason": "; ".join(reasons) or "higher measured difficulty and all floors met",
        "old": before, "new": after, "budget": common_budget, "instances_per_template": instances, "floor": floor,
        "min_coverage": min_coverage, "min_reference_accuracy": ref_floor,
    }
