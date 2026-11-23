"""Run one operator-library update on a small, source-disjoint image split."""

import csv
import fcntl
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from . import evolution, models
from .evaluation import make_judge
from .operators import seed_library
from .records import QuestionView
from .sources import load_seeds
from .synthesis import generate
from .verification import answer_blind


def _write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_mix(value):
    counts = {}
    for item in value.split(","):
        kind, equals, raw = item.partition("=")
        kind = kind.strip()
        if not kind or not equals or kind in counts or not raw.strip().isdigit() or int(raw) < 1:
            raise ValueError("Use unique domain names with positive counts, e.g. chart=2,document=1,photo=1.")
        counts[kind] = int(raw)
    return counts


def _split_rows(manifest, args):
    by_kind = defaultdict(list)
    seen_hashes = set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "group", "kind"} <= set(reader.fieldnames or ()):
            raise ValueError("Seed CSV must include path, group and kind.")
        for row in reader:
            if row.get("purpose", "").strip() not in ("", "candidate", "seed"):
                continue
            if not all(row[key].strip() for key in ("path", "kind", "group")):
                raise ValueError("Missing path, kind or source group in seed CSV.")
            path = (manifest.parent / row["path"]).resolve(strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if row.get("sha256") and digest != row["sha256"].lower():
                raise ValueError(f"Seed checksum changed: {path.name}")
            if digest not in seen_hashes:
                by_kind[row["kind"]].append({**row, "path": path, "sha256": digest})
                seen_hashes.add(digest)
    if not by_kind:
        raise ValueError("No eligible candidate/seed images in manifest.")
    # Assign the scarcest domains first. Use one image per parent across both sides.
    sides = {"development": [], "holdout": []}
    dev_counts = args.mix or {kind: args.per_kind for kind in by_kind}
    hold_counts = {} if args.reuse_development_for_comparison else (args.holdout_mix or dev_counts)
    counts = {"development": dev_counts, "holdout": hold_counts}
    rng = random.Random(args.split_seed)
    groups = set()
    for kind in sorted(dev_counts.keys() | hold_counts.keys(), key=lambda k: (len({r["group"] for r in by_kind[k]}), k)):
        rows = sorted(by_kind[kind], key=lambda r: (r["group"], r["sha256"]))
        rng.shuffle(rows)
        for index in range(max(dev_counts.get(kind, 0), hold_counts.get(kind, 0))):
            for name, side in sides.items():
                if index >= counts[name].get(kind, 0):
                    continue
                row = next((r for r in rows if r["group"] not in groups), None)
                if row is None:
                    raise ValueError(f"Insufficient distinct images and source groups for {kind} split.")
                groups.add(row["group"])
                side.append(row)
    if args.reuse_development_for_comparison:
        sides["holdout"] = list(sides["development"])
    return sides


def _prepare(args, out):
    manifest = Path(args.manifest).resolve(strict=True)
    instances = getattr(args, "instances_per_template", 3)
    if min(args.per_kind, args.budget, instances) < 1:
        raise ValueError("--per-kind, --budget and --instances-per-template must be positive.")
    if not 0 <= args.reference_floor <= 1 or not 0 <= args.yield_floor <= args.budget:
        raise ValueError("Reference floor must be in [0, 1] and yield floor in [0, budget].")
    setup = dict(manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 per_kind=args.per_kind, split_seed=args.split_seed, budget=args.budget,
                 reference_floor=args.reference_floor, yield_floor=args.yield_floor,
                 min_families=1, answers_per_question=1, selection_policy=evolution.SELECTION_POLICY,
                 instances_per_template=instances)
    if args.mix:
        setup["mix"] = args.mix
    if args.holdout_mix:
        setup["holdout_mix"] = args.holdout_mix
    if args.reuse_development_for_comparison:
        setup["comparison_population"] = "in_sample"
    statefile = out / "run.json"
    if statefile.exists():
        state = json.loads(statefile.read_text())
        if state["settings"].get("selection_policy") != evolution.SELECTION_POLICY:
            raise ValueError("Existing run used a different selection policy; preserve it and use another --out.")
        can_retry = state["status"] in ("interrupted", "running", "retrying_observer")
        if can_retry and args.draft_only and state.get("verifier_usage", {}).get("requests_attempted"):
            raise ValueError("Resume this interrupted evaluation in full mode to preserve verifier accounting.")
        if args.retry_observer:
            if (state["status"] != "no_usable_edit" or state.get("observer_attempt", 1) != 1
                    or not args.reuse_development_for_comparison or state.get("observer_outcome") == "declined"):
                raise ValueError("Observer retry requires one unusable edit and a frozen in-sample baseline.")
        elif state["status"] not in ("prepared", "drafting", "drafted") and not can_retry:
            raise ValueError("Output already records a run. Choose another --out.")
        if state["settings"] != setup:
            raise ValueError("Output already records a run or different settings. Choose another --out.")
        for name, digest in state["pinned_files"].items():
            if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Pinned run input changed: {name}")
        for side in ("development", "holdout"):
            with (out / f"{side}.csv").open(newline="") as stream:
                for row in csv.DictReader(stream):
                    if hashlib.sha256((out / row["path"]).read_bytes()).hexdigest() != row["sha256"]:
                        raise ValueError(f"Pinned seed changed: {row['path']}")
        return state
    if args.retry_observer:
        raise ValueError("Observer retry requires an existing run.")
    if any(out.iterdir()):
        raise ValueError("Choose an empty run directory.")
    sides = _split_rows(manifest, args)
    for name, rows in sides.items():
        with (out / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["path", "group", "page", "kind", "purpose", "sha256"]
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                purpose = "in_sample_comparison" if name == "holdout" and args.reuse_development_for_comparison else name
                writer.writerow({**row, "path": os.path.relpath(row["path"], out), "purpose": purpose})
        load_seeds(out / f"{name}.csv")
    _write_json(out / "initial-library.json", seed_library())
    pinned = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
              for name in ("initial-library.json", "development.csv", "holdout.csv")}
    attempts = args.budget
    # Per candidate: two audit calls, plus at most one target-equivalence call.
    # Each holdout arm also has a reference solve and possible equivalence call.
    limit = attempts * (3 * len(sides["development"]) + 2 * 5 * len(sides["holdout"]))
    if args.reuse_development_for_comparison:
        limit = attempts * 10 * len(sides["development"])
    state = dict(status="prepared", settings=setup, pinned_files=pinned,
                 external_call_upper_bound=limit, external_call_bound_excludes_retries=True,
                 images={name: len(rows) for name, rows in sides.items()})
    _write_json(statefile, state)
    return state


def _load_env(path):
    """Read plain KEY=value lines; preserve explicit environment overrides."""
    if not path.exists():
        return
    if path.stat().st_mode & 0o077:
        raise ValueError(f"Restrict key-file permissions with chmod 600 {path}.")
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, sep, value = line.removeprefix("export ").partition("=")
            if not sep or not key.strip().isidentifier():
                raise ValueError("Invalid config line. Use KEY=value.")
            value = value.strip()
            if len(value) > 1 and value[0] in {"'", '"'} and value[-1] == value[0]:
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)


def _append_event(out, phase, event, row):
    with (out / "progress.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"phase": phase, "event": event, **row}, ensure_ascii=False) + "\n")


def _draft_images(args, out, state, library, seeds, synthesis, target, phase):
    """Checkpoint the planner's full-image route before running any selected operators."""
    path = out / f"{phase}-drafts.json"
    library_hash = hashlib.sha256(json.dumps(library, sort_keys=True).encode()).hexdigest()
    drafts = json.loads(path.read_text()) if path.exists() else [
        {"seed": asdict(seed), "library_sha256": library_hash, "generated": False} for seed in seeds
    ]
    for seed, image in zip(seeds, drafts, strict=True):
        if image["seed"] != asdict(seed) or image["library_sha256"] != library_hash:
            raise ValueError(f"{phase} generation checkpoint uses different sources or library")
        if not image["generated"]:
            def checkpoint(selection):
                image["selection"] = selection
                _write_json(path, drafts)
                _append_event(out, phase, "selection", {"seed": image["seed"], "selection": selection})

            report = generate(seed, library, {
                "model": synthesis, "budget": args.budget,
                "instances_per_template": state["settings"]["instances_per_template"],
                "on_selection": checkpoint, "frozen_selection": image.get("selection"),
            })
            image.update(generated=True, selection=report["selection"], summary=report["summary"],
                         operator_attempts=report["operator_attempts"],
                         duplicates=report["duplicates"], failures=report["failures"],
                         candidates=[asdict(cand) for cand in report["candidates"]])
            _write_json(path, drafts)
            _append_event(out, phase, "generation_summary", {
                "seed": image["seed"], "summary": image["summary"], "generation_duplicates": image["duplicates"],
                "operator_attempts": image["operator_attempts"],
            })
            for raw in image["candidates"]:
                _append_event(out, phase, "candidate", raw)
            for failure in image["failures"]:
                _append_event(out, phase, "generation_failure", {"seed": image["seed"], **failure})
        for raw in image["candidates"]:
            if "target_answers" in raw:
                continue
            view = QuestionView(str(seed.path), raw["question"])
            answer, meta = answer_blind(view, target)
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError("empty local target response while drafting")
            raw.update(target_answers=[answer], target_models=[(meta or {}).get("model")])
            _write_json(path, drafts)
            _append_event(out, phase, "target_answer", {
                "seed": image["seed"], **raw["plan"][0], "question": raw["question"],
                "answers": raw["target_answers"], "models": raw["target_models"],
            })
    state["pinned_files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if phase == "development":
        state["draft_questions"] = sum(len(img["candidates"]) for img in drafts)
    _write_json(out / "run.json", state)
    return drafts


def _comparison_report(out, comparison, seeds):
    lines = ["# Old and new questions by image", "", f"Comparison population: {comparison['comparison_population']}",
             "", comparison["reason"], ""]
    with (out / "frozen-questions.jsonl").open("w", encoding="utf-8") as stream:
        for seed in seeds:
            lines.extend([f"## {seed.group} · {Path(seed.path).name}", ""])
            for arm in ("old", "new"):
                lines.extend([f"### {arm.capitalize()} library", ""])
                rows = [r for r in comparison[arm]["records"] if r["seed"]["path"] == seed.path and r["status"] == "accepted"]
                if not rows:
                    lines.extend(["No verified questions.", ""])
                for row in rows:
                    stream.write(json.dumps({"library": arm, **row}, ensure_ascii=False) + "\n")
                    lines.extend([f"- {row['question']}", f"  - Construction answer: {row['answer']}",
                                  f"  - Target answer: {row['target_answers']} · Correct: {row['target_correct']}",
                                  f"  - Reference answer: {row['reference_answers']} · Correct: {row['reference_correct']}", ""])
    (out / "comparison-by-image.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _archive_first_attempt(out):
    names = ["run.json", "proposed-edit.json"]
    if (out / "observer-responses.json").exists():
        names.append("observer-responses.json")
    files = {name: (out / name).read_bytes() for name in names}
    if json.loads(files["proposed-edit.json"]) is not None:
        raise ValueError("The first attempt already has an edit.")
    archive = out / "observer-attempt-1"
    if archive.exists():
        if any((archive / name).read_bytes() != data for name, data in files.items()):
            raise ValueError("Prior observer archive differs from the recorded first attempt.")
        return
    with tempfile.TemporaryDirectory(dir=out) as tempdir:
        draft = Path(tempdir) / "archive"
        draft.mkdir()
        for name, data in files.items():
            (draft / name).write_bytes(data)
        draft.rename(archive)


def _frozen_baseline(out, state):
    dev = json.loads((out / "development.json").read_text())
    accepted = [row for row in dev["records"] if row["status"] == "accepted"]
    exported = [json.loads(s) for s in (out / "development.jsonl").read_text().splitlines() if s.strip()]
    if (not accepted or len(accepted) != state["verified_development_questions"]
            or len(accepted) != dev["accepted"] or dev["images"] != state["images"]["holdout"]
            or [(r["question"], r["answer"]) for r in exported] != [(r["question"], r["answer"]) for r in accepted]
            or any(not row["target_correct"] or not row["reference_correct"] for row in accepted)):
        raise ValueError("Frozen baseline or its accepted-question export is incomplete.")
    return dev


def _verifier_for_phase(out, state, model):
    """Persist completed paid calls. Repeated blind samples have separate phase/ordinal slots."""
    folder = out / "verifier-calls" / f"attempt-{state['observer_attempt']}"
    keys = ("requests_attempted", "responses", "reported_prompt_tokens", "reported_completion_tokens",
            "unresolved_previous_attempts")
    previous = state.get("verifier_usage_before_retry", dict.fromkeys(keys, 0))
    current = dict.fromkeys(keys, 0)
    for path in folder.glob("*/*.json"):
        entry = json.loads(path.read_text())
        current["requests_attempted"] += entry["attempts"]
        current["unresolved_previous_attempts"] += entry.get("unresolved_previous_attempts", 0)
        if entry["status"] == "complete":
            current["responses"] += 1
            for field in ("prompt_tokens", "completion_tokens"):
                current[f"reported_{field}"] += entry["meta"]["usage"].get(field) or 0
    state["verifier_usage_this_attempt"] = current
    dispatched = set()

    def tally():
        totals = {key: previous.get(key, 0) + current[key] for key in keys}
        current["possible_unreported_usage"] = bool(current["unresolved_previous_attempts"])
        totals["possible_unreported_usage"] = bool(totals["unresolved_previous_attempts"])
        state["verifier_usage"] = totals
        _write_json(out / "run.json", state)

    def scoped(phase):
        scope = folder / phase
        scope.mkdir(parents=True, exist_ok=True)
        ordinal = 0

        def call(request):
            nonlocal ordinal
            ordinal += 1
            path = scope / f"{ordinal:06}.json"
            digest = hashlib.sha256(json.dumps(request, sort_keys=True, default=str).encode()).hexdigest()
            entry = {"request_sha256": digest, "attempts": 0}
            if path.exists():
                entry = json.loads(path.read_text())
                if entry["request_sha256"] != digest:
                    raise ValueError(f"Request changed at {phase} verifier call {ordinal}; preserving its checkpoint.")
                if entry["status"] == "complete":
                    return entry["text"], entry["meta"]
                if path in dispatched:
                    raise RuntimeError(f"{phase} verifier call {ordinal} already dispatched during this invocation.")
                if entry["status"] == "pending":
                    entry["unresolved_previous_attempts"] = entry.get("unresolved_previous_attempts", 0) + 1
                    entry["possible_unreported_usage"] = True
                    current["unresolved_previous_attempts"] += 1
            dispatched.add(path)
            entry.update(status="pending", attempts=entry["attempts"] + 1)
            current["requests_attempted"] += 1
            _write_json(path, entry)
            tally()
            try:
                text, meta = model(request)
            except Exception as error:
                entry.update(status="error", error_type=type(error).__name__)
                _write_json(path, entry)
                raise
            usage = (meta or {}).get("usage") or {}
            stored = {"model": (meta or {}).get("model"),
                      "usage": {field: usage.get(field) or 0 for field in ("prompt_tokens", "completion_tokens")}}
            entry.update(status="complete", text=text, meta=stored)
            entry.pop("error_type", None)
            _write_json(path, entry)
            current["responses"] += 1
            for field, tokens in stored["usage"].items():
                current[f"reported_{field}"] += tokens
            tally()
            return text, stored
        return call
    tally()
    return scoped


def _write_dataset(out, name, metrics):
    temp = out / (name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        for row in metrics["records"]:
            if row["status"] == "accepted":
                data = {"image": os.path.relpath(row["seed"]["path"], out), "group": row["seed"]["group"],
                        "page": row["seed"]["page"], "question": row["question"], "answer": row["answer"],
                        "template": row["template"], "evidence": row["evidence"], "plan": row["plan"]}
                stream.write(json.dumps(data, ensure_ascii=False) + "\n")
    temp.replace(out / name)


def _execute(args, out, state, roles, drafts, frozen=None):
    initial = json.loads((out / "initial-library.json").read_text())
    holdout = load_seeds(out / "holdout.csv")
    scoped_verifier = _verifier_for_phase(out, state, roles["verifier"])
    settings = dict(model=roles["synthesis"], solver=roles["target"], min_coverage=1,
                    min_reference_accuracy=args.reference_floor, min_yield_per_image=args.yield_floor,
                    k=1, instances_per_template=state["settings"]["instances_per_template"])

    def evaluate(phase, library, seeds, drafts, with_reference):
        verifier = scoped_verifier(phase)
        return evolution.evaluate_library(
            library, seeds, {**settings, "verifier": verifier, "judge": make_judge(verifier),
                             "reference_solver": verifier if with_reference else None}, budget=args.budget,
            drafts=drafts, on_event=lambda event, row: _append_event(out, phase, event, row))
    if frozen is None:
        development = load_seeds(out / "development.csv")
        dev = evaluate("development", initial, development, drafts, args.reuse_development_for_comparison)
        _write_json(out / "development.json", dev)
        _write_dataset(out, "development.jsonl", dev)
    else:
        dev = frozen
    for name in ("development.json", "development.jsonl"):
        state["pinned_files"][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()

    selected = initial
    chosen_metrics, chosen_arm, chosen_split = dev, "old", "development"
    state["verified_development_questions"] = dev["accepted"]
    if not dev["accepted"]:
        state["status"] = "no_verified_development_questions"
    else:
        observations = dev["selections"] + dev["records"] + dev["failures"]
        diagnostics = []

        def on_response(row):
            diagnostics.append(row)
            _write_json(out / "observer-responses.json", diagnostics)
            _append_event(out, "observer", "response", {"attempt": state["observer_attempt"], **row})

        if state.get("observer_result_attempt") == state["observer_attempt"]:
            edit = json.loads((out / "proposed-edit.json").read_text())
        else:
            edit = evolution.propose_edit(initial, observations, roles["observer"], budget=args.budget,
                                          repair=True, on_response=on_response)
            state["observer_outcome"] = diagnostics[-1]["outcome"]
            _write_json(out / "proposed-edit.json", edit)
            name = "proposed-edit.json"
            state["pinned_files"][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
            state["observer_result_attempt"] = state["observer_attempt"]
            _write_json(out / "run.json", state)
            _append_event(out, "observer", "edit", {"edit": edit})
        if edit is None:
            state["status"] = "observer_declined" if state["observer_outcome"] == "declined" else "no_usable_edit"
        else:
            candidate = evolution.apply_edit(initial, edit, budget=args.budget)
            _write_json(out / "candidate-library.json", candidate)
            def comparison_arm(phase, library):
                arm_drafts = _draft_images(args, out, state, library, holdout, roles["synthesis"], roles["target"], phase)
                return evaluate(phase, library, holdout, arm_drafts, True)

            settings["reference_solver"] = roles["verifier"]
            comparison = evolution.compare_libraries(
                initial, candidate, holdout, settings, budget=args.budget,
                baseline=dev if args.reuse_development_for_comparison else None,
                evaluate_arm=comparison_arm)
            comparison["comparison_population"] = "in_sample" if args.reuse_development_for_comparison else "held_out"
            _write_json(out / "comparison.json", comparison)
            _comparison_report(out, comparison, holdout)
            state["status"] = "adopted" if comparison["adopted"] else "retained"
            state["reason"] = comparison["reason"]
            if comparison["adopted"]:
                selected = candidate
            chosen_arm = "new" if comparison["adopted"] else "old"
            chosen_metrics, chosen_split = comparison[chosen_arm], comparison["comparison_population"]
    _write_dataset(out, "selected-data.jsonl", chosen_metrics)
    state["selected_data"] = {"path": "selected-data.jsonl", "library": chosen_arm,
                              "scope": chosen_split, "questions": chosen_metrics["accepted"]}
    _write_json(out / "selected-library.json", selected)
    _write_json(out / "run.json", state)


def run_cycle(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # flock is released automatically if a GPU job is interrupted.
    fd = os.open(out, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another process is using this run directory.") from error
        state = _prepare(args, out)
        if not args.prepare_only:
            if not args.model_path:
                raise ValueError("Provide --model-path for local GPU inference.")
            model_path = str(Path(args.model_path).resolve())
            if state.get("local_model", model_path) != model_path:
                raise ValueError("The pinned local checkpoint path differs from --model-path.")
            if args.draft_only:
                local = models.make_local_model(args.model_path)
                roles = {"synthesis": local, "target": local}
            else:
                _load_env(Path(args.env_file))
                verifier_id = args.verifier_model or os.environ.get("VISFORGE_VERIFIER_MODEL")
                if state.get("verifier"):
                    verifier_id = verifier_id or state.get("verifier")
                    if not verifier_id or verifier_id != state.get("verifier"):
                        raise ValueError("Observer retry must use the same verifier model as the frozen baseline.")
                roles = models.build_cycle_models(args.model_path, verifier_base_url=args.verifier_base_url,
                                                  verifier_model=verifier_id,
                                                  verifier_api_key_env=args.verifier_key_env)
            frozen = None
            if args.retry_observer:
                frozen = _frozen_baseline(out, state)
                _archive_first_attempt(out)
                state["verifier_usage_before_retry"] = state["verifier_usage"].copy()
                state.update(status="retrying_observer", observer_attempt=2)
            else:
                if not args.draft_only and "development.json" in state["pinned_files"]:
                    frozen = json.loads((out / "development.json").read_text())
                state.setdefault("observer_attempt", 1)
                state.update(status="drafting" if frozen is None else "running", local_model=model_path)
            _write_json(out / "run.json", state)
            try:
                drafts = None
                if frozen is None:
                    initial = json.loads((out / "initial-library.json").read_text())
                    drafts = _draft_images(args, out, state, initial, load_seeds(out / "development.csv"),
                                           roles["synthesis"], roles["target"], "development")
                    if args.draft_only:
                        state["status"] = "drafted"
                        _write_json(out / "run.json", state)
                if not args.draft_only:
                    state.update(status="running", verifier=verifier_id,
                                 reference_role="same verifier model, fresh blind call on comparison questions")
                    _write_json(out / "run.json", state)
                    _execute(args, out, state, roles, drafts, frozen=frozen)
            except BaseException as error:
                state.update(status="interrupted", error_type=type(error).__name__)
                _write_json(out / "run.json", state)
                raise
        print(json.dumps({"run": str(out), "status": state["status"], "images": state["images"]}))
        return 0
    finally:
        os.close(fd)
