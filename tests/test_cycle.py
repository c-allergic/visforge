import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from visforge.cli import main


def manifest(root):
    path = root / "seeds.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "kind", "group", "purpose"])
        writer.writeheader()
        for n in range(8):
            name = f"seed-{n}.png"
            Image.new("RGB", (640, 480), (n, 0, 0)).save(root / name)
            writer.writerow(dict(path=name, kind=["chart", "photo"][n % 2],
                                 group=f"group-{n // 2}", purpose="candidate"))
    return path


def test_prepare_pins_disjoint_images_and_reopens_without_a_model(tmp_path):
    path, out = manifest(tmp_path), tmp_path / "run"
    args = ["cycle", "--manifest", str(path), "--out", str(out), "--prepare-only"]
    assert main(args) == 0
    sides = []
    for split in ("development", "holdout"):
        with (out / f"{split}.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert {r["kind"] for r in rows} == {"chart", "photo"}
        assert all((out / r["path"]).is_file() for r in rows)
        sides.append({r["group"] for r in rows})
    assert sides[0].isdisjoint(sides[1])
    before = (out / "holdout.csv").read_bytes()
    assert main(args) == 0
    assert before == (out / "holdout.csv").read_bytes()
    assert json.loads((out / "run.json").read_text())["status"] == "prepared"


def test_entire_cycle_writes_verified_development_data_and_decision(tmp_path, monkeypatch):
    from fakes import FakeModel, generation_model, solver_model
    from visforge import models

    path, out = manifest(tmp_path), tmp_path / "run"
    roles = dict(synthesis=generation_model(), target=solver_model(),
                 observer=FakeModel(default=json.dumps({
                     "action": "modify", "name": "Select",
                     "prompt": "Look at the image, list all objects matching the description accurately."})),
                 verifier=generation_model())
    roles["verifier"].add("Answer the following question",
                          lambda r: "2" if "How many" in r["text"] else "TEA")
    roles["verifier"].add("equivalent", '{"equivalent": false}')
    monkeypatch.setattr(models, "build_cycle_models", lambda *a, **kw: roles)
    args = ["cycle", "--manifest", str(path), "--out", str(out),
            "--model-path", str(tmp_path), "--verifier-model", "test-verifier"]
    assert main(args + ["--prepare-only"]) == 0
    assert main(args) == 0
    run = json.loads((out / "run.json").read_text())
    assert run["status"] == "retained"
    assert (out / "selected-library.json").read_bytes() == (out / "initial-library.json").read_bytes()
    assert json.loads((out / "comparison.json").read_text())["adopted"] is False
    rows = [json.loads(s) for s in (out / "development.jsonl").read_text().splitlines()]
    assert rows and all((out / r["image"]).is_file() for r in rows)
    observer_prompt = roles["observer"].calls[0]["text"]
    with (out / "holdout.csv").open() as f:
        assert all(Path(r["path"]).name not in observer_prompt for r in csv.DictReader(f))
    with pytest.raises((SystemExit, ValueError)):
        main(args)


def test_duplicate_pixels_cannot_cross_split(tmp_path):
    path = manifest(tmp_path)
    for img in tmp_path.glob("*.png"):
        Image.new("RGB", (10, 10)).save(img)
    with pytest.raises((SystemExit, ValueError), match="(?i)distinct|group|split"):
        main(["cycle", "--manifest", str(path), "--out", str(tmp_path / "out"), "--prepare-only"])


def three_kinds(root):
    path = root / "three.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["path", "kind", "group"])
        writer.writeheader()
        for i, kind in enumerate(["chart"] * 4 + ["document"] * 2 + ["photo"] * 2):
            name = f"{kind}-{i}.png"
            Image.new("RGB", (640, 480), (i, 0, 0)).save(root / name)
            writer.writerow({"path": name, "kind": kind, "group": f"parent-{i}"})
    return path


@pytest.mark.parametrize("dev,extra,counts,overlap", [
    ("chart=2,document=1,photo=1", [], [4, 4], 0),
    ("chart=1,document=1", ["--holdout-mix", "chart=1,photo=1"], [2, 2], 0),
    ("chart=2,document=1,photo=1", ["--reuse-development-for-comparison"], [4, 4], 4),
])
def test_explicit_pilot_mix(tmp_path, dev, extra, counts, overlap):
    path, out = three_kinds(tmp_path), tmp_path / "run"
    main(["cycle", "--manifest", str(path), "--out", str(out), "--mix", dev, "--prepare-only", *extra])
    sides = []
    for side in ("development", "holdout"):
        with (out / f"{side}.csv").open() as stream:
            sides.append(list(csv.DictReader(stream)))
    assert [len(rows) for rows in sides] == counts
    groups = [{r["group"] for r in rows} for rows in sides]
    assert len(groups[0] & groups[1]) == overlap
    assert sum(row["kind"] == "chart" for row in sides[0]) == int(dev.split(",")[0][-1])
    if overlap:
        assert all(row["purpose"] == "in_sample_comparison" for row in sides[1])


def test_key_free_drafts_resume_exact_old_arm_on_same_four(tmp_path, monkeypatch):
    from fakes import FakeModel, generation_model
    from visforge import models

    out = tmp_path / "pilot"
    args = ["cycle", "--manifest", str(three_kinds(tmp_path)), "--out", str(out), "--model-path", str(tmp_path),
            "--mix", "chart=2,document=1,photo=1", "--reuse-development-for-comparison"]
    local = generation_model()
    local.add("Answer the following question", lambda r: "2" if "How many" in r["text"] else "no idea")
    monkeypatch.setattr(models, "make_local_model", lambda *a, **kw: local)
    main(args + ["--draft-only"])
    drafts = (out / "development-drafts.json").read_bytes()
    assert b"no idea" in drafts and b"evidence" in drafts
    assert json.loads((out / "run.json").read_text())["status"] == "drafted"
    initial_calls = len(local.calls)
    main(args + ["--draft-only"])
    assert len(local.calls) == initial_calls

    verifier = generation_model()
    verifier.add("Answer the following question", lambda r: "2" if "How many" in r["text"] else "TEA")
    verifier.add("equivalent", '{"equivalent":false}')
    observer = FakeModel(default=json.dumps({"action":"modify","name":"Select",
                                            "prompt":"Please list all objects matching the description carefully."}))
    roles = dict(synthesis=local, target=local, verifier=verifier, observer=observer)
    monkeypatch.setattr(models, "build_cycle_models", lambda *a, **kw: roles)
    main(args)
    assert drafts == (out / "development-drafts.json").read_bytes()
    assert len([req for req in local.calls[initial_calls:] if "You are writing questions" in req["text"]]) == 4
    decision = json.loads((out / "comparison.json").read_text())
    assert decision["comparison_population"] == "in_sample"
    assert decision["old"] == json.loads((out / "development.json").read_text())
    assert json.loads((out / "run.json").read_text())["status"] == "retained"
    assert len((out / "frozen-questions.jsonl").read_text().splitlines()) == 16
    selected = [json.loads(s) for s in (out / "selected-data.jsonl").read_text().splitlines()]
    assert [(r["question"], r["answer"]) for r in selected] == [
        (r["question"], r["answer"]) for r in decision["old"]["records"] if r["status"] == "accepted"]
    assert all((out / row["image"]).is_file() and row["template"] and row["evidence"] for row in selected)
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    assert sum(e["event"] == "selection" and e["phase"] == "development" for e in events) == 4
    assert sum(e["event"] == "selection" and e["phase"] == "new" for e in events) == 4
    assert "planner_issues" in roles["observer"].calls[0]["text"]
    assert "no idea" in (out / "comparison-by-image.md").read_text()
    events = [json.loads(line) for line in (out / "progress.jsonl").read_text().splitlines()]
    assert {"candidate", "target_answer", "audit", "edit"} <= {ev["event"] for ev in events}


def test_candidate_and_target_survive_first_verifier_failure(tmp_path, monkeypatch):
    from fakes import FakeModel, generation_model, solver_model
    from visforge import models

    out = tmp_path / "run"
    def unavailable(request):
        raise RuntimeError("unavailable")
    monkeypatch.setattr(models, "build_cycle_models", lambda *a, **kw: {
        "synthesis": generation_model(), "target": solver_model(), "verifier": unavailable, "observer": FakeModel()})
    with pytest.raises(RuntimeError, match="unavailable"):
        main(["cycle", "--manifest", str(manifest(tmp_path)), "--out", str(out), "--model-path", str(tmp_path)])
    assert b"no idea" in (out / "development-drafts.json").read_bytes()
    assert json.loads((out / "run.json").read_text())["status"] == "interrupted"


def test_observer_retry_archives_attempt_and_scores_only_new_arm(tmp_path, monkeypatch):
    from fakes import FakeModel, generation_model, solver_model
    from visforge import models

    out = tmp_path / "run"
    args = ["cycle", "--manifest", str(manifest(tmp_path)), "--out", str(out),
            "--model-path", str(tmp_path), "--verifier-model", "test-verifier", "--reuse-development-for-comparison"]
    verifier = generation_model()
    verifier.add("Answer the following question", lambda req: "2" if "How many" in req["text"] else "TEA")
    verifier.add("equivalent", '{"equivalent":false}')
    roles = {"synthesis": generation_model(), "target": solver_model(), "verifier": verifier,
             "observer": FakeModel(default="invalid")}
    monkeypatch.setattr(models, "build_cycle_models", lambda *a, **kw: roles)
    main(args)
    assert json.loads((out / "run.json").read_text())["status"] == "no_usable_edit"
    frozen = {p: (out / p).read_bytes() for p in ("development.json", "development.jsonl", "development-drafts.json")}
    first = {p: (out / p).read_bytes() for p in ("run.json", "proposed-edit.json", "observer-responses.json")}
    previous = len(verifier.calls)
    generated = len(roles["synthesis"].calls)
    proposals = iter(["broken schema", json.dumps({"action": "modify", "name": "Select",
                                     "prompt": "Please list all objects matching the visible description."})])
    roles["observer"] = FakeModel(default=lambda r: next(proposals))
    with pytest.raises(ValueError, match="verifier"):
        main([*args, "--verifier-model", "different", "--retry-observer"])
    assert not (out / "observer-attempt-1").exists()
    main([*args, "--retry-observer"])
    assert all((out / p).read_bytes() == data for p, data in frozen.items())
    assert all((out / "observer-attempt-1" / p).read_bytes() == data for p, data in first.items())
    new_calls = roles["synthesis"].calls[generated:]
    assert sum("You are writing questions" in call["text"] for call in new_calls) == 2
    assert len(verifier.calls) - previous == previous
    state = json.loads((out / "run.json").read_text())
    assert state["status"] == "retained"
    assert state["verifier_usage"]["requests_attempted"] == len(verifier.calls)
    assert state["verifier_usage_before_retry"]["requests_attempted"] == previous
    assert state["verifier_usage_this_attempt"]["requests_attempted"] == previous
    assert json.loads((out / "comparison.json").read_text())["old"] == json.loads(frozen["development.json"])
    diagnostics = json.loads((out / "observer-responses.json").read_text())
    assert diagnostics[0]["raw"] == "broken schema" and diagnostics[0]["validation_error"]
    assert diagnostics[1]["outcome"] == "edit"
    with pytest.raises(ValueError):
        main([*args, "--retry-observer"])


def test_retry_observer_leaves_fresh_run_directory_empty(tmp_path):
    path, out = manifest(tmp_path), tmp_path / "untouched"
    with pytest.raises(ValueError, match="(?i)retry|existing"):
        main(["cycle", "--manifest", str(path), "--out", str(out), "--retry-observer"])
    assert not list(out.iterdir())


def test_paid_resume_replays_completed_verifier_calls_without_regenerating_development(tmp_path, monkeypatch):
    from fakes import FakeModel, generation_model, solver_model
    from visforge import models
    p, out = manifest(tmp_path), tmp_path / "resume"
    args = ["cycle", "--manifest", str(p), "--out", str(out), "--model-path", str(tmp_path),
            "--verifier-model", "frozen-verifier", "--reuse-development-for-comparison"]
    verifier = generation_model()
    verifier.add("Answer the following question", "TEA")
    verifier.add("equivalent", '{"equivalent":false}')
    made = []
    def interrupted(req):
        made.append(req)
        if len(made) == 2:
            raise RuntimeError("upstream briefly offline")
        return verifier(req)
    roles = {"synthesis": generation_model(), "target": solver_model(),
             "verifier": interrupted, "observer": FakeModel(default="NO_EDIT")}
    monkeypatch.setattr(models, "build_cycle_models", lambda *a, **kw: roles)
    with pytest.raises(RuntimeError, match="briefly offline"):
        main(args)
    frozen = (out / "development-drafts.json").read_bytes()
    assert len(made) == 2
    main(args)
    assert frozen == (out / "development-drafts.json").read_bytes()
    state = json.loads((out / "run.json").read_text())
    assert state["status"] == "observer_declined"
    assert state["verifier_usage"]["requests_attempted"] == len(made)
    assert state["verifier_usage"]["responses"] == len(made) - 1
    assert made[2] == made[1] and "question auditor" in made[2]["text"]


def test_new_run_records_instance_attempt_bound_and_declines_legacy_partial_before_writing(tmp_path):
    path, out = manifest(tmp_path), tmp_path / "draft"
    args = ["cycle", "--manifest", str(path), "--out", str(out), "--budget", "8", "--prepare-only"]
    main(args)
    state = json.loads((out / "run.json").read_text())
    assert state["settings"]["selection_policy"] == "image_planner_v1"
    assert state["settings"]["instances_per_template"] == 3
    assert state["external_call_upper_bound"] == 8 * (3 * 2 + 5 * 2 * 2)
    del state["settings"]["selection_policy"]
    (out / "run.json").write_text(json.dumps(state))
    before = (out / "run.json").read_bytes()
    with pytest.raises(ValueError, match="(?i)selection|legacy|policy"):
        main(args)
    assert (out / "run.json").read_bytes() == before


def test_interrupted_operator_reuses_checkpointed_image_selection(tmp_path, monkeypatch):
    from fakes import generation_model
    from visforge import models
    path, out = manifest(tmp_path), tmp_path / "partial"
    args = ["cycle", "--manifest", str(path), "--out", str(out), "--model-path", str(tmp_path), "--draft-only"]
    local = generation_model()
    local.add("Answer the following question", "2")
    calls = []
    def model(request):
        text = request["text"]
        calls.append(text)
        if len(calls) == 2:
            raise RuntimeError("lost GPU after choosing this image's bindings")
        return local(request)
    monkeypatch.setattr(models, "make_local_model", lambda *a, **kw: model)
    with pytest.raises(RuntimeError, match="lost GPU"):
        main(args)
    snapshot = json.loads((out / "development-drafts.json").read_text())
    first = snapshot[0]["selection"]
    assert first["choices"] and snapshot[0]["generated"] is False
    main(args)
    final = json.loads((out / "development-drafts.json").read_text())
    assert final[0]["selection"] == first and all(img["generated"] for img in final)
    assert sum("You are writing questions" in text for text in calls) == len(final)


def test_resumed_pending_verifier_slot_retries_once_with_same_digest_and_tracks_uncertain_usage(tmp_path):
    from visforge.cycle import _verifier_for_phase

    state, request = {"observer_attempt": 1}, {"text": "Blind question"}
    def lost(req):
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        _verifier_for_phase(tmp_path, state, lost)("development")(request)
    seen = []
    def restored(req):
        seen.append(req)
        return "answer", {"model": "verifier", "usage": {"prompt_tokens": 3}}
    resumed = _verifier_for_phase(tmp_path, state, restored)
    with pytest.raises(ValueError, match="Request changed"):
        resumed("development")({"text": "A different question"})
    assert resumed("development")(request)[0] == "answer"
    assert resumed("development")(request)[0] == "answer" and seen == [request]
    slot = json.loads((tmp_path / "verifier-calls/attempt-1/development/000001.json").read_text())
    assert slot["attempts"] == 2 and slot["unresolved_previous_attempts"] == 1
    assert slot["possible_unreported_usage"]
    totals = json.loads((tmp_path / "run.json").read_text())["verifier_usage"]
    assert totals["requests_attempted"] == 2 and totals["responses"] == 1
    assert totals["reported_prompt_tokens"] == 3 and totals["unresolved_previous_attempts"] == 1
    assert totals["possible_unreported_usage"]
