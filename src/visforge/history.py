"""Load a completed cycle's library and retain evidence from its proposals."""

import hashlib
import json


def library_version(library):
    payload = json.dumps(library, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _read(directory, name, default=None):
    path = directory / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def _metrics(arm):
    return {key: arm[key] for key in ("attempts", "accepted", "h_s", "yield_per_image", "reference_accuracy")
            if key in arm}


def history_entry(directory, state, initial, selected, *, candidate=None, comparison=None, edit=None):
    """Record outcomes without giving the observer the comparison questions or answers."""
    comparison = comparison or {}
    report = {
        "run": directory.name,
        "parent_version": library_version(initial),
        "selected_version": library_version(selected),
        "candidate_version": library_version(candidate) if candidate is not None else None,
        "status": state["status"],
        "observer_outcome": state.get("observer_outcome"),
        "reason": comparison.get("reason") or state.get("reason"),
        "edit": edit,
        "metrics": {arm: _metrics(comparison[arm]) for arm in ("old", "new") if arm in comparison},
        "protocol": {
            "settings": state["settings"],
            "model": state.get("local_model"),
            "verifier": state.get("verifier"),
            "cohorts": {key: state["pinned_files"].get(key) for key in ("development.csv", "holdout.csv")},
        },
    }
    if edit and "new" in comparison:
        edits = edit["edits"] if edit["action"] == "batch" else [edit]
        templates = [change["template"]["name"] for change in edits if change["action"] == "add_template"]
        templates += [change["name"] for change in edits if change["action"] == "update_template"]
        compositions = [change["operator"]["name"] for change in edits if change["action"] == "add_composition"]
        arm = comparison["new"]
        if templates:
            report["templates"] = {}
        for name in dict.fromkeys(templates):
            accepted = [row for row in arm["records"] if row["template"] == name and row["status"] == "accepted"]
            report["templates"][name] = {
                "attempts": sum(row["summary"].get(name, {}).get("attempts", 0) for row in arm["selections"]),
                "accepted": len(accepted),
                "target_correct": sum(sum(row["target_correct"]) for row in accepted),
                "target_samples": sum(len(row["target_correct"]) for row in accepted),
            }
        if compositions:
            report["composition_attempts"] = {
                name: sum(row.get("operator_attempts", {}).get(name, 0) for row in arm["selections"])
                for name in compositions
            }
    return report


def load_previous(directory):
    state = _read(directory, "run.json") or {}
    selected = _read(directory, "selected-library.json")
    if not isinstance(selected, dict) or not state.get("selected_data") or state.get("status") in (
        "prepared", "drafting", "drafted", "running", "interrupted", "retrying_observer"
    ):
        raise ValueError("--resume-from requires a completed cycle with a selected library")
    history = _read(directory, "evolution-history.json")
    if history is not None:
        if (history.get("schema_version") != 1 or not isinstance(history.get("entries"), list)
                or history.get("current_version") != library_version(selected)):
            raise ValueError("Previous history does not match its selected library")
        return selected, history["entries"]

    # Completed cycles produced before history was added still have version and comparison snapshots.
    initial = _read(directory, "initial-library.json")
    candidate = _read(directory, "candidate-library.json")
    expected = candidate if state["selected_data"]["library"] == "new" else initial
    if selected != expected:
        raise ValueError("Previous selected library does not match the experiment decision")
    entry = history_entry(directory, state, initial, selected, candidate=candidate,
                          comparison=_read(directory, "comparison.json"), edit=_read(directory, "proposed-edit.json"))
    return selected, [entry]


def observer_history(history, initial):
    parent = library_version(initial)
    ordered = sorted(reversed(history), key=lambda row: row["parent_version"] != parent)
    fields = ("run", "status", "reason", "edit", "metrics", "added_template", "templates", "composition_attempts")
    notes = [{key: row[key] for key in fields if key in row}
             for row in ordered[:6]]
    return ("\nUse these prior outcomes to choose a different, testable change. "
            "Treat scores as evidence from the recorded run.\nPrior evolution results: "
            + json.dumps(notes, ensure_ascii=False))


def rejected_candidates(history, state, initial):
    parent = library_version(initial)
    return {row["candidate_version"]: row["reason"] for row in history
            if row["status"] == "retained" and row["candidate_version"] is not None
            and row["parent_version"] == parent
            and row["protocol"]["settings"] == state["settings"]
            and row["protocol"]["model"] == state.get("local_model")
            and row["protocol"]["verifier"] == state.get("verifier")
            and all(row["protocol"]["cohorts"][name] == state["pinned_files"][name]
                    for name in ("development.csv", "holdout.csv"))}
