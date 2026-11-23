"""命令行入口：synthesize / evaluate / evolve / compare。薄层，只做参数解析和拼装。"""

import argparse
import json
import sys
from pathlib import Path

from .cycle import parse_mix, run_cycle
from .dataset import export_verified
from .evaluation import h_s, make_judge, sample_answers, score
from .evolution import apply_edit, compare_libraries, propose_edit
from .models import make_model
from .operators import seed_library
from .records import QuestionView
from .sources import load_seeds
from .synthesis import generate
from .verification import answer_blind, audit


def _add_model_args(p):
    p.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL (defaults to VISFORGE_API_BASE)")
    p.add_argument("--api-key", default=None, help="API key (defaults to VISFORGE_API_KEY)")
    p.add_argument("--model", default=None, help="Model name (defaults to VISFORGE_MODEL)")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=None)


def _model_settings(args):
    out = {}
    for key in ("base_url", "api_key", "model", "temperature", "max_tokens"):
        value = getattr(args, key)
        if value is not None:
            out[key] = value
    return out


def _load_library(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def cmd_synthesize(args):
    model = make_model(_model_settings(args))
    library = _load_library(args.library) if args.library else seed_library()
    seeds = load_seeds(args.manifest, "synthesize")
    total, accepted = 0, []
    for seed in seeds:
        report = generate(seed, library, {"model": model, "budget": args.budget,
                                          "instances_per_template": args.instances_per_template})
        total += len(report["candidates"])
        for cand in report["candidates"]:
            blind, _meta = answer_blind(QuestionView(str(seed.path), cand.question), model)
            audit(cand, blind, model)
            if cand.status == "accepted":
                accepted.append(cand)
    result = export_verified(accepted, args.out)
    print(json.dumps({"images": len(seeds), "candidates": total, "accepted": len(accepted), **result}, ensure_ascii=False))
    return 0


def cmd_evaluate(args):
    solver = make_model(_model_settings(args))
    entries = [
        json.loads(line)
        for line in Path(args.data).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    base = Path(args.data).parent
    views = [QuestionView(str((base / e["image"]).resolve()), e["question"]) for e in entries]
    judge = make_judge(solver) if args.judge else None
    records = sample_answers(views, solver, k=args.k)
    correct, detail = [], []
    for rec, entry in zip(records, entries):
        hits = [score(a, str(entry["answer"]), judge) for a in rec["answers"]]
        correct.extend(hits)
        detail.append({"question": entry["question"], "answers": rec["answers"], "correct": hits})
    print(json.dumps({"h_s": h_s(correct), "questions": len(correct), "detail": detail}, ensure_ascii=False, indent=2))
    return 0


def cmd_evolve(args):
    model = make_model(_model_settings(args))
    library = _load_library(args.library)
    records = json.loads(Path(args.records).read_text(encoding="utf-8"))
    edit = propose_edit(library, records, model)
    if edit is None:
        print("observer did not produce a usable edit", file=sys.stderr)
        return 1
    new = apply_edit(library, edit)
    Path(args.out).write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"edit": edit, "out": args.out}, ensure_ascii=False))
    return 0


def cmd_compare(args):
    model = make_model(_model_settings(args))
    library_old, library_new = _load_library(args.old), _load_library(args.new)
    seeds = load_seeds(args.manifest, "holdout")
    settings = {
        "model": model,
        "solver": model,  # 第一版：同一个模型既出题也答题
        "min_yield_per_image": args.floor,
        "k": args.k,
        "instances_per_template": args.instances_per_template,
    }
    decision = compare_libraries(library_old, library_new, seeds, settings, budget=args.budget)
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="visforge", description="Generate questions from images and self-evolve (research skeleton)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("cycle", help="Run one library update with local Qwen and an external visual verifier")
    p.add_argument("--manifest", required=True, help="Seed CSV with path, kind and source group")
    p.add_argument("--out", required=True, help="Empty or previously prepared run directory")
    p.add_argument("--resume-from", help="Start a new cycle from a completed run's selected library and proposal history")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument("--prepare-only", action="store_true", help="Pin and check images without calling a model")
    stage.add_argument("--draft-only", action="store_true", help="Save local development candidates and blind target answers")
    stage.add_argument("--retry-observer", action="store_true",
                       help="Retry an unusable observer response once using the frozen in-sample baseline")
    sample = p.add_mutually_exclusive_group()
    sample.add_argument("--per-kind", type=int, default=1, help="Images per domain in each half")
    sample.add_argument("--mix", type=parse_mix, metavar="KIND=N,...", help="Domain counts for development and comparison")
    population = p.add_mutually_exclusive_group()
    population.add_argument("--holdout-mix", type=parse_mix, metavar="KIND=N,...", help="Separate counts for held-out images")
    population.add_argument("--reuse-development-for-comparison", action="store_true",
                            help="Compare on the same development images and reuse their measured baseline")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--budget", type=int, default=2, help="Maximum grounded question attempts per image and library")
    p.add_argument("--instances-per-template", type=int, default=3, help="Maximum distinct targets per selected template per image")
    p.add_argument("--yield-floor", type=float, default=0.5, help="Minimum accepted questions per held-out image")
    p.add_argument("--reference-floor", type=float, default=0.5, help="Minimum reference-solver accuracy on accepted questions")
    p.add_argument("--model-path", help="Local Qwen checkpoint on a GPU node")
    p.add_argument("--verifier-base-url", help="Defaults to VISFORGE_VERIFIER_API_BASE")
    p.add_argument("--verifier-model", help="Defaults to VISFORGE_VERIFIER_MODEL")
    p.add_argument("--verifier-key-env", default="VISFORGE_VERIFIER_API_KEY",
                   help="Name of the environment variable holding the verifier key")
    p.add_argument("--env-file", default=".env.local", help="Optional private KEY=value file")
    p.set_defaults(func=run_cycle)

    p = sub.add_parser("synthesize", help="Generate questions from each image in the manifest, audit, and export")
    p.add_argument("--manifest", required=True, help="Image manifest (CSV or plain-text path list)")
    p.add_argument("--library", default=None, help="Rule-library JSON (defaults to the built-in seed library)")
    p.add_argument("--out", required=True, help="Dataset output directory")
    p.add_argument("--budget", type=int, default=None, help="Maximum grounded question attempts per image")
    p.add_argument("--instances-per-template", type=int, default=3, help="Maximum distinct targets per selected template per image")
    _add_model_args(p)
    p.set_defaults(func=cmd_synthesize)

    p = sub.add_parser("evaluate", help="Answer the exported data.jsonl blind and score it")
    p.add_argument("--data", required=True, help="data.jsonl produced by the export step")
    p.add_argument("--k", type=int, default=1, help="Answers per question")
    p.add_argument("--judge", action="store_true", help="Use a model to judge equivalence when rule matching fails")
    _add_model_args(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("evolve", help="Let the observer propose an edit from failed-question records and write a new library")
    p.add_argument("--library", required=True, help="Old rule-library JSON")
    p.add_argument("--records", required=True, help="Failed-question records JSON file (a list)")
    p.add_argument("--out", required=True, help="Output path for the new rule-library JSON")
    _add_model_args(p)
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("compare", help="Compare two rule libraries on the same holdout images against the hard gates")
    p.add_argument("--old", required=True, help="Old rule-library JSON")
    p.add_argument("--new", required=True, help="New rule-library JSON")
    p.add_argument("--manifest", required=True, help="Holdout image manifest")
    p.add_argument("--floor", type=float, default=0.0, help="Floor on average accepted questions per image")
    p.add_argument("--budget", type=int, default=None, help="Maximum grounded question attempts per image and library")
    p.add_argument("--instances-per-template", type=int, default=3, help="Maximum distinct targets per selected template per image")
    p.add_argument("--k", type=int, default=1, help="Answers per question")
    _add_model_args(p)
    p.set_defaults(func=cmd_compare)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
