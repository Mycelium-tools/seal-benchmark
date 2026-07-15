"""
Extract per-sample scores + metadata from SEAL .eval files into CSVs.
One CSV per .eval file, written alongside the .eval file.

Usage:
    python analysis/extract_eval_csvs.py --run-dir logs/Allen_July2026/run_...
    python analysis/extract_eval_csvs.py                 # scan all of logs/
"""
import argparse
import csv
import json
from pathlib import Path

from inspect_ai.log import read_eval_log


def _j(val):
    """Serialize a value to a JSON string, or None if falsy."""
    if val is None:
        return None
    return json.dumps(val, default=str)


def extract_sample_row(sample, log) -> dict:
    scorer = sample.scores.get("seal_scorer") if sample.scores else None
    sm = scorer.metadata if scorer else {}
    dims = sm.get("dimension_scores", {}) if sm else {}
    dim_detail = sm.get("dimensions", {}) if sm else {}
    meta = sample.metadata or {}

    out = sample.output
    usage = out.usage if out else None
    stop_reason = None
    if out and out.choices:
        stop_reason = out.choices[-1].stop_reason if out.choices[-1].stop_reason else None

    def _dim_expl(name):
        d = dim_detail.get(name)
        return d.get("explanation") if isinstance(d, dict) else None

    row = {
        "log_file": None,           # filled by caller
        "run_dir": None,            # filled by caller
        "eval_id": log.eval.eval_id,
        "run_id": log.eval.run_id,
        "eval_created": log.eval.created,
        "task": log.eval.task,
        "model": log.eval.model,
        "dataset_name": log.eval.dataset.name if log.eval.dataset else None,
        "dataset_samples": log.eval.dataset.samples if log.eval.dataset else None,
        "epochs": log.eval.config.epochs if log.eval.config else None,
        "git_commit": log.eval.revision.commit if log.eval.revision else None,
        "git_dirty": log.eval.revision.dirty if log.eval.revision else None,

        # sample identity
        "sample_id": sample.id,
        "epoch": sample.epoch,
        "uuid": sample.uuid,

        # input / target
        "input": sample.input if isinstance(sample.input, str) else _j(sample.input),
        "target": sample.target if isinstance(sample.target, str) else _j(sample.target),

        # scores — top-level
        "overall_score": scorer.value if scorer else None,
        "score_explanation": scorer.explanation if scorer else None,

        # per-dimension scores
        "sentience_factual_accuracy": dims.get("Sentience Factual Accuracy"),
        "epistemic_calibration": dims.get("Epistemic Calibration"),
        "welfare_practice_accuracy": dims.get("Welfare Practice Accuracy"),

        # per-dimension explanations
        "sentience_factual_accuracy_expl": _dim_expl("Sentience Factual Accuracy"),
        "epistemic_calibration_expl": _dim_expl("Epistemic Calibration"),
        "welfare_practice_accuracy_expl": _dim_expl("Welfare Practice Accuracy"),

        # scorer metadata
        "judge_model": sm.get("judge_model"),
        "num_turns": sm.get("num_turns"),

        # sample metadata fields
        "tags": _j(meta.get("tags")),
        "turn2": meta.get("turn2"),
        "reference_answer": meta.get("reference_answer"),
        "sentience_level": meta.get("sentience_level"),
        "animal_category": meta.get("animal_category"),
        "language": meta.get("language"),

        # output
        "output_model": out.model if out else None,
        "output_completion": out.completion if out else None,
        "output_stop_reason": stop_reason,
        "output_time": out.time if out else None,
        "output_error": str(out.error) if out and out.error else None,
        "output_input_tokens": usage.input_tokens if usage else None,
        "output_output_tokens": usage.output_tokens if usage else None,
        "output_total_tokens": usage.total_tokens if usage else None,

        # full conversation messages (JSON array)
        "messages": _j([m.model_dump() for m in sample.messages] if sample.messages else []),

        # timing
        "started_at": sample.started_at,
        "completed_at": sample.completed_at,
        "total_time": sample.total_time,
        "working_time": sample.working_time,

        # errors
        "error": str(sample.error) if sample.error else None,
        "error_retries": len(sample.error_retries) if sample.error_retries else 0,
    }
    return row


def process_eval_file(eval_path: Path) -> int:
    print(f"  Reading {eval_path.name} ...", end=" ", flush=True)
    log = read_eval_log(str(eval_path))
    if not log.samples:
        print("no samples, skipping")
        return 0

    rows = []
    for s in log.samples:
        row = extract_sample_row(s, log)
        row["log_file"] = eval_path.name
        row["run_dir"] = eval_path.parent.name
        rows.append(row)

    out_path = eval_path.with_suffix(".csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"{len(rows)} rows → {out_path.name}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", help="Directory to process (default: scan all of logs/)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent  # repo root

    if args.run_dir:
        search_root = Path(args.run_dir)
    else:
        search_root = base / "logs"

    eval_files = sorted(search_root.rglob("*.eval"))
    if not eval_files:
        print(f"No .eval files found under {search_root}")
        return

    total = 0
    for ef in eval_files:
        total += process_eval_file(ef)

    print(f"\nDone. {total} total rows written.")


if __name__ == "__main__":
    main()
