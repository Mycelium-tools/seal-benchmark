"""
Sample Questions
=============================
Builds samples.json from the SEAL question dataset. By default loads the
published HuggingFace dataset; pass --local to build directly from the local
CSV (dataset/seal_questions.csv) — useful before the HF dataset exists.

Parses tags from the CSV string repr to lists and normalizes turn2 /
reference_answer to plain strings. Outputs all questions as a flat list under
the "all" key.

Called automatically by sync_questions_to_hf.py after each sync.
Output: samples.json

Usage:
    python sample_questions.py            # from HuggingFace
    python sample_questions.py --local    # from local CSV
"""

import ast
import csv
import json
import sys

from canary import CANARY

LOCAL_CSV = "dataset/seal_questions.csv"
HF_DATASET = "mycelium-ai/seal-benchmark-questions"
HF_CSV = "seal_questions.csv"


def parse_tags(tags_val) -> list[str]:
    """Parse tags from CSV string repr (e.g. "['Sentience Factual Accuracy']") to a list."""
    if not tags_val:
        return []
    if isinstance(tags_val, list):
        return tags_val
    if not isinstance(tags_val, str):
        return []
    try:
        result = ast.literal_eval(tags_val)
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []


def parse_dimensions(val) -> list[str]:
    """Parse the dimensions/tags column: accepts a Python-list repr
    (e.g. "['Epistemic Calibration']") or a plain comma-separated string."""
    if not val:
        return []
    if isinstance(val, list):
        return val
    parsed = parse_tags(val)
    if parsed:
        return parsed
    s = str(val).strip()
    return [p.strip() for p in s.split(",") if p.strip()]


def normalize_row(row: dict) -> dict:
    """Normalize one raw CSV/HF row into a clean question dict.

    Maps the Google Sheet schema (question_1 / question_2 / dimensions) to the
    internal samples.json schema (question / turn2 / tags). Older column names
    (question / turn2 / tags) are still accepted as a fallback.
    """
    return {
        "id": row.get("id"),
        "question": (row.get("question_1") or row.get("question") or "").strip(),
        "turn2": (row.get("question_2") or row.get("turn2") or "").strip(),
        "tags": parse_dimensions(
            row.get("dimensions") if row.get("dimensions") is not None else row.get("tags")
        ),
        "reference_answer": (row.get("ref_answer") or row.get("reference_answer") or "").strip(),
        "sentience_level": (row.get("sentience_level") or "").strip(),
        "animal_category": (row.get("animal_category") or "").strip(),
    }


def load_rows(use_local: bool) -> list[dict]:
    if use_local:
        print(f"Loading SEAL questions from local CSV ({LOCAL_CSV})...")
        with open(LOCAL_CSV, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    print("Loading SEAL questions from HuggingFace...")
    # revision= should be pinned to a specific commit SHA for reproducibility;
    # using "main" here as a minimum — replace with a commit SHA once stable.
    from datasets import load_dataset

    dataset = load_dataset(HF_DATASET, data_files=HF_CSV, revision="main")
    return [dict(r) for r in dataset["train"]]


def main():
    use_local = "--local" in sys.argv[1:]
    rows = load_rows(use_local)

    all_questions = [normalize_row(r) for r in rows]
    print(f"\nTotal questions: {len(all_questions)}")

    samples = {
        "canary": CANARY,
        "all": all_questions,
    }

    with open("samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("Saved samples to samples.json")

    if all_questions:
        print(f"\n{'='*60}")
        print("EXAMPLE QUESTION:")
        print(f"{'='*60}")
        print(all_questions[0]["question"])


if __name__ == "__main__":
    main()
