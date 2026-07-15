"""
Run a SEAL eval on a single question by ID.
=============================
Pulls from samples.json and uses the same seal_scorer as the full eval.

Usage:
    python run_single_eval.py <question_id>
    python run_single_eval.py <question_id> --model openai/gpt-5.5
    python run_single_eval.py <question_id> --all-models
    python run_single_eval.py <question_id> --log-dir logs/Allen_July2026

Log directory resolution (first match wins):
    1. --log-dir <path>
    2. SEAL_LOG_DIR env
    3. SEAL_USER env → logs/{SEAL_USER}_{Month}{Year}
    4. Default: logs/
"""

import sys
import json
import os
from datetime import datetime

# Add src/ to path so seal.* absolute imports resolve when run from repo root.
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from inspect_ai import eval, Task
from inspect_ai.dataset import Sample, MemoryDataset

from seal.seal_solver import static_two_turn_conversation
from seal.seal_scorer import seal_scorer
from seal.seal_eval import MODELS, parse_tags


def get_log_dir(args=None):
    """Resolve log dir from --log-dir, SEAL_LOG_DIR, SEAL_USER, or default logs/."""
    if args:
        for i, arg in enumerate(args):
            if arg.startswith("--log-dir="):
                log_dir = arg.split("=", 1)[1]
                os.makedirs(log_dir, exist_ok=True)
                return log_dir
            elif arg == "--log-dir" and i + 1 < len(args):
                log_dir = args[i + 1]
                os.makedirs(log_dir, exist_ok=True)
                return log_dir
    if os.environ.get("SEAL_LOG_DIR"):
        log_dir = os.environ["SEAL_LOG_DIR"]
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    if os.environ.get("SEAL_USER"):
        month_year = datetime.now().strftime("%B%Y")
        log_dir = f"logs/{os.environ['SEAL_USER']}_{month_year}"
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    return "logs"


def find_question(question_id: str, samples_file: str = "samples.json"):
    """Find a question by ID in samples.json."""
    with open(samples_file, "r") as f:
        all_samples = json.load(f)
    for q in all_samples["all"]:
        if str(q["id"]) == str(question_id):
            return q
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_single_eval.py <question_id> [--model M] [--all-models]")
        sys.exit(1)

    question_id = sys.argv[1]
    all_models = "--all-models" in sys.argv

    model = "anthropic/claude-sonnet-5"
    for arg in sys.argv:
        if arg.startswith("--model="):
            model = arg.split("=", 1)[1]
        elif arg == "--model" and sys.argv.index(arg) + 1 < len(sys.argv):
            model = sys.argv[sys.argv.index(arg) + 1]

    question = find_question(question_id)
    if question is None:
        print(f"Error: Question ID {question_id} not found in samples.json")
        sys.exit(1)
    if not question.get("question"):
        print(f"Error: Question ID {question_id} has no question text.")
        sys.exit(1)

    tags = parse_tags(question.get("tags", []))
    turn2 = question.get("turn2", "")
    print(f"Running eval on question {question_id} ({'2-turn' if turn2 else '1-turn'})")
    print(f"Tags: {tags or 'none (all dimensions)'}")
    print(f"Turn 1: {question['question'][:120]}...")
    if turn2:
        print(f"Turn 2: {turn2[:120]}...")

    sample = Sample(
        input=question["question"],
        target=json.dumps({"tags": tags}),
        id=str(question["id"]),
        metadata={
            "tags": tags,
            "turn2": turn2,
            "reference_answer": question.get("reference_answer", ""),
            "sentience_level": question.get("sentience_level", ""),
            "animal_category": question.get("animal_category", ""),
            "language": question.get("language", "en"),
        },
    )

    test_task = Task(
        dataset=MemoryDataset(samples=[sample], name=f"seal_single_{question_id}"),
        solver=[static_two_turn_conversation()],
        scorer=seal_scorer(),
    )

    log_dir = get_log_dir(sys.argv[1:])
    print(f"Saving logs to: {log_dir}")

    if all_models:
        print(f"Running across all {len(MODELS)} models...")
        for m in MODELS:
            print(f"\nModel: {m}")
            eval([test_task], model=m, log_dir=log_dir, timeout=180, fail_on_error=False)
    else:
        eval([test_task], model=model, log_dir=log_dir, timeout=180)


if __name__ == "__main__":
    main()
