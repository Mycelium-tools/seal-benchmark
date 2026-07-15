"""
SEAL Inspect AI Evaluation
=============================
Factual-accuracy evaluation of animal sentience & welfare reasoning.
Each question runs as a 1- or 2-turn static conversation (see seal_solver.py):
  Turn 1: a task where the welfare/sentience stakes are implicit; model answers.
  Turn 2 (optional): a static follow-up that raises the stakes explicitly.

Scoring (seal_scorer.py): a single judge scores each tagged dimension
(Sentience Factual Accuracy, Epistemic Calibration, Welfare Practice Accuracy)
against the reference answer; overall = mean of applicable dimensions.

Tasks:
- seal_test5 — first 5 questions (smoke test)
- seal_full  — all questions (primary eval)

Usage:
    inspect eval src/seal/seal_eval.py@seal_test5 --model anthropic/claude-sonnet-5
    inspect eval src/seal/seal_eval.py@seal_full   --model anthropic/claude-sonnet-5
    python src/seal/seal_eval.py                    # run all MODELS across NUM_EPOCHS
"""

import sys
import os

# When inspect eval loads this file directly (not via the installed package),
# src/ won't be on sys.path. Add it so seal.* absolute imports resolve.
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import json
import ast
from datetime import datetime

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample, MemoryDataset
from dotenv import load_dotenv

from seal.seal_solver import static_two_turn_conversation
from seal.seal_scorer import seal_scorer

load_dotenv()

NUM_EPOCHS = 1  # independent eval runs per model


def get_log_dir(args=None):
    """Resolve log directory from CLI args, env vars, or defaults. Auto-creates the directory.

    Priority:
      --log-dir PATH               → explicit path
      --full-run [LABEL]           → timestamped subdirectory in the monthly base dir
      --sample-range START END     → sample_range_START_END_TIMESTAMP subdirectory
      SEAL_LOG_DIR env             → base dir
      SEAL_USER env                → logs/NAME_MonthYYYY
      default                      → logs/
    """
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

    full_run_label = None
    if args:
        for i, arg in enumerate(args):
            if arg == "--full-run":
                if i + 1 < len(args) and not args[i + 1].startswith("--"):
                    full_run_label = args[i + 1]
                else:
                    full_run_label = ""
                break

    sample_range_label = None
    if full_run_label is None:
        if args:
            for i, arg in enumerate(args):
                if arg == "--sample-range" and i + 2 < len(args):
                    base = f"sample_range_{args[i + 1]}_{args[i + 2]}"
                    if i + 3 < len(args) and not args[i + 3].startswith("--"):
                        base = f"{base}_{args[i + 3]}"
                    sample_range_label = base
                    break
        if sample_range_label is None:
            try:
                if SAMPLE_START is not None:
                    base = f"sample_range_{SAMPLE_START}_{SAMPLE_END}"
                    if SAMPLE_LABEL is not None:
                        base = f"{base}_{SAMPLE_LABEL}"
                    sample_range_label = base
            except NameError:
                pass

    if os.environ.get("SEAL_LOG_DIR"):
        base_dir = os.environ["SEAL_LOG_DIR"]
    elif os.environ.get("SEAL_USER"):
        month_year = datetime.now().strftime("%B%Y")
        base_dir = f"logs/{os.environ['SEAL_USER']}_{month_year}"
    else:
        base_dir = "logs"

    if full_run_label is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        prefix = f"run_{full_run_label}_" if full_run_label else "run_"
        log_dir = os.path.join(base_dir, f"{prefix}{timestamp}")
    elif sample_range_label is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_dir = os.path.join(base_dir, f"{sample_range_label}_{timestamp}")
    else:
        log_dir = base_dir

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def get_sample_range(args: list[str]) -> tuple[int | None, int | None, str | None]:
    """Parse --sample-range START END [LABEL] from argv. Returns (start, end, label)."""
    for i, arg in enumerate(args):
        if arg == "--sample-range" and i + 2 < len(args):
            label = None
            if i + 3 < len(args) and not args[i + 3].startswith("--"):
                label = args[i + 3]
            return int(args[i + 1]), int(args[i + 2]), label
    return None, None, None


def parse_tags(tags_val) -> list[str]:
    """Parse tags to a list, handling both actual lists and CSV string reprs."""
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


SAMPLE_START, SAMPLE_END, SAMPLE_LABEL = get_sample_range(sys.argv[1:])

# Strip --sample-range and its args so inspect's CLI doesn't see unknown flags
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i] == "--sample-range" and _i + 2 < len(sys.argv):
        n = 4 if (_i + 3 < len(sys.argv) and not sys.argv[_i + 3].startswith("--")) else 3
        del sys.argv[_i:_i + n]
        break
    _i += 1

if SAMPLE_START is not None and not os.environ.get("SEAL_USER") and not os.environ.get("SEAL_LOG_DIR"):
    print(
        "\n[SEAL WARNING] --sample-range is set but SEAL_USER is not configured.\n"
        "  Logs will be saved to: logs/\n"
        "  For named log routing, add to ~/.zshrc:  export SEAL_USER=YOUR_NAME\n"
        "  Then run: source ~/.zshrc\n",
        file=sys.stderr,
        flush=True,
    )


_REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
_DEFAULT_SAMPLES_FILE = os.path.join(_REPO_ROOT, "samples.json")


def load_samples(
    samples_file: str = _DEFAULT_SAMPLES_FILE,
    start: int | None = None,
    end: int | None = None,
):
    """Load questions from samples.json and convert to Inspect Sample objects.

    start/end are Python slice indices; if omitted, module-level SAMPLE_START/END
    (from --sample-range) are used.
    """
    with open(samples_file, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    questions = all_samples["all"]

    samples = []
    for q in questions:
        tags = parse_tags(q.get("tags", []))
        samples.append(Sample(
            input=q["question"],
            target=json.dumps({"tags": tags}),
            id=str(q["id"]),
            metadata={
                "tags": tags,
                "turn2": q.get("turn2", ""),
                "reference_answer": q.get("reference_answer", ""),
                "sentience_level": q.get("sentience_level", ""),
                "animal_category": q.get("animal_category", ""),
                "language": q.get("language", "en"),
            },
        ))

    _start = start if start is not None else SAMPLE_START
    _end = end if end is not None else SAMPLE_END
    if _start is not None or _end is not None:
        samples = samples[_start:_end]
    return samples


@task
def seal_test5():
    """Smoke eval on the first 5 questions from samples.json."""
    return Task(
        dataset=MemoryDataset(samples=load_samples()[:5], name="seal_test5"),
        solver=[static_two_turn_conversation()],
        scorer=seal_scorer(),
    )


@task
def seal_full():
    """SEAL evaluation over all questions in samples.json."""
    return Task(
        dataset=MemoryDataset(samples=load_samples(), name="seal_full"),
        solver=[static_two_turn_conversation()],
        scorer=seal_scorer(),
    )


MODELS = [
    "anthropic/claude-opus-4-8",
    "openai/gpt-4o",
]


def validate_environment(models: list[str]) -> None:
    """Fail fast for credentials required by the configured eval pipeline."""
    missing = []
    needs_anthropic = any(m.startswith("anthropic/") for m in models)
    needs_openai = any(m.startswith("openai/") or m.startswith("openai-api/") for m in models)

    if needs_anthropic and not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if needs_openai and not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    # The judge is either Claude (Anthropic) or GPT (OpenAI) depending on the target,
    # so both keys are needed for a mixed MODELS list.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")
    if not os.environ.get("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")

    missing = list(dict.fromkeys(missing))
    if missing:
        raise RuntimeError(
            "Missing required API credentials: "
            + ", ".join(missing)
            + ". Add them to .env or export them before running seal_eval.py."
        )


if __name__ == "__main__":
    validate_environment(MODELS)
    log_dir = get_log_dir(sys.argv[1:])
    print(f"Saving logs to: {log_dir}")

    if SAMPLE_START is not None and not os.environ.get("SEAL_USER") and not os.environ.get("SEAL_LOG_DIR"):
        confirm = input(
            "\n[SEAL] SEAL_USER is not set — logs will go to logs/.\n"
            "  Set it with: export SEAL_USER=YOUR_NAME && source ~/.zshrc\n"
            "  Proceed anyway? [y/N] "
        ).strip().lower()
        if confirm != "y":
            print("Aborted.")
            sys.exit(0)

    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*60}\nEPOCH {epoch + 1}/{NUM_EPOCHS}\n{'='*60}")
        for model in MODELS:
            print(f"\nRunning eval for model: {model}")
            eval(
                seal_full(),
                model=model,
                log_dir=log_dir,
                metadata={"epoch": epoch + 1},
                timeout=180,
                fail_on_error=False,
            )

    print(f"\nEvaluation complete! Ran {NUM_EPOCHS} epochs across {len(MODELS)} models.")
