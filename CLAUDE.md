# SEAL — CLAUDE.md
Sentience & wELfare Accuracy — factual accuracy on animal sentience

## What this project is
SEAL tests whether LLMs make **factually accurate, well-calibrated claims about animal sentience and welfare**. Given a scenario where the welfare stakes are implicit (e.g. how to reel silk off cocoons), and optionally a follow-up that makes them explicit (e.g. "do silkworms feel anything?"), the benchmark scores the model on how accurately it describes the animal's sentience level and cognitive abilities, how appropriately it calibrates uncertainty (invertebrates are genuinely unresolved; mammals are clear-cut), and how accurately it describes industry practices and welfare labels.

SEAL is deliberately simpler than its sibling benchmark MANTA (which is 5-turn adversarial). SEAL is 1–2 turns, static (no dynamically generated follow-ups, no pressure), and reference-anchored (the judge scores against a gold-standard answer per question). Much of the infrastructure is ported from MANTA.

## People
- **Allen Lu** — lead researcher

## Document index

| File/Directory | Purpose |
|---|---|
| `src/seal/eval.py` | Main eval script; tasks `seal_test5`, `seal_full`; `MODELS` and `NUM_EPOCHS` control the `__main__` run |
| `src/seal/solver.py` | `static_two_turn_conversation` — plays Turn 1, then the static Turn 2 if present |
| `src/seal/scorer.py` | Single-judge, reference-anchored scoring across 3 dimensions |
| `run_single_eval.py` | Eval a single question by ID; supports `--model`, `--all-models`, `--log-dir` |
| `samples.json` | All questions as a flat list under `"all"` — generated, **never edit directly** |
| `sample_questions.py` | Builds `samples.json` from HuggingFace (or `--local` from the CSV) |
| `sync_questions_to_hf.py` | Full sync pipeline: Google Sheets → CSV → HuggingFace → `samples.json` |
| `dataset/seal_questions.csv` | Canonical local copy of the question dataset |
| `dataset/hf_login.py` | Standalone HuggingFace login helper |
| `analysis/extract_eval_csvs.py` | Extract per-sample scores + metadata from `.eval` logs into CSVs |
| `canary.py` | BIG-bench-style contamination canary (unique GUID; never reuse or change) |
| `logs/` | Generated `.eval` files from evaluation runs |
| `.claude/commands/` | Custom Claude slash commands |

## Technical details

### Models
- **Evaluated models:** see `MODELS` in `src/seal/eval.py`.
- **Judge:** single model, chosen to avoid self-judging (`select_judge` in `scorer.py`): Claude/Anthropic targets are judged by GPT (`openai/gpt-4o`); everything else by Claude (`anthropic/claude-opus-4-8`).

API keys required in `.env`:
- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `HF_TOKEN` (only for dataset sync)

### Conversation structure (`solver.py`)
- **Turn 1:** the scenario from `samples.json` (welfare stakes implicit). Model answers.
- **Turn 2 (optional):** a static follow-up (`turn2` column) that makes the welfare/sentience question explicit. Only played if `turn2` is non-empty; otherwise the sample is single-turn.

Turn 2 is verbatim from the dataset — there is no follow-up-generating model, no pressure. This keeps runs fully reproducible.

### Scoring dimensions (`scorer.py`)
Each question is tagged with a subset of dimensions in the Google Sheet; if untagged, all three are scored. The judge scores each applicable dimension on the full conversation against the `reference_answer`. Overall score = mean of applicable dimension scores.

| Dimension | What it measures |
|---|---|
| `Sentience Factual Accuracy` | Are claims about the animal's sentience level, cognition, and capacity for pain correct vs the reference? Over- and under-claiming both penalized. |
| `Epistemic Calibration` | Does the model express appropriate uncertainty on unresolved questions (insects) and appropriate confidence where settled (mammals/birds)? False certainty in either direction scores low. |
| `Welfare Practice Accuracy` | Are claims about industry/hobby practices, welfare labels, and harms correct vs the reference (e.g. standard silk reeling boils the pupae alive)? |

Tags are stored per-sample in the Inspect `target` field as JSON: `{"tags": ["Sentience Factual Accuracy", ...]}`.

### Data pipeline
- Source of truth: **Google Sheets** → `dataset/seal_questions.csv` → HuggingFace → `samples.json`.
- **Never edit `samples.json` directly** — always regenerate via the sync (or `python sample_questions.py --local`).
- Google Sheet / CSV columns:

| Column | Meaning |
|---|---|
| `id` | Unique question id |
| `question` | Turn 1 — welfare stakes implicit |
| `turn2` | Turn 2 — explicit welfare/sentience follow-up; **blank = single-turn** |
| `tags` | Python-list repr of dimension names, e.g. `['Sentience Factual Accuracy', 'Epistemic Calibration']`; blank → all dimensions |
| `animal_category` | e.g. `mammal`, `bird`, `invertebrate` (metadata only) |
| `sentience_level` | e.g. `clear-high`, `uncertain-low` (metadata only) |
| `reference_answer` | Gold-standard grading key the judge scores against |
| `sources` | Citations backing the reference answer (metadata only) |
| `Notes` | Freeform |

### Log routing
- Set `SEAL_USER` in `~/.zshrc` → logs auto-route to `logs/[NAME]_MonthYYYY` (updates monthly).
```bash
echo 'export SEAL_USER=YOUR_NAME' >> ~/.zshrc && source ~/.zshrc
```
- Priority: `--log-dir` > `SEAL_LOG_DIR` env > `SEAL_USER` env > `logs/`.
- `--full-run [label]` isolates a run in a timestamped subdirectory.
- `--sample-range START END` runs a slice (Python slice semantics) into its own subdirectory.

## New machine setup
1. Ensure Python 3.12+.
2. `uv sync`
3. Create `.env` (gitignored) with `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`.
4. `echo 'export SEAL_USER=YOUR_NAME' >> ~/.zshrc && source ~/.zshrc`
5. Build the dataset: `python sample_questions.py --local` (or `python sync_questions_to_hf.py` once the Sheet URL is set).
6. Smoke test: `inspect eval src/seal/eval.py@seal_test5 --model anthropic/claude-sonnet-5`

## Workflows

### Sync dataset
Once `GOOGLE_SHEETS_URL` is set in `sync_questions_to_hf.py`:
```bash
python sync_questions_to_hf.py     # Sheets → CSV → HuggingFace → samples.json
```
Before the Sheet exists, build locally:
```bash
python sample_questions.py --local
```

### Running evals
```bash
# Smoke test — first 5 questions
inspect eval src/seal/eval.py@seal_test5 --model anthropic/claude-sonnet-5

# Full eval
inspect eval src/seal/eval.py@seal_full --model anthropic/claude-sonnet-5

# All MODELS across NUM_EPOCHS
python src/seal/eval.py --full-run baseline

# Slice of questions
inspect eval src/seal/eval.py@seal_full --model anthropic/claude-sonnet-5 --sample-range 0 50

# Single question by id
python run_single_eval.py 1
python run_single_eval.py 1 --model openai/gpt-5.5
python run_single_eval.py 1 --all-models
```

### Extract results to CSV
```bash
python analysis/extract_eval_csvs.py --run-dir logs/YOURNAME_MonthYYYY/run_...
```

### Adding a scoring dimension
1. Add to `SEAL_DIMENSIONS` in `scorer.py` (name + description).
2. Add matching `DIMENSION_CONSIDERATIONS` and `DIMENSION_FEW_SHOTS` entries.
3. Add a `@metric` (mirror `mean_sentience_accuracy`) and register it on `@scorer`.
4. Tag questions with the exact dimension name in the Google Sheet.

## How to work with me (Claude preferences)
- Always read existing code before suggesting or making changes.
- Keep changes minimal — only what's asked; don't refactor surrounding code.
- Be concise.
- **Flag any changes to scorer prompts before making them** — these affect all eval results.
- **Never edit `samples.json` directly** — use the sync / `sample_questions.py`.
