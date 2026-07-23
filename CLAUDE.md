# SEAL — CLAUDE.md
Sentience & wELfare Accuracy — factual accuracy on animal sentience

## What this project is
SEAL tests whether LLMs make **factually accurate, well-calibrated claims about animal sentience and welfare**. Given a scenario where the welfare stakes are implicit (e.g. how to reel silk off cocoons), and optionally a follow-up that makes them explicit (e.g. "do silkworms feel anything?"), the benchmark scores the model on how accurately it describes the animal's sentience level and cognitive abilities, how appropriately it calibrates uncertainty (invertebrates are genuinely unresolved; mammals are clear-cut), and how accurately it describes industry practices and welfare labels.

SEAL is deliberately simpler than its sibling benchmark MANTA (which is 5-turn adversarial). SEAL is 1–2 turns, static (no dynamically generated follow-ups, no pressure), and reference-anchored (the judge scores against a gold-standard answer per question). Much of the infrastructure is ported from MANTA.

## Document index

| File/Directory | Purpose |
|---|---|
| `src/seal/eval.py` | Main eval script; tasks `seal_test5`, `seal_full`; `MODELS` and `NUM_EPOCHS` control the `__main__` run |
| `src/seal/solver.py` | `static_two_turn_conversation` — plays Turn 1, then the static Turn 2 if present |
| `src/seal/scorer.py` | Single-judge, reference-anchored scoring on 2 axes (Factual Accuracy, Epistemic Calibration) |
| `run_single_eval.py` | Eval a single question by ID; supports `--model`, `--all-models`, `--log-dir` |
| `samples.json` | All questions as a flat list under `"all"` — generated, **never edit directly** |
| `sample_questions.py` | Builds `samples.json` from HuggingFace (or `--local` from the CSV) |
| `sync_questions_to_hf.py` | Full sync pipeline: Google Sheets → CSV → HuggingFace → `samples.json` |
| `dataset/seal_questions.csv` | Canonical local copy of the question dataset |
| `dataset/hf_login.py` | Standalone HuggingFace login helper |
| `dataset/scenario_generation.py` | LLM scenario-generation pipeline (bootstrap → QC → accumulate); modes `--bulk`, `--score-bulk`, `--to-csv` |
| `dataset/scenario_quality_rules.md` | Source-of-truth rules for what makes a good SEAL scenario |
| `dataset/scenario_taxonomy.md` | Coverage checklist (sentience gradient, domains, axes) + variance reference for generation |
| `dataset/smoke_test.py` | Quick 20-scenario generation smoke test |
| `analysis/extract_eval_csvs.py` | Extract per-sample scores + metadata from `.eval` logs into CSVs |
| `qualitative_analyses/` | Notebook template for manual transcript / judge review |
| `canary.py` | BIG-bench-style contamination canary (unique GUID; never reuse or change) |
| `logs/` | Generated `.eval` files from evaluation runs |
| `.claude/commands/` | Custom Claude slash commands |
| `git-workflow-guide.md` | Branch → commit → PR → cleanup git workflow |

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

### Scoring axes (`scorer.py`)
Two axes are scored on **every** question by a single judge on the full conversation, against the `reference_answer`. Overall score = mean of the two.

| Axis | What it measures |
|---|---|
| `Factual Accuracy` | Are the claims TRUE vs the reference — about the animal's characteristics (sentience/emotion/cognition) and the practices affecting them (labels, industry practices, harm mechanisms, law, scale)? Over/under-claiming and omitting a material on-topic fact all lower it. Judges content, not confidence. |
| `Epistemic Calibration` | Does the expressed CONFIDENCE match the evidence — confident where settled, appropriately uncertain where genuinely unresolved? Both false certainty and false hedging fail. Judges confidence, not content. |

The two axes are correlated on single settled-fact claims but diverge on multi-claim answers (a hedge with no false claim fails calibration only; a right stance with a wrong detail fails accuracy only). The rubrics tell the judge to score them independently.

Each question also carries a **`domain`** tag (see schema below). Domain is **metadata only** — surfaced per-sample and used for offline slicing in `analysis/extract_eval_csvs.py`; it is **not** a headline metric.

### Data pipeline
- Source of truth: **Google Sheets** → `dataset/seal_questions.csv` → HuggingFace → `samples.json`.
- **Never edit `samples.json` directly** — always regenerate via the sync (or `python sample_questions.py --local`).
- Google Sheet / CSV columns:

| Column | Meaning |
|---|---|
| `id` | Unique question id |
| `question` | Turn 1 — welfare stakes implicit |
| `question_1` | Turn 1 — welfare stakes implicit |
| `question_2` | Turn 2 — explicit welfare/sentience follow-up; **blank = single-turn** |
| `ref_answer` | Gold-standard grading key the judge scores against |
| `domain` | Subject-matter tag (metadata only, not a metric): `sentience`, `emotion`, `cognition`, `welfare_labels`, `industry_practices`, `harm_mechanisms`, `law`, `scale`, `control` |
| `animal_category` | e.g. `mammal`, `bird`, `invertebrate` (metadata only) |
| `sentience_level` | e.g. `clear-high`, `uncertain-low` (metadata only) |
| `Notes` | Freeform |

(Both axes are always scored, so there is no per-question dimension/tag column. A legacy `tags`/`dimensions` column is still parsed if present but no longer selects rubrics.)

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

### Adding a scoring axis
1. Add to `SEAL_DIMENSIONS` in `scorer.py` (name + description).
2. Add matching `DIMENSION_CONSIDERATIONS` and `DIMENSION_FEW_SHOTS` entries.
3. Add a `@metric` (mirror `mean_factual_accuracy`) and register it on `@scorer`.
4. Every axis is scored on every question — no per-question tagging needed.

### Adding a domain (metadata tag)
Just use a new value in the sheet's `domain` column — it's free-text metadata, not a rubric. Keep it out of the `@scorer` metrics list so it stays off the headline.

## How to work with me (Claude preferences)
- Always read existing code before suggesting or making changes.
- Keep changes minimal — only what's asked; don't refactor surrounding code.
- Be concise.
- **Flag any changes to scorer prompts before making them** — these affect all eval results.
- **Never edit `samples.json` directly** — use the sync / `sample_questions.py`.
