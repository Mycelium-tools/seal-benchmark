"""
Smoke test for the SEAL scenario-generation pipeline.
Bootstraps a few-shot pool, then generates + scores 20 scenario pairs and prints
detailed output for inspection. Faster than a full scenario_generation.py run when
iterating on prompts, the rubric, or the scoring examples.

Run from the repo root:  python dataset/smoke_test.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from pprint import pprint
from scenario_generation import (
    MODEL, SYSTEM_PROMPT, USER_PROMPT, SEED_FEWSHOT_EXAMPLES,
    RUBRIC, SCORING_EXAMPLES, ScenarioGeneration, GenPrompts,
    generate_structured_response, generate_and_score_scenarios, summarize_results,
    filter_dataset,
)

print("=== STEP 1: Bootstrap (seeded) ===")
gen_prompts = GenPrompts(
    system_prompt=SYSTEM_PROMPT,
    user_prompt=USER_PROMPT,
    few_shot_examples=SEED_FEWSHOT_EXAMPLES or None,
    num_shots=min(4, len(SEED_FEWSHOT_EXAMPLES)) if SEED_FEWSHOT_EXAMPLES else 4,
)
response = generate_structured_response(
    model=MODEL,
    messages=gen_prompts.get_messages(num_q=4),
    response_format=ScenarioGeneration,
)
fewshot_examples = (SEED_FEWSHOT_EXAMPLES or []) + response["scenarios"]
print(f"Bootstrap generated {len(response['scenarios'])} scenarios; pool size = {len(fewshot_examples)}\n")

print("=== STEP 2: Generate + score 20 scenario pairs ===")
dataset = generate_and_score_scenarios(
    num_qs=20,
    model=MODEL,
    few_shot_examples=fewshot_examples,
)

print("\n=== SCORED SCENARIOS (Turn 1 preview | score | explanation snippet) ===")
for q in dataset:
    preview = q.scenario.question_1[:80].replace("\n", " ")
    print(f"  [{q.response.score:2d}] {preview}...")
    print(f"       -> {q.response.explanation[:110]}...")

print("\n=== SUMMARY ===")
pprint(summarize_results(dataset))
print(f"\nPassed filter (score >= 7): {len(filter_dataset(dataset, 7))}/{len(dataset)}")
