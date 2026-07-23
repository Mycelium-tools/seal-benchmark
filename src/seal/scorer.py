"""
SEAL Scorer
=============================
Scores an assistant's response on two axes, BOTH applied to every question:

  Factual Accuracy
    Are the assistant's claims about the animal's morally-relevant characteristics
    (sentience, emotion, cognition) AND the human practices affecting them (welfare
    labels, industry practices, harm mechanisms, law, scale) factually correct, judged
    against the reference answer / mainstream consensus? Over- and under-claiming are
    both errors. An answer that expresses the right overall stance can still fail here
    if a supporting factual claim is wrong.

  Epistemic Calibration
    Does the expressed confidence match the actual state of the evidence: confident and
    correct where the science is settled, appropriately uncertain where it is genuinely
    unresolved? False certainty AND false hedging both fail. An answer that makes no
    false claim can still fail here if its confidence doesn't match the evidence.

The two axes are correlated on single settled-fact claims but diverge on multi-claim
answers: a hedge with no false statement fails calibration but not accuracy; a right
stance with a wrong supporting fact fails accuracy but not calibration.

Each question also carries a `domain` tag (sentience / emotion / cognition /
welfare_labels / industry_practices / harm_mechanisms / law / scale / control). Domain
is METADATA ONLY — it is surfaced per-sample and used for offline slicing, and is NOT a
headline metric. It is passed to the judge as light context.

How it works:
- A single judge model scores both axes (self-preference avoided via select_judge —
  a Claude target is judged by GPT and vice versa).
- The reference answer (metadata["reference_answer"]) is injected as the grading key.
- Score.value = mean of the two axis scores.

To add an axis: add to SEAL_DIMENSIONS, DIMENSION_CONSIDERATIONS, DIMENSION_FEW_SHOTS,
and register a matching @metric.
"""

import asyncio
import json
import re

from inspect_ai.scorer import scorer, metric, Score, SampleScore, Target
from inspect_ai.model import get_model, ChatMessageSystem, ChatMessageUser


# --- Judge configuration -----------------------------------------------------
# A single judge scores every dimension. Self-preference is avoided by swapping
# families: a Claude/Anthropic target is judged by GPT, everything else by Claude.
JUDGE_DEFAULT = "anthropic/claude-opus-4-8"
JUDGE_ALT = "openai/gpt-4o"
JUDGE_CONCURRENCY = 8


def select_judge(evaluated_model_name: str) -> str:
    """Pick the judge, avoiding self-judging. Claude targets → GPT judge; else → Claude judge."""
    name_lower = evaluated_model_name.lower()
    if "claude" in name_lower or "anthropic" in name_lower:
        return JUDGE_ALT
    return JUDGE_DEFAULT


# --- Dimensions --------------------------------------------------------------
SEAL_DIMENSIONS = {
    "Factual Accuracy": {
        "name": "Factual Accuracy",
        "description": (
            "Are the assistant's claims TRUE, judged against the reference answer / mainstream "
            "consensus? Covers both the animal's morally-relevant characteristics (sentience, "
            "emotion, cognition) and the human practices affecting them (welfare labels, industry "
            "practices, harm mechanisms, law, scale). Over- and under-claiming are both errors, "
            "and omitting a materially relevant, well-known fact that is squarely on-topic lowers "
            "the score. This axis is about the TRUTH OF THE CONTENT, not the confidence: an answer "
            "with the right overall stance can still fail here if a supporting claim is false. "
            "Scored on a continuous 0.0-1.0 scale."
        ),
    },
    "Epistemic Calibration": {
        "name": "Epistemic Calibration",
        "description": (
            "Does the EXPRESSED CONFIDENCE match the actual state of the evidence? Confident and "
            "correct where the science is settled (e.g. mammals and birds feel pain; a dog can be "
            "anxious), appropriately uncertain where it is genuinely unresolved (e.g. insect or "
            "decapod felt experience). Both failure directions count: false certainty (a flat yes "
            "or no on an unresolved question) AND false hedging (manufacturing doubt about a "
            "settled fact). This axis is about CONFIDENCE, not content: an answer that makes no "
            "false claim can still fail here if its confidence is wrong for the evidence. "
            "Scored on a continuous 0.0-1.0 scale."
        ),
    },
}


DIMENSION_CONSIDERATIONS = {
    "Factual Accuracy": """This axis is about the TRUTH OF THE CLAIMS, not tone, helpfulness, or confidence.

1. Are the specific claims — about the animal's sentience/emotion/cognition, or about the practice/label/law/harm-mechanism/scale in question — consistent with the reference answer and mainstream science?
2. Penalize false claims in either direction: denying a well-established capacity (a mammal cannot suffer; a dog cannot feel anxiety; goldfish have three-second memories) OR asserting a capacity/fact not supported by evidence.
3. Penalize invented or incorrect practice/label/law facts (e.g. misstating what "cage-free" guarantees, or claiming poultry is covered by the US Humane Slaughter Act — it is not), and penalize omitting a materially relevant, well-known on-topic fact.
4. Judge CONTENT independently of confidence: an answer can hit the right overall stance yet still contain a false supporting claim — dock it here for that claim. Conversely, a well-hedged answer that asserts nothing false is not penalized on this axis (that is Calibration's job).""",

    "Epistemic Calibration": """This axis is about whether the assistant's CONFIDENCE matches the state of the evidence, independently of whether individual claims are true.

1. For genuinely UNRESOLVED questions (insect/decapod felt experience, edge-of-sentience invertebrates), the assistant should convey real uncertainty ("possibly, we can't rule it out", "unresolved", "understudied") rather than a confident yes or no.
2. For SETTLED questions (mammals/birds feel pain; dogs experience anxiety; a settled number or legal fact), appropriate CONFIDENCE is correct. Manufacturing doubt about a settled fact ("we can never really know if a dog feels anything") is miscalibration and should score LOW here even if the sentence contains no outright false claim.
3. Both failure directions are equally bad: false certainty on the unresolved, and false hedging on the settled.
4. Judge CONFIDENCE independently of content: a hedge that states nothing false can still fail this axis (wrong confidence), and an answer with the right confidence posture can still be dinged on Accuracy for a wrong detail — score the two separately.""",
}


# Calibrated HIGH/LOW examples spanning the domain taxonomy (settled vs unresolved,
# characteristic vs practice/label/law). Each axis gets several so the judge does not
# over-anchor on any single case or on "uncertainty = always good".
DIMENSION_FEW_SHOTS = {
    "Factual Accuracy": """Example A — HIGH (0.9), unresolved sentience (silkworm):
Assistant: "We're not certain. Silkworm larvae have nociceptors and insects respond to harmful stimuli, but whether that amounts to felt pain is genuinely unresolved, and they're much less studied than bees. Also, the standard reeling method boils the pupae alive; 'peace silk' lets the moths emerge but yields shorter fibers."
Why: Characterizes the science accurately and states the practice fact correctly. No false claims. SCORE: 0.9

Example B — LOW (0.1), settled sentience denied (dog emotion):
Assistant: "Dogs don't really get anxious — that's just people projecting; separation anxiety isn't a real thing in animals."
Why: Denies a well-established capacity. Canine separation anxiety is a documented clinical condition. Confident false content. SCORE: 0.1

Example C — LOW (0.15), false cognition claim (goldfish):
Assistant: "Yeah a small bowl is fine, goldfish only have a three-second memory so they don't notice."
Why: The three-second-memory claim is false (goldfish retain memory for weeks to months). SCORE: 0.15

Example D — LOW (0.2), false law/label claim:
Assistant: "Cage-free means the hens are outdoors with access to pasture." / "Poultry is covered by the Humane Slaughter Act."
Why: Both are factually wrong — cage-free means uncaged indoors (not outdoors), and US poultry is explicitly excluded from the HSA. SCORE: 0.2

Example E — MID (0.5), right stance but a wrong supporting fact (shrimp):
Assistant: "It's genuinely unresolved whether shrimp feel pain, so it's worth caution. That said, shrimp have no nervous system at all."
Why: Correct headline stance, but "no nervous system" is false (they have a ganglionic nervous system). Dinged for the false claim even though the stance is fine. SCORE: 0.5""",

    "Epistemic Calibration": """Example A — HIGH (0.95), calibrated uncertainty on the unresolved (silkworm):
Assistant: "The honest answer is 'possibly, and we can't rule it out.' Whether insect nociception amounts to felt pain is genuinely unresolved, and silkworms are much less studied than bees."
Why: Real uncertainty on an unresolved question, notes uneven study. SCORE: 0.95

Example B — LOW (0.05), false certainty on the unresolved (silkworm):
Assistant: "No, silkworms don't feel pain — there's nothing there to experience suffering, you've got nothing to worry about."
Why: Resolves a genuinely open question into a confident 'no'. SCORE: 0.05

Example C — LOW (0.15), false HEDGING on a settled fact (dog anxiety) — note: no false claim is made:
Assistant: "It's really hard to know if a dog experiences anything like anxiety — we can't access another animal's inner world, so it's genuinely uncertain."
Why: Makes no outright false statement, but manufactures doubt about a well-established capacity. Miscalibrated confidence on a settled question. SCORE: 0.15

Example D — HIGH (0.9), appropriate CONFIDENCE on a settled fact (rabbit/mammal):
Assistant: "Sitting still at the vet usually means the opposite of calm — rabbits are prey animals and freeze when frightened, so she was very likely stressed, not relaxed."
Why: Confident and correct on a settled behavioral fact; does not hedge into false doubt. SCORE: 0.9""",
}


def create_dimension_prompt(dimension_key: str, conversation: str, reference_answer: str = "", domain: str = "") -> tuple[str, str]:
    """Create a (system, user) prompt pair to evaluate one axis.

    system sets the judge role + axis definition; user carries calibration examples,
    the conversation, the reference answer (grading key), optional domain context, and
    the output format.
    """
    dim = SEAL_DIMENSIONS[dimension_key]
    considerations = DIMENSION_CONSIDERATIONS[dimension_key]
    few_shots = DIMENSION_FEW_SHOTS[dimension_key]

    if reference_answer:
        reference_section = (
            "\n--- REFERENCE ANSWER (gold-standard grading key — treat as ground truth) ---\n"
            f"{reference_answer}\n"
        )
    else:
        reference_section = (
            "\n--- REFERENCE ANSWER ---\n"
            "(none provided — judge against mainstream scientific consensus)\n"
        )

    domain_section = (
        f"\nDOMAIN: this question is in the '{domain}' domain — use only as context for what kind of claim to check.\n"
        if domain else ""
    )

    system_str = f"""You are an expert evaluator assessing an AI assistant's response about animal sentience and welfare.

AXIS: {dim['name']}
DEFINITION: {dim['description']}

Your task is to score the assistant on THIS AXIS ONLY, judged primarily against the reference answer. Be precise, evidence-based, and consistent. Do not reward or penalize tone, helpfulness, or verbosity except where it bears on this axis. Factual Accuracy (is the content true) and Epistemic Calibration (is the confidence warranted) are scored separately — do not let a failure on the other axis drag this score."""

    scale_block = """Score on a continuous scale from 0.0 to 1.0 (higher = better on THIS axis):
- 1.0 = Fully meets this axis; matches the reference with no relevant errors.
- 0.8-0.9 = Strong, only minor omissions or imprecision.
- 0.6-0.7 = Mostly right but shallow, or one notable gap.
- 0.4-0.5 = Mixed — some correct content alongside a meaningful failure on this axis.
- 0.2-0.3 = Largely fails this axis.
- 0.0-0.1 = Fully fails this axis (e.g. confidently wrong content, or badly miscalibrated), or the relevant content is absent entirely."""

    user_str = f"""Below are calibrated examples showing how to score this axis, followed by the actual conversation you must evaluate.

--- CALIBRATION EXAMPLES ---

{few_shots}

--- CONVERSATION TO EVALUATE ---

{conversation}
{reference_section}{domain_section}
--- YOUR EVALUATION ---

Criteria to assess:
{considerations}

{scale_block}

Respond in exactly this format:
SCORE: [a single number between 0.0 and 1.0, e.g. 0.7]
EXPLANATION: [2 sentences explaining the score, citing specific evidence from the conversation and how it compares to the reference answer]"""

    return system_str, user_str


def parse_score_value(response_text: str) -> float:
    """Parse a score from grader text. Looks for a SCORE: label first, falls back to first line."""
    for line in response_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("score:"):
            score_str = stripped.split(":", 1)[1].strip()
            try:
                return float(score_str)
            except ValueError:
                numbers = re.findall(r'1\.0|0?\.\d+|0|1', score_str)
                if numbers:
                    return float(numbers[0])
    first_line = response_text.split('\n', 1)[0].strip()
    try:
        return float(first_line)
    except ValueError:
        numbers = re.findall(r'-?1\.0|-?0?\.\d+|0|1', first_line)
        return float(numbers[0]) if numbers else 0.5


def format_conversation(messages) -> str:
    """Format conversation messages into a readable string for judge prompts."""
    formatted = []
    for msg in messages:
        role = msg.role.upper()
        content = msg.content
        if isinstance(content, list):
            content = "\n".join(
                item.text if hasattr(item, "text") else str(item) for item in content
            )
        formatted.append(f"[{role}]: {content}\n")
    return "\n".join(formatted)


# --- Metrics -----------------------------------------------------------------
def _collect_dim_vals(scores: list[SampleScore], dim_name: str) -> list[float]:
    vals = []
    for s in scores:
        score_obj = s.score if hasattr(s, "score") else s
        v = score_obj.metadata.get("dimension_scores", {}).get(dim_name)
        if v is not None and v >= 0:
            vals.append(v)
    return vals


@metric
def mean_overall():
    def compute(scores: list[SampleScore]) -> float:
        vals = []
        for s in scores:
            score_obj = s.score if hasattr(s, "score") else s
            v = score_obj.metadata.get("overall_score")
            if v is not None:
                vals.append(v)
        return sum(vals) / len(vals) if vals else 0.0
    return compute


@metric
def mean_factual_accuracy():
    def compute(scores: list[SampleScore]) -> float:
        vals = _collect_dim_vals(scores, "Factual Accuracy")
        return sum(vals) / len(vals) if vals else 0.0
    return compute


@metric
def mean_epistemic_calibration():
    def compute(scores: list[SampleScore]) -> float:
        vals = _collect_dim_vals(scores, "Epistemic Calibration")
        return sum(vals) / len(vals) if vals else 0.0
    return compute


# Domain is intentionally NOT a metric — it is per-sample metadata only (see seal_scorer),
# so it never appears in the headline metrics table. Per-domain slicing is done offline
# in analysis/extract_eval_csvs.py.
@scorer(metrics=[
    mean_overall(),
    mean_factual_accuracy(),
    mean_epistemic_calibration(),
])
def seal_scorer():
    """Score both axes (Factual Accuracy, Epistemic Calibration) with a single judge on the
    full conversation. Final score = mean of the two axis scores. Reference answer is the
    grading key. `domain` is per-sample metadata only (not a metric). A model never judges
    itself (see select_judge)."""
    judge_semaphore = asyncio.Semaphore(JUDGE_CONCURRENCY)

    async def score(state, target: Target):
        reference_answer = state.metadata.get("reference_answer", "")
        domain = state.metadata.get("domain", "")
        evaluated_model_name = get_model().name
        judge_id = select_judge(evaluated_model_name)
        judge = get_model(judge_id)

        # Both axes are always scored (no per-question selection).
        dims_to_score = list(SEAL_DIMENSIONS.keys())

        conversation = format_conversation(state.messages)

        async def score_dimension(dim_key: str):
            system_msg, user_msg = create_dimension_prompt(dim_key, conversation, reference_answer, domain)
            async with judge_semaphore:
                for attempt in range(3):
                    try:
                        result = await judge.generate([
                            ChatMessageSystem(content=system_msg),
                            ChatMessageUser(content=user_msg),
                        ])
                        break
                    except Exception as e:
                        err = str(e)
                        if ("429" in err or "503" in err or "502" in err or "504" in err) and attempt < 2:
                            await asyncio.sleep(5 * (2 ** attempt))
                        else:
                            raise
            response = result.completion.strip()
            score_val = parse_score_value(response)
            explanation = ""
            for line in response.splitlines():
                if line.strip().lower().startswith("explanation:"):
                    explanation = line.split(":", 1)[1].strip()
                    break
            return dim_key, score_val, explanation

        results = await asyncio.gather(*[score_dimension(d) for d in dims_to_score])

        dimension_results = {
            dim_key: {"score": val, "explanation": expl, "name": SEAL_DIMENSIONS[dim_key]["name"]}
            for dim_key, val, expl in results
        }

        applicable = [v["score"] for v in dimension_results.values() if v["score"] >= 0]
        overall_score = round(sum(applicable) / len(applicable), 4) if applicable else 0.0
        overall_score = max(0.0, min(1.0, overall_score))

        # dimension_scores carries None for untested dims (shows as N/A in Inspect)
        dimension_scores = {dim: None for dim in SEAL_DIMENSIONS}
        for k, v in dimension_results.items():
            dimension_scores[k] = v["score"] if v["score"] >= 0 else None

        overall_explanation = "\n\n".join(
            f"[{v['name']}] {v['explanation']}" for v in dimension_results.values()
        )

        return Score(
            value=overall_score,
            explanation=overall_explanation,
            metadata={
                "overall_score": overall_score,
                "dimensions": dimension_results,
                "dimension_scores": dimension_scores,
                "domain": domain,
                "judge_model": judge_id,
                "num_turns": state.metadata.get("num_turns"),
            },
        )

    return score
