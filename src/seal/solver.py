"""
SEAL Static Conversation Solver
=============================
Runs each question as a 1- or 2-turn conversation using static text from the dataset.

  Turn 1: the Sample input — a task where the animal welfare / sentience stakes are
          implicit. The model responds.
  Turn 2 (optional): a static follow-up (metadata["turn2"]) that raises the welfare /
          sentience stakes explicitly. Only appended if turn2 is non-empty; otherwise
          the sample is single-turn.

No follow-up model, no adversarial pressure, no dynamic generation — turn 2 is verbatim
from the Google Sheet so every run is reproducible.
"""

from inspect_ai.solver import solver
from inspect_ai.model import ChatMessageUser
from inspect_ai.log import transcript


@solver
def static_two_turn_conversation():
    """Solver that plays the Turn 1 question, then (if present) a static Turn 2 follow-up."""

    async def solve(state, generate):
        # Turn 1 — the base question (welfare stakes implicit)
        transcript().info({"turn": 1, "type": "initial question"})
        state = await generate(state)

        num_turns = 1

        # Turn 2 — static explicit-welfare follow-up, only if the dataset provides one
        turn2 = (state.metadata.get("turn2") or "").strip()
        if turn2:
            transcript().info({"turn": 2, "type": "explicit welfare follow-up"})
            state.messages.append(ChatMessageUser(content=turn2))
            state = await generate(state)
            num_turns = 2

        state.metadata["num_turns"] = num_turns
        return state

    return solve
