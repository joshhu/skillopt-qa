"""The text-space optimizer.

This is the core of SkillOpt: instead of updating model weights, an optimizer
LLM reads a batch of agent trajectories (successes and, especially, failures)
and proposes a *bounded* edit to the natural-language skill. The trainer then
gates the candidate on validation data and keeps it only if it helps.

Stabilizers from the paper that we implement here:
  - "enough evidence": show both correct and incorrect trajectories;
  - "bounded textual updates": ask for small, targeted edits, not rewrites;
  - "rejected-edit feedback": tell the optimizer which past edits were rejected
    so it does not repeat them;
  - "slow update": one edit per optimization step, validation-gated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Trajectory
from .config import Config
from .llm import ChatLLM

SEED_SKILL = (
    "Read the question carefully and identify which paragraphs are relevant. "
    "Combine facts across paragraphs when the question requires multiple hops. "
    "Give the most specific answer supported by the context."
)

_OPTIMIZER_SYSTEM = (
    "You are a prompt optimizer. You improve a natural-language SKILL that is "
    "given to a frozen question-answering agent. You cannot change the model; "
    "you can only edit the skill text.\n\n"
    "Principles:\n"
    "- Make a SMALL, targeted edit grounded in the failures shown. Do not rewrite "
    "everything; keep what already works.\n"
    "- Add concrete, general guidance (how to read context, resolve multi-hop "
    "questions, format answers) — never memorize specific answers or facts.\n"
    "- Keep the skill concise (aim for under 200 words). Prefer imperative bullet "
    "points.\n"
    "- Output ONLY the full revised skill text. No preamble, no explanation, no "
    "code fences."
)


@dataclass
class OptimizerState:
    """Carries cross-step memory: edits that were tried and rejected."""

    rejected_edits: list[str] = field(default_factory=list)

    def remember_rejection(self, skill: str) -> None:
        # Keep only a short tail to bound the prompt size.
        self.rejected_edits.append(skill)
        self.rejected_edits = self.rejected_edits[-3:]


def _render_trajectories(trajectories: list[Trajectory], limit: int) -> str:
    lines = []
    for t in trajectories[:limit]:
        verdict = "CORRECT" if t.correct else "WRONG"
        lines.append(
            f"[{verdict}] Q: {t.question}\n"
            f"  gold: {t.gold}\n"
            f"  agent answered: {t.prediction!r}"
        )
    return "\n".join(lines)


def propose_skill(
    llm: ChatLLM,
    cfg: Config,
    current_skill: str,
    trajectories: list[Trajectory],
    state: OptimizerState,
    evidence_limit: int = 12,
) -> str:
    """Ask the optimizer model for a revised skill given batch evidence."""
    wrong = [t for t in trajectories if not t.correct]
    right = [t for t in trajectories if t.correct]
    # Lead with failures (most informative) but include some successes.
    evidence = _render_trajectories(wrong + right, evidence_limit)

    rejected_block = ""
    if state.rejected_edits:
        joined = "\n---\n".join(state.rejected_edits)
        rejected_block = (
            "\n\nThese previous skill versions were tried and REJECTED because "
            f"they did not improve validation. Do not propose them again:\n{joined}"
        )

    user = (
        f"# Current skill\n{current_skill}\n\n"
        f"# Agent results on a training batch\n{evidence}\n"
        f"{rejected_block}\n\n"
        "Propose an improved skill that would fix the WRONG cases without "
        "breaking the CORRECT ones. Output only the revised skill text."
    )
    revised = llm.chat(
        _OPTIMIZER_SYSTEM,
        user,
        model=cfg.model.optimizer_model,
        temperature=cfg.model.optimizer_temperature,
        max_tokens=cfg.model.optimizer_max_tokens,
    )
    revised = _strip_fences(revised).strip()
    return revised or current_skill


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
    return text
