"""The frozen QA agent.

The agent is *not* trained. Its only tunable input is the skill text, which is
injected into the system prompt. Given a task item it produces an answer; we
record a lightweight trajectory for the optimizer to learn from.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .config import Config
from .llm import ChatLLM

BASE_INSTRUCTIONS = (
    "You are a question-answering agent. You are given a question and reference "
    "context made of titled paragraphs. Answer using only the context.\n"
    "Output ONLY the final answer on a single line, with no explanation, no "
    "prefix, and no trailing punctuation. For yes/no questions answer 'yes' or 'no'."
)

ANSWER_TAG = "Final answer:"


@dataclass
class Trajectory:
    item_id: str
    question: str
    gold: list[str]
    prediction: str
    raw_response: str
    correct: bool = False  # filled in by the evaluator


def _build_user_prompt(item: dict[str, Any]) -> str:
    return (
        f"Context:\n{item['context']}\n\n"
        f"Question: {item['question']}\n\n"
        f"{ANSWER_TAG}"
    )


def _parse_answer(raw: str) -> str:
    """Extract the answer, tolerating models that add a prefix or extra lines."""
    text = raw.strip()
    if ANSWER_TAG.lower() in text.lower():
        idx = text.lower().rindex(ANSWER_TAG.lower())
        text = text[idx + len(ANSWER_TAG):].strip()
    # Keep the first non-empty line only.
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line
    return text


def run_one(llm: ChatLLM, cfg: Config, skill: str, item: dict[str, Any]) -> Trajectory:
    system = f"{BASE_INSTRUCTIONS}\n\n# Skill\n{skill}".strip()
    raw = llm.chat(
        system,
        _build_user_prompt(item),
        model=cfg.model.target_model,
        temperature=cfg.model.temperature,
        max_tokens=cfg.model.max_tokens,
    )
    return Trajectory(
        item_id=item["id"],
        question=item["question"],
        gold=item["answers"],
        prediction=_parse_answer(raw),
        raw_response=raw,
    )


def run_batch(llm: ChatLLM, cfg: Config, skill: str,
              items: list[dict[str, Any]]) -> list[Trajectory]:
    """Roll out the agent over items in parallel (API calls are IO-bound)."""
    workers = max(1, cfg.train.workers)
    if workers == 1:
        return [run_one(llm, cfg, skill, it) for it in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda it: run_one(llm, cfg, skill, it), items))
