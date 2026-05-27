"""凍結的 QA agent。

這個 agent 不會被訓練。它唯一可調整的輸入就是技能文字,該文字會被注入到
system prompt。給定一個任務項目,它會產生答案;我們記錄一份輕量的軌跡供
optimizer 從中學習。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .config import Config
from .llm import ChatLLM

# 注意:給模型的 prompt 字串刻意保留英文,實務上對模型推理較穩定。
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
    correct: bool = False  # 由 evaluator 填入


def _build_user_prompt(item: dict[str, Any]) -> str:
    return (
        f"Context:\n{item['context']}\n\n"
        f"Question: {item['question']}\n\n"
        f"{ANSWER_TAG}"
    )


def _parse_answer(raw: str) -> str:
    """擷取答案,容忍模型加上前綴或多餘行數的情況。"""
    text = raw.strip()
    if ANSWER_TAG.lower() in text.lower():
        idx = text.lower().rindex(ANSWER_TAG.lower())
        text = text[idx + len(ANSWER_TAG):].strip()
    # 只保留第一行非空白內容。
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
    """平行 rollout agent 跑過所有項目(API 呼叫是 IO-bound)。"""
    workers = max(1, cfg.train.workers)
    if workers == 1:
        return [run_one(llm, cfg, skill, it) for it in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda it: run_one(llm, cfg, skill, it), items))
