"""文字空間優化器。

這是 SkillOpt 的核心:不更新模型權重,而是由一個 optimizer LLM 讀取一批 agent
軌跡(成功的,尤其是失敗的),對自然語言技能提出一次「**有界**」編輯。接著由
trainer 在驗證資料上把關,只有確實有幫助的候選才會被保留。

此處實作了論文中的穩定機制:
  - 「足夠證據」:同時呈現答對與答錯的軌跡;
  - 「有界文字更新」:要求小而精準的編輯,而非整篇重寫;
  - 「拒絕編輯回饋」:告訴 optimizer 哪些過往編輯被拒絕,以免重複;
  - 「慢更新」:每個優化步驟只做一次編輯,且須通過驗證閘門。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .agent import Trajectory
from .config import Config
from .llm import ChatLLM

# 注意:種子技能與 optimizer 的 system prompt 刻意保留英文,對模型較穩定。
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
    """承載跨步驟的記憶:曾經嘗試但被拒絕的編輯。"""

    rejected_edits: list[str] = field(default_factory=list)

    def remember_rejection(self, skill: str) -> None:
        # 只保留最近少數幾筆,以限制 prompt 大小。
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
    """根據這一批的證據,請 optimizer 模型提出修訂後的技能。"""
    wrong = [t for t in trajectories if not t.correct]
    right = [t for t in trajectories if t.correct]
    # 先列失敗(資訊量最大),再附上部分成功案例。
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
