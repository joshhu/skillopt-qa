"""SQuAD/HotpotQA 風格的答案評分(Exact Match 與 token F1)。

正規化沿用官方 HotpotQA 評估:轉小寫、移除冠詞、去標點、收斂空白。
正規化後完全相符即把該軌跡標記為 `correct`。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from .agent import Trajectory

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(pred: str, golds: list[str]) -> float:
    npred = normalize(pred)
    return float(any(npred == normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    pred_toks = normalize(pred).split()
    for gold in golds:
        gold_toks = normalize(gold).split()
        if not pred_toks or not gold_toks:
            best = max(best, float(pred_toks == gold_toks))
            continue
        common = Counter(pred_toks) & Counter(gold_toks)
        same = sum(common.values())
        if same == 0:
            continue
        precision = same / len(pred_toks)
        recall = same / len(gold_toks)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


@dataclass
class EvalResult:
    em: float
    f1: float
    n: int

    def metric(self, name: str) -> float:
        return {"em": self.em, "f1": self.f1}[name]


def evaluate(trajectories: list[Trajectory]) -> EvalResult:
    """就地評分(設定 `.correct`)並彙總。"""
    if not trajectories:
        return EvalResult(em=0.0, f1=0.0, n=0)
    em_sum = f1_sum = 0.0
    for t in trajectories:
        em = exact_match(t.prediction, t.gold)
        t.correct = em >= 1.0
        em_sum += em
        f1_sum += f1(t.prediction, t.gold)
    n = len(trajectories)
    return EvalResult(em=em_sum / n, f1=f1_sum / n, n=n)
