"""SQuAD/HotpotQA 風格的答案評分(Exact Match 與 token F1)。

正規化沿用官方 HotpotQA 評估:轉小寫、移除冠詞、去標點、收斂空白。
正規化後完全相符即把該軌跡標記為 `correct`。

中文支援:斷詞採 **mixed segmentation**——CJK 字元逐字當作一個 token,Latin
字母/數字連續段保留為整個 token。中文沒有空白,若沿用 `.split()` 整個答案會塌成
單一 token,使 F1 退化成 EM;逐字斷詞讓中文 F1 重新具有部分重疊的意義,同時對
英文行為零影響。標點移除也擴及常見全形/CJK 標點。
"""

from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass

from .agent import Trajectory

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
# 全形/CJK 標點(ASCII 標點仍由 string.punctuation 處理)。
_CJK_PUNCT = "，。、！？；：「」『』（）〔〕【】《》〈〉…—～·．“”‘’"
_PUNCT_TABLE = str.maketrans("", "", string.punctuation + _CJK_PUNCT)
# 比對是否為 CJK 字元(中日韓統一表意文字及常見擴展區)。
_CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
# 連續的 Latin 字母/數字段,或單一 CJK 字元。
_TOKEN_RE = re.compile(r"[a-z0-9]+|[㐀-䶿一-鿿豈-﫿]")


def normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    """mixed segmentation:CJK 逐字、Latin 連續字母/數字為一詞。"""
    return _TOKEN_RE.findall(normalize(text))


def exact_match(pred: str, golds: list[str]) -> float:
    npred = normalize(pred)
    return float(any(npred == normalize(g) for g in golds))


def f1(pred: str, golds: list[str]) -> float:
    best = 0.0
    pred_toks = _tokenize(pred)
    for gold in golds:
        gold_toks = _tokenize(gold)
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
