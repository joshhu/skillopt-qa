"""HotpotQA 資料載入與切分建立。

SkillOpt 預期每個切分是一個 JSON 陣列,陣列中每個任務項目格式為::

    {"id": str, "question": str, "context": str, "answers": [str, ...]}

`context` 是候選段落(distractor 設定)串接後的純文字,對應 SkillOpt 內建的
SearchQA 風格格式。
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _format_context(context: Any) -> str:
    """把 HotpotQA 的結構化 context 攤平成純文字段落。

    HF 的 `distractor` 設定把 context 存成 {"title": [...],
    "sentences": [[...], ...]};較舊的 dump 則用 [title, sentences] 的清單。
    """
    titles: list[str]
    sentences: list[list[str]]
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    else:  # [title, [sent, ...]] 形式的清單
        titles = [c[0] for c in context]
        sentences = [c[1] for c in context]

    paragraphs = []
    for title, sents in zip(titles, sentences):
        body = " ".join(s.strip() for s in sents)
        paragraphs.append(f"## {title}\n{body}")
    return "\n\n".join(paragraphs)


def _to_item(row: dict[str, Any], idx: int) -> dict[str, Any]:
    return {
        "id": str(row.get("id", idx)),
        "question": row["question"].strip(),
        "context": _format_context(row["context"]),
        "answers": [str(row["answer"]).strip()],
    }


def build_split(
    out_dir: str | Path,
    n_train: int = 64,
    n_val: int = 64,
    n_test: int = 200,
    seed: int = 0,
    hf_config: str = "distractor",
) -> Path:
    """從 HuggingFace 下載 HotpotQA 並寫出 train/val/test 項目。

    train 與 val 取樣自 HF 的 `train` 切分;test 取自 `validation`
    (HotpotQA 的 test 集沒有公開答案),因此 test 集在優化過程中始終未被看過。
    """
    from datasets import load_dataset

    out_dir = Path(out_dir)
    rng = random.Random(seed)

    train_raw = load_dataset("hotpotqa/hotpot_qa", hf_config, split="train")
    val_raw = load_dataset("hotpotqa/hotpot_qa", hf_config, split="validation")

    train_idx = list(range(len(train_raw)))
    rng.shuffle(train_idx)
    need = n_train + n_val
    if need > len(train_idx):
        raise ValueError(f"Requested {need} items but train split has {len(train_idx)}")
    train_sel, val_sel = train_idx[:n_train], train_idx[n_train:n_train + n_val]

    test_idx = list(range(len(val_raw)))
    rng.shuffle(test_idx)
    test_sel = test_idx[:n_test]

    splits = {
        "train": [_to_item(train_raw[i], i) for i in train_sel],
        "val": [_to_item(train_raw[i], i) for i in val_sel],
        "test": [_to_item(val_raw[i], i) for i in test_sel],
    }
    for name, items in splits.items():
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return out_dir


# --- DRCD(繁體中文閱讀理解)-------------------------------------------------
# DRCD(`voidful/DRCD`)是 SQuAD 式繁體中文資料集,巢狀結構為:
#   row = {title, id, paragraphs: [{context, id,
#            qas: [{id, question, answers: [{text, answer_start}, ...]}]}]}
# 我們攤平成「每個 qa 一筆」item,沿用與 HotpotQA 相同的 item 結構與 `## 標題` 慣例。


def _flatten_drcd(rows: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        title = str(row.get("title", "")).strip()
        for para in row.get("paragraphs", []):
            context = str(para.get("context", "")).strip()
            heading = f"## {title}\n{context}" if title else context
            for qa in para.get("qas", []):
                answers = qa.get("answers", []) or []
                texts: list[str] = []
                for a in answers:
                    t = str(a.get("text", "")).strip()
                    if t and t not in texts:  # 去重、保序
                        texts.append(t)
                if not texts:
                    continue
                items.append({
                    "id": str(qa.get("id", len(items))),
                    "question": str(qa["question"]).strip(),
                    "context": heading,
                    "answers": texts,
                })
    return items


def build_drcd_split(
    out_dir: str | Path,
    n_train: int = 64,
    n_val: int = 64,
    n_test: int = 200,
    seed: int = 0,
) -> Path:
    """從 HuggingFace 下載 DRCD 並寫出 train/val/test 項目(繁體中文)。

    train 與 val 取樣自 HF 的 `train` 切分;test 取自 `test` 切分,因此 test 集在
    優化過程中始終未被看過。
    """
    from datasets import load_dataset

    out_dir = Path(out_dir)
    rng = random.Random(seed)

    train_pool = _flatten_drcd(load_dataset("voidful/DRCD", split="train"))
    test_pool = _flatten_drcd(load_dataset("voidful/DRCD", split="test"))

    rng.shuffle(train_pool)
    need = n_train + n_val
    if need > len(train_pool):
        raise ValueError(f"Requested {need} items but DRCD train has {len(train_pool)}")
    train_sel = train_pool[:n_train]
    val_sel = train_pool[n_train:n_train + n_val]

    rng.shuffle(test_pool)
    test_sel = test_pool[:n_test]

    splits = {"train": train_sel, "val": val_sel, "test": test_sel}
    for name, items in splits.items():
        d = out_dir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "items.json").write_text(json.dumps(items, ensure_ascii=False, indent=2))
    return out_dir


def load_split(split_dir: str | Path, name: str) -> list[dict[str, Any]]:
    path = Path(split_dir) / name / "items.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `skillopt-download --out {split_dir}` first."
        )
    return json.loads(path.read_text())
