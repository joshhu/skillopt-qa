"""HotpotQA data loading and split construction.

SkillOpt expects each split as a JSON array of task items. For QA an item is::

    {"id": str, "question": str, "context": str, "answers": [str, ...]}

`context` is the concatenation of the candidate paragraphs (distractor setting),
mirroring the SearchQA-style format SkillOpt ships with.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _format_context(context: Any) -> str:
    """Flatten HotpotQA's structured context into plain text paragraphs.

    The HF `distractor` config stores context as {"title": [...],
    "sentences": [[...], ...]}; older dumps use a list of [title, sentences].
    """
    titles: list[str]
    sentences: list[list[str]]
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
    else:  # list of [title, [sent, ...]]
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
    """Download HotpotQA from HuggingFace and write train/val/test items.

    Train and val are sampled from the HF `train` split; test from `validation`
    (HotpotQA's test set has no public answers), so the test set stays unseen
    during optimization.
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


def load_split(split_dir: str | Path, name: str) -> list[dict[str, Any]]:
    path = Path(split_dir) / name / "items.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `skillopt-download --out {split_dir}` first."
        )
    return json.loads(path.read_text())
