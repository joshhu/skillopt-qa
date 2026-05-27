"""執行 SkillOpt 訓練,接著在 test 集上評估最佳技能。

    uv run skillopt-train --config configs/hotpotqa/default.yaml \
        --split-dir data/hotpotqa --out-root outputs

CLI 旗標會覆寫對應的設定欄位。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from skillopt.config import Config
from skillopt.data import load_split
from skillopt.llm import OpenAICompatLLM
from skillopt.trainer import Trainer, evaluate_skill


def main() -> None:
    p = argparse.ArgumentParser(description="Train a SkillOpt skill on HotpotQA")
    p.add_argument("--config", default="configs/hotpotqa/default.yaml")
    p.add_argument("--split-dir", default="data/hotpotqa")
    p.add_argument("--out-root", default="outputs")
    p.add_argument("--run-name", default=None)
    # 常用覆寫參數
    p.add_argument("--base-url", dest="base_url")
    p.add_argument("--target-model", dest="target_model")
    p.add_argument("--optimizer-model", dest="optimizer_model")
    p.add_argument("--num-epochs", dest="num_epochs", type=int)
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--val-size", dest="val_size", type=int)
    p.add_argument("--workers", dest="workers", type=int)
    p.add_argument("--metric", choices=["em", "f1"])
    p.add_argument("--no-test", action="store_true", help="略過最終的 test 評估")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    cfg.apply_overrides({
        "base_url": args.base_url,
        "target_model": args.target_model,
        "optimizer_model": args.optimizer_model,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "val_size": args.val_size,
        "workers": args.workers,
        "metric": args.metric,
    })

    run_name = args.run_name or f"{cfg.benchmark}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.out_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train_items = load_split(args.split_dir, "train")
    val_items = load_split(args.split_dir, "val")
    print(f"Loaded {len(train_items)} train / {len(val_items)} val items")

    llm = OpenAICompatLLM(cfg.model)
    trainer = Trainer(cfg, llm, out_dir)
    best_skill = trainer.fit(train_items, val_items)

    print("\n===== best_skill.md =====")
    print(best_skill)
    print("=========================\n")

    if not args.no_test:
        test_items = load_split(args.split_dir, "test")
        res = evaluate_skill(cfg, llm, best_skill, test_items)
        print(f"[TEST] em={res.em:.4f} f1={res.f1:.4f} on {res.n} items")
        (out_dir / "test_result.json").write_text(
            json.dumps({"em": res.em, "f1": res.f1, "n": res.n}, indent=2))


if __name__ == "__main__":
    main()
