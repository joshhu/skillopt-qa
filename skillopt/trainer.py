"""The SkillOpt training loop.

Treats skill learning like model training: epochs over the train set, mini-batch
rollouts, a candidate edit per step, and a validation gate that accepts the edit
only if the held-out metric improves. The single surviving artifact is
`best_skill.md` — deployable and model-agnostic.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

from . import agent, evaluator, optimizer
from .config import Config
from .llm import ChatLLM


def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


class Trainer:
    def __init__(self, cfg: Config, llm: ChatLLM, out_dir: str | Path):
        self.cfg = cfg
        self.llm = llm
        self.out = Path(out_dir)
        (self.out / "skills").mkdir(parents=True, exist_ok=True)
        (self.out / "steps").mkdir(parents=True, exist_ok=True)
        self.history: list[dict] = []
        self.version = 0

    # -- evaluation helpers -------------------------------------------------
    def _val_metric(self, skill: str, val_items: list) -> evaluator.EvalResult:
        trajs = agent.run_batch(self.llm, self.cfg, skill, val_items)
        return evaluator.evaluate(trajs)

    def _save_skill(self, skill: str, tag: str) -> None:
        self.version += 1
        (self.out / "skills" / f"skill_v{self.version:04d}_{tag}.md").write_text(skill)

    # -- main loop ----------------------------------------------------------
    def fit(self, train_items: list, val_items: list) -> str:
        cfg = self.cfg
        rng = random.Random(cfg.train.seed)
        metric_name = cfg.metric

        if cfg.train.val_size and cfg.train.val_size < len(val_items):
            val_items = val_items[:cfg.train.val_size]

        skill = cfg.seed_skill.strip() or optimizer.SEED_SKILL
        state = optimizer.OptimizerState()

        base = self._val_metric(skill, val_items)
        best_metric = base.metric(metric_name)
        best_skill = skill
        self._save_skill(skill, "seed")
        print(f"[baseline] val {metric_name}={best_metric:.4f} "
              f"(em={base.em:.4f} f1={base.f1:.4f}) on {base.n} items")

        no_improve = 0
        step = 0
        for epoch in range(cfg.train.num_epochs):
            order = list(train_items)
            rng.shuffle(order)
            for batch in _batches(order, cfg.train.batch_size):
                step += 1
                # 1. roll out current skill on the train batch
                trajs = agent.run_batch(self.llm, self.cfg, best_skill, batch)
                train_eval = evaluator.evaluate(trajs)

                # 2. propose a bounded edit from the trajectory evidence
                candidate = optimizer.propose_skill(
                    self.llm, self.cfg, best_skill, trajs, state)

                # 3. validation gate
                cand_eval = self._val_metric(candidate, val_items)
                cand_metric = cand_eval.metric(metric_name)
                improved = cand_metric > best_metric + cfg.train.min_improvement

                tag = "accept" if improved else "reject"
                self._save_skill(candidate, tag)
                record = {
                    "step": step,
                    "epoch": epoch,
                    "train_em": train_eval.em,
                    "train_f1": train_eval.f1,
                    "val_metric_before": best_metric,
                    "val_metric_candidate": cand_metric,
                    "val_em_candidate": cand_eval.em,
                    "val_f1_candidate": cand_eval.f1,
                    "accepted": improved,
                }
                self.history.append(record)
                (self.out / "steps" / f"step_{step:04d}.json").write_text(
                    json.dumps({**record, "candidate_skill": candidate}, indent=2))

                if improved:
                    best_metric, best_skill = cand_metric, candidate
                    no_improve = 0
                    state = optimizer.OptimizerState()  # clear rejections on progress
                    print(f"[step {step}] ACCEPT  val {metric_name}={best_metric:.4f} "
                          f"(train_f1={train_eval.f1:.3f})")
                else:
                    no_improve += 1
                    state.remember_rejection(candidate)
                    print(f"[step {step}] reject  cand {metric_name}={cand_metric:.4f} "
                          f"<= best {best_metric:.4f}  ({no_improve}/{cfg.train.patience})")

                self._flush_outputs(best_skill)
                if no_improve >= cfg.train.patience:
                    print(f"[early stop] {no_improve} consecutive rejects")
                    self._flush_outputs(best_skill)
                    return best_skill

        self._flush_outputs(best_skill)
        print(f"[done] best val {metric_name}={best_metric:.4f}")
        return best_skill

    def _flush_outputs(self, best_skill: str) -> None:
        (self.out / "best_skill.md").write_text(best_skill)
        (self.out / "history.json").write_text(json.dumps(self.history, indent=2))


def evaluate_skill(cfg: Config, llm: ChatLLM, skill: str,
                   items: list) -> evaluator.EvalResult:
    """Score a fixed skill on a (test) set — used after training."""
    trajs = agent.run_batch(llm, cfg, skill, items)
    return evaluator.evaluate(trajs)
