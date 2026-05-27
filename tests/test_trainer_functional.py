"""End-to-end optimization loop with a fake LLM — no network, fully offline.

The fake agent answers correctly only once the skill contains the phrase
"step by step"; the fake optimizer proposes exactly that. This exercises the
real rollout -> propose -> validation-gate -> accept path in Trainer.fit.
"""

from skillopt.agent import _parse_answer
from skillopt.config import Config
from skillopt.trainer import Trainer, evaluate_skill

ITEMS = [
    {"id": "1", "question": "What is the capital of France?", "context": "...",
     "answers": ["Paris"]},
    {"id": "2", "question": "Who wrote Hamlet?", "context": "...",
     "answers": ["Shakespeare"]},
    {"id": "3", "question": "What color is the sky?", "context": "...",
     "answers": ["blue"]},
    {"id": "4", "question": "How many legs does a spider have?", "context": "...",
     "answers": ["8"]},
]


class FakeLLM:
    def __init__(self):
        self.gold = {it["question"]: it["answers"][0] for it in ITEMS}
        self.agent_calls = 0
        self.optimizer_calls = 0

    def chat(self, system, user, *, model, temperature, max_tokens):
        if "prompt optimizer" in system:
            self.optimizer_calls += 1
            return "Read the context and think step by step, then answer concisely."
        self.agent_calls += 1
        question = user.split("Question:")[1].split("Final answer:")[0].strip()
        gold = self.gold.get(question, "")
        if "step by step" in system.lower():
            return f"Final answer: {gold}"
        return "i don't know"


def _cfg():
    return Config.from_dict({
        "metric": "f1",
        "train": {"num_epochs": 1, "batch_size": 2, "val_size": 0,
                  "workers": 1, "patience": 4, "seed": 0},
    })


def test_parse_answer_handles_tag_and_extra_lines():
    assert _parse_answer("Final answer: Paris") == "Paris"
    assert _parse_answer("blah\nFinal answer: 42\nextra") == "42"
    assert _parse_answer("just text") == "just text"


def test_fit_accepts_improving_edit(tmp_path):
    llm = FakeLLM()
    trainer = Trainer(_cfg(), llm, tmp_path)

    best = trainer.fit(ITEMS, ITEMS)

    assert "step by step" in best.lower()
    assert any(r["accepted"] for r in trainer.history)
    assert (tmp_path / "best_skill.md").read_text() == best
    assert (tmp_path / "history.json").exists()
    # baseline skill produced zero correct answers; final skill is perfect.
    final = evaluate_skill(_cfg(), llm, best, ITEMS)
    assert final.f1 == 1.0


def test_baseline_seed_skill_scores_zero(tmp_path):
    llm = FakeLLM()
    res = evaluate_skill(_cfg(), llm, "be helpful", ITEMS)
    assert res.f1 == 0.0
