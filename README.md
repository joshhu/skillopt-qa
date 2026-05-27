# skillopt-qa

A minimal, faithful re-implementation of **[Microsoft SkillOpt](https://github.com/microsoft/SkillOpt)**
for the **HotpotQA** multi-hop question-answering task.

SkillOpt is a *text-space optimizer*: instead of fine-tuning model weights, it
trains a reusable natural-language **skill** (a `best_skill.md` file) for a
**frozen** LLM agent. Skill learning is run like model training — epochs,
mini-batches, and a **validation gate** that only accepts an edit if held-out
accuracy improves. The single deployable artifact at the end is `best_skill.md`,
which transfers across models and harnesses.

> Paper: *SkillOpt: Executive Strategy for Self-Evolving Agent Skills*
> ([arXiv:2605.23904](https://arxiv.org/abs/2605.23904)).
> This repo is an independent, simplified educational implementation, not the
> official code.

## How it works

```
seed skill ──► [roll out agent on a train batch] ──► trajectories (right/wrong)
                          │
                          ▼
              [optimizer LLM proposes a bounded edit]
                          │
                          ▼
              [evaluate candidate on validation set]
                          │
              improved? ──�yes──► accept, becomes new best skill
                          │
                          └─no──► reject, feed back to optimizer (don't repeat)
```

The optimizer follows the paper's stabilizers: *enough evidence* (show
successes and failures), *bounded textual updates* (small targeted edits),
*rejected-edit feedback* (memory of rejected versions), and *slow update*
(one validation-gated edit per step).

## Requirements

- Python ≥ 3.10, [`uv`](https://docs.astral.sh/uv/)
- An **OpenAI-compatible** chat endpoint. This project assumes a local **Qwen**
  server (e.g. vLLM or Ollama) already running at `http://localhost:8000/v1`.
  Nothing GPU-specific is installed by this repo — point it at any endpoint.

## Setup

```bash
cd skillopt-qa
uv venv
uv pip install -e ".[dev]"
cp .env.example .env        # set OPENAI_API_KEY (any value for most local servers)
```

Confirm your server and the model id it reports:

```bash
curl http://localhost:8000/v1/models
```

Set `model.target_model` / `model.optimizer_model` in
`configs/hotpotqa/default.yaml` to **exactly** that id.

## 1. Download the dataset

```bash
uv run skillopt-download --out data/hotpotqa --n-train 64 --n-val 64 --n-test 200
```

Downloads HotpotQA from HuggingFace and writes:

```
data/hotpotqa/
├── train/items.json
├── val/items.json
└── test/items.json
```

Each item: `{"id", "question", "context", "answers": [...]}`. Train/val come from
HotpotQA `train`; test comes from `validation` (its answers are public), so the
test set stays unseen during optimization.

## 2. Train the skill

```bash
uv run skillopt-train \
    --config configs/hotpotqa/default.yaml \
    --split-dir data/hotpotqa \
    --out-root outputs
```

Useful overrides: `--target-model`, `--base-url`, `--num-epochs`, `--batch-size`,
`--val-size`, `--workers`, `--metric {em,f1}`, `--no-test`.

Outputs land in `outputs/<run_name>/`:

```
outputs/<run_name>/
├── best_skill.md          # the deployable artifact
├── history.json           # per-step train/val metrics + accept/reject
├── skills/skill_vXXXX_*.md
├── steps/step_XXXX.json
└── test_result.json       # final EM / F1 on the held-out test set
```

## 3. Deploy

`best_skill.md` is just text. Prepend it to any QA agent's system prompt (see
`skillopt/agent.py` `BASE_INSTRUCTIONS`) — including against a different model.

## Project layout

| Path | Role |
|------|------|
| `skillopt/config.py` | YAML config + CLI overrides |
| `skillopt/llm.py` | OpenAI-compatible chat client |
| `skillopt/data.py` | HotpotQA download & split building |
| `skillopt/agent.py` | frozen QA agent + parallel rollouts |
| `skillopt/evaluator.py` | EM / token-F1 scoring |
| `skillopt/optimizer.py` | the text-space skill optimizer |
| `skillopt/trainer.py` | epoch/batch loop + validation gate |
| `scripts/` | `download_data.py`, `train.py` CLIs |
| `tests/` | unit + offline end-to-end tests (no network) |

## Tests

```bash
uv run pytest            # all tests run offline with a fake LLM
```

## License

MIT
