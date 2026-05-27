"""Configuration loading for SkillOpt.

A run is parameterised by a single YAML file plus optional CLI overrides.
The config covers three concerns: which model serves the agent/optimizer,
the training hyper-parameters (epochs, batch size, ...), and bookkeeping.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """Where the LLM lives and how to call it.

    `base_url` defaults to a local vLLM OpenAI-compatible server. The same
    server can host both the target (agent) and optimizer models; set
    `optimizer_model` to a stronger model if you have one.
    """

    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"  # vLLM ignores the value, but the SDK requires one
    target_model: str = "Qwen/Qwen2.5-7B-Instruct"
    optimizer_model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.0
    optimizer_temperature: float = 0.7
    max_tokens: int = 512
    optimizer_max_tokens: int = 2048
    timeout: float = 120.0
    max_retries: int = 4


@dataclass
class TrainConfig:
    """Training-as-optimization hyper-parameters (cf. the paper)."""

    num_epochs: int = 3
    batch_size: int = 8          # train items rolled out per optimization step
    val_size: int = 50           # how many val items to gate on (0 = all)
    workers: int = 8             # parallel rollout workers (IO-bound API calls)
    patience: int = 4            # stop after this many consecutive rejected steps
    min_improvement: float = 0.0  # val metric must improve by at least this to accept
    seed: int = 0


@dataclass
class Config:
    benchmark: str = "hotpotqa"
    metric: str = "f1"          # primary val gate metric: "f1" or "em"
    seed_skill: str = ""        # initial skill text; empty -> a tiny generic seed
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    @staticmethod
    def from_yaml(path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        return Config.from_dict(raw)

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> "Config":
        raw = dict(raw)
        model = ModelConfig(**(raw.pop("model", {}) or {}))
        train = TrainConfig(**(raw.pop("train", {}) or {}))
        return Config(model=model, train=train, **raw)

    def apply_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Apply flat CLI overrides like {"target_model": "...", "num_epochs": 5}.

        Keys are routed to the matching nested section automatically.
        """
        model_fields = {f.name for f in dataclasses.fields(ModelConfig)}
        train_fields = {f.name for f in dataclasses.fields(TrainConfig)}
        for key, value in overrides.items():
            if value is None:
                continue
            if key in model_fields:
                setattr(self.model, key, value)
            elif key in train_fields:
                setattr(self.train, key, value)
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Unknown config override: {key}")
        return self
