"""SkillOpt 的設定載入。

一次 run 由單一 YAML 檔加上選擇性的 CLI 覆寫參數決定。設定涵蓋三個面向:
要用哪個模型來擔任 agent/optimizer、訓練超參數(epoch、batch size……),
以及記錄用的雜項。
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    """LLM 在哪裡、以及如何呼叫它。

    `base_url` 預設指向本機 vLLM 的 OpenAI 相容服務。同一個服務可同時擔任
    target(agent)與 optimizer 模型;若你有更強的模型,可把 `optimizer_model`
    設成它。
    """

    base_url: str = "http://localhost:8000/v1"
    api_key_env: str = "OPENAI_API_KEY"  # vLLM 不檢查值,但 SDK 仍要求要有
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
    """把訓練當成優化的超參數(對應論文)。"""

    num_epochs: int = 3
    batch_size: int = 8          # 每次優化步驟所 rollout 的 train 筆數
    val_size: int = 50           # 用來把關的 val 筆數(0 = 全部)
    workers: int = 8             # 平行 rollout 的 worker 數(API 呼叫是 IO-bound)
    patience: int = 4            # 連續被拒絕這麼多次就提早停止
    min_improvement: float = 0.0  # val 指標至少要提升這麼多才接受
    seed: int = 0


@dataclass
class Config:
    benchmark: str = "hotpotqa"
    metric: str = "f1"          # 主要的 val 把關指標:"f1" 或 "em"
    seed_skill: str = ""        # 初始技能文字;留空則使用內建的小型種子技能
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
        """套用扁平的 CLI 覆寫,例如 {"target_model": "...", "num_epochs": 5}。

        每個 key 會自動被導向對應的巢狀區段。
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
