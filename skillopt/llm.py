"""輕量的 OpenAI 相容 chat 客戶端。

預設指向本機 vLLM 服務,但可搭配任何 OpenAI 相容的 endpoint(OpenAI、Azure、
Ollama 的 /v1……)。agent 與 optimizer 只依賴 `chat()` 這個小介面,因此測試時
可用假物件替換。
"""

from __future__ import annotations

import os
import time
from typing import Protocol

from openai import OpenAI

from .config import ModelConfig


class ChatLLM(Protocol):
    """agent 與 optimizer 所依賴的最小介面。"""

    def chat(self, system: str, user: str, *, model: str,
             temperature: float, max_tokens: int) -> str: ...


class OpenAICompatLLM:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        api_key = os.environ.get(cfg.api_key_env, "EMPTY")  # vLLM 接受任何非空的 key
        self.client = OpenAI(base_url=cfg.base_url, api_key=api_key, timeout=cfg.timeout)

    def chat(self, system: str, user: str, *, model: str,
             temperature: float, max_tokens: int) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as err:  # noqa: BLE001 - 暫時性的 API/網路錯誤
                last_err = err
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} retries") from last_err
