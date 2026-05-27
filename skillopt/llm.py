"""Thin OpenAI-compatible chat client.

Points at a local vLLM server by default but works with any OpenAI-compatible
endpoint (OpenAI, Azure, Ollama's /v1, ...). The agent and optimizer depend
only on the small `chat()` surface, so tests can substitute a fake object.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

from openai import OpenAI

from .config import ModelConfig


class ChatLLM(Protocol):
    """Minimal interface the agent and optimizer rely on."""

    def chat(self, system: str, user: str, *, model: str,
             temperature: float, max_tokens: int) -> str: ...


class OpenAICompatLLM:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        api_key = os.environ.get(cfg.api_key_env, "EMPTY")  # vLLM accepts any non-empty key
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
            except Exception as err:  # noqa: BLE001 - transient API/network failures
                last_err = err
                time.sleep(min(2.0 * (2 ** attempt), 30.0))
        raise RuntimeError(f"LLM call failed after {self.cfg.max_retries} retries") from last_err
