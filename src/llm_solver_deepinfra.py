"""DeepSeek solver via DeepInfra (deepseek-ai/DeepSeek-V3.2) -- fallback route
after the native DeepSeek API (api.deepseek.com) came back with an
insufficient-balance error. Same DEEPINFRA_API_KEY already used for
embeddings throughout this project."""
from __future__ import annotations

import time
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"  # repo root
API_URL = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3.2"

MAX_RETRIES = 6


def _read_env_var(name: str) -> str:
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in {ENV_PATH}")


class _HttpError(Exception):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:300]}")
        self.status = status
        self.text = text


def call_solver(prompt: str, max_tokens: int, temperature: float = 0.0) -> dict:
    """Returns {"text": str, "prompt_tokens": int, "completion_tokens": int,
    "finish_reason": str, "estimated_cost": float | None}. Prefers DeepInfra's
    own usage.estimated_cost (its authoritative billing figure) when present."""
    api_key = _read_env_var("DEEPINFRA_API_KEY")
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=300,
            )
            if resp.status_code != 200:
                raise _HttpError(resp.status_code, resp.text)
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            return {
                "text": choice["message"].get("content") or "",
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "finish_reason": choice.get("finish_reason", ""),
                "estimated_cost": usage.get("estimated_cost"),
            }
        except (_HttpError, requests.exceptions.RequestException) as e:
            status = getattr(e, "status", None)
            if attempt == MAX_RETRIES - 1 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(30, 2 ** attempt))
