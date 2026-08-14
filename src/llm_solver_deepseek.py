"""DeepSeek native API (api.deepseek.com) solver client -- NOT DeepInfra. The
user has credits on DeepSeek's own platform and wants those used specifically,
not DeepInfra credit, even though DeepInfra also hosts DeepSeek checkpoints.

Native API only serves deepseek-v4-flash and deepseek-v4-pro as of 2026-08-11
(confirmed via GET /models) -- the older "DeepSeek-V3.2" name referenced
earlier in this project was a DeepInfra-hosted checkpoint name and is not
available here.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"  # repo root
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-v4-flash"

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
    "finish_reason": str}. Real usage counts come straight from the API
    response, not estimated."""
    api_key = _read_env_var("DEEPSEEK_API_KEY")
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
            }
        except (_HttpError, requests.exceptions.RequestException) as e:
            status = getattr(e, "status", None)
            if attempt == MAX_RETRIES - 1 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(30, 2 ** attempt))
