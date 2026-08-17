"""Claude Haiku 4.5 (raw HTTP, native Anthropic API) reranker-judge client --
matches this project's convention (requests-based, no SDK) used by the
other provider clients. Third judge for Task 2.

SHARED, BUDGET-CAPPED FAMILY KEY -- treat every call as real spend.

Confirmed before writing this (claude-api skill, Thinking & Effort table):
Haiku 4.5 is an "older" model for thinking purposes -- omitting `thinking`
runs with NO thinking (unlike current-tier models, where omitting it means
adaptive-on). `output_config.effort` errors on this model tier, so it is
never sent. `temperature` is accepted (unlike current-tier Opus/Sonnet,
where non-default sampling params 400). No beta header needed for a plain
POST /v1/messages call.
"""
from __future__ import annotations

import time
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"  # repo root
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-haiku-4-5"

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


def call_judge(prompt: str, max_tokens: int = 16, temperature: float = 0.0) -> dict:
    """Returns {"text": str, "input_tokens": int, "output_tokens": int,
    "stop_reason": str}. Real usage counts come straight from the API
    response, not estimated. No `thinking` field sent (defaults to off on
    this model tier); no `output_config.effort` (errors on Haiku 4.5)."""
    api_key = _read_env_var("ANTHROPIC_API_KEY")
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                API_URL,
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": MODEL,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            if resp.status_code != 200:
                raise _HttpError(resp.status_code, resp.text)
            data = resp.json()
            text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
            usage = data.get("usage", {})
            return {
                "text": text,
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "stop_reason": data.get("stop_reason", ""),
            }
        except (_HttpError, requests.exceptions.RequestException) as e:
            status = getattr(e, "status", None)
            if attempt == MAX_RETRIES - 1 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(30, 2 ** attempt))
