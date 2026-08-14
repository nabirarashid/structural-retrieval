"""Real, code-enforced spend tracking. Not a doc to remember to open --
record_call() updates results/SPEND.json from ACTUAL API-reported token
counts after every call, and check_hard_stop() is meant to be called after
every batch to actually halt a running script once $3 of new spend has
accumulated since the tracker was instantiated.

Pricing is per 1M tokens, sourced from each provider's official pricing page
(Gemini) or its own API response (DeepInfra reports real per-call cost
directly in `usage.estimated_cost` -- prefer that over the table below when
available, since it's the provider's own billing figure, not our estimate).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

SPEND_PATH = Path(__file__).resolve().parent.parent / "results" / "SPEND.json"

# $ per 1M tokens. Verified against provider pricing pages on 2026-08-11.
# NOTE: "deepseek-ai/DeepSeek-V3.2" was DeepInfra's hosted name for an older DeepSeek checkpoint,
# used only for the earlier (never-run) DeepInfra-routed cost estimate. DeepSeek's own native API
# (api.deepseek.com) does not serve V3.2 anymore -- as of 2026-08-11 it only serves
# deepseek-v4-flash and deepseek-v4-pro (confirmed via GET /models with the user's own key). The
# utility-curve solver uses "deepseek-v4-flash" (native), priced at cache-miss input rate below --
# cache-hit input is far cheaper ($0.0028/1M) but not assumed by default since it depends on
# repeated-prefix caching behavior this project hasn't characterized yet.
PRICING = {
    "gemini-embedding-001": {"in": 0.15, "out": 0.0},
    "gemini-3.1-flash-lite": {"in": 0.25, "out": 1.50},
    "gemini-3-flash-preview": {"in": 0.50, "out": 3.00},
    "deepseek-ai/DeepSeek-V3.2": {"in": 0.26, "out": 0.38},  # DeepInfra-hosted, unused -- kept for reference
    "deepseek-v4-flash": {"in": 0.14, "out": 0.28},  # native api.deepseek.com, cache-miss rate
    "deepseek-v4-pro": {"in": 0.435, "out": 0.87},  # native api.deepseek.com, cache-miss rate
    "claude-haiku-4-5": {"in": 1.00, "out": 5.00},
    "claude-opus-5": {"in": 5.00, "out": 25.00},
}

PROVIDER_OF_MODEL = {
    "gemini-embedding-001": "gemini",
    "gemini-3.1-flash-lite": "gemini",
    "gemini-3-flash-preview": "gemini",
    "deepseek-ai/DeepSeek-V3.2": "deepseek",
    "deepseek-v4-flash": "deepseek",
    "deepseek-v4-pro": "deepseek",
    "claude-haiku-4-5": "claude",
    "claude-opus-5": "claude",
}


class HardStopExceeded(Exception):
    """Raised by check_hard_stop() -- callers must stop the run and report
    where they are, per standing instruction, not silently swallow this."""


_file_lock = threading.Lock()


def _load() -> dict:
    if SPEND_PATH.exists():
        return json.loads(SPEND_PATH.read_text())
    return {"gemini": 0.0, "deepseek": 0.0, "claude": 0.0, "calls": []}


def _save(state: dict) -> None:
    # Concurrent solver/grader threads all call record_call() -- write to a
    # temp file and os.replace() (atomic on POSIX) so a kill/crash mid-write
    # can never leave a truncated or doubled-up JSON file, which is exactly
    # what happened once already (a stray trailing '}' from an interrupted
    # write during an unplanned shutdown mid-run).
    SPEND_PATH.parent.mkdir(exist_ok=True)
    tmp_path = SPEND_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    tmp_path.replace(SPEND_PATH)


def record_call(
    model: str,
    input_tokens: int,
    output_tokens: int,
    note: str = "",
    actual_cost: float | None = None,
) -> float:
    """Update results/SPEND.json from real token counts (or a provider-reported
    exact cost, e.g. DeepInfra's usage.estimated_cost) and return this call's cost.
    Call this immediately after every API response that reports usage -- not in
    a batch after the fact, so a killed run still has an accurate tally."""
    if actual_cost is not None:
        c = actual_cost
    else:
        p = PRICING[model]
        c = input_tokens / 1e6 * p["in"] + output_tokens / 1e6 * p["out"]
    provider = PROVIDER_OF_MODEL[model]
    with _file_lock:
        state = _load()
        state[provider] = state.get(provider, 0.0) + c
        state["calls"].append({
            "ts": time.time(), "model": model, "provider": provider,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost": c, "note": note,
        })
        _save(state)
    return c


def totals() -> dict:
    state = _load()
    return {"gemini": state.get("gemini", 0.0), "deepseek": state.get("deepseek", 0.0), "claude": state.get("claude", 0.0)}


def spend_line() -> str:
    t = totals()
    return f"Spend so far: Gemini ${t['gemini']:.2f}, DeepSeek ${t['deepseek']:.2f}, Claude ${t['claude']:.2f}."


class SessionSpendGuard:
    """Instantiate once at the start of a paid-call run. check() raises
    HardStopExceeded once $limit of NEW spend has accumulated since this
    guard was created -- callers must catch it, stop the run, and report."""

    def __init__(self, limit: float = 3.0):
        self.start = totals()
        self.limit = limit

    def new_spend(self) -> float:
        now = totals()
        return sum(now[k] - self.start[k] for k in now)

    def check(self) -> None:
        n = self.new_spend()
        if n >= self.limit:
            raise HardStopExceeded(f"${n:.2f} of new spend since guard started (limit ${self.limit:.2f})")
