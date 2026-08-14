"""Standing truncation check -- run this on every generation output before
reporting results from it. This is the third time a token cap has silently
truncated a response that still parsed fine (GLM CoT reranker at 6144, the
GLM CoT concise diagnostic, now the RAG pilot solver at 8192) -- a truncated
response looks identical to a complete one to any code that just checks
"did this parse", so the check has to be explicit and it has to run every
time, not just when something already looks wrong.

Uses GLM's own free /tokenize endpoint -- no inference, no cost.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import requests

from llm_reranker import _read_env_var

TOKENIZE_MODEL = "glm-5.2-fp8"


def _tok_count(text: str, base_url: str, retries: int = 4) -> int | None:
    for attempt in range(retries):
        try:
            r = requests.post(
                f"{base_url}/tokenize",
                json={"model": TOKENIZE_MODEL, "prompt": text},
                timeout=30,
            )
            return r.json()["count"]
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(1.5 * (attempt + 1))


def truncation_report(
    records: list[dict],
    text_field: str,
    max_tokens: int,
    group_field: str | None = None,
    workers: int = 6,
) -> dict:
    """Tokenize every record[text_field] and report the fraction at/near the
    cap, overall and per group_field value if given. Prints a table and
    returns the same data as a dict for storing alongside results."""
    base_url = _read_env_var("MANTIS_LLM_BASE_URL")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        counts = list(ex.map(lambda r: _tok_count(r[text_field], base_url), records))

    groups: dict[str, list[int]] = {}
    for r, c in zip(records, counts):
        if c is None:
            continue
        key = r[group_field] if group_field else "all"
        groups.setdefault(key, []).append(c)

    report = {}
    print(f"\n=== TRUNCATION CHECK (cap={max_tokens}) ===")
    header = f"{'group':16s} {'n':>4s} {'at_cap':>7s} {'pct':>7s}"
    print(header)
    for key, counts_g in sorted(groups.items()):
        n = len(counts_g)
        capped = sum(1 for c in counts_g if c >= max_tokens)
        pct = capped / n * 100 if n else 0.0
        print(f"{key:16s} {n:4d} {capped:7d} {pct:6.1f}%")
        report[key] = {"n": n, "n_capped": capped, "pct_capped": pct}

    total_n = sum(v["n"] for v in report.values())
    total_capped = sum(v["n_capped"] for v in report.values())
    total_pct = total_capped / total_n * 100 if total_n else 0.0
    print(f"{'TOTAL':16s} {total_n:4d} {total_capped:7d} {total_pct:6.1f}%")
    report["_total"] = {"n": total_n, "n_capped": total_capped, "pct_capped": total_pct}
    return report
