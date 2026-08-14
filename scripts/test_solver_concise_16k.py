"""Follow-up to test_solver_16k.py: doubling the cap (8192->16384) only cut
truncation on the worst-case subset from 80% to 50% -- several answers hit
the new cap exactly too, suggesting GLM fills whatever budget it's given
rather than running out near a natural stopping point. This tests whether an
explicit concise-solving instruction (same fix class that took the CoT
reranker from 63.3%->10.0% truncation) does what a bigger cap alone couldn't.

Same 20 (condition, query_id) pairs as the plain 16k test, same 16384 cap --
only the prompt changes. Solver-only (GLM, free lab endpoint), no cost.
"""
import json
import sys
import time

import requests

sys.path.insert(0, "src")

from data import MathNetEasy
from llm_reranker import _read_env_var
from truncation_check import truncation_report

SOLVER_MODEL = "glm-5.2-fp8"
SOLVER_BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")
CAP = 16384

SOLVE_PROMPT_NONE_CONCISE = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

Be efficient: work toward the answer directly, state each key step once, and do not repeatedly re-derive, backtrack over, or restate work you have already done. Once you reach a complete, verified solution, stop -- do not continue exploring alternative approaches "just to check."

PROBLEM:
{problem}"""

SOLVE_PROMPT_WITH_CONTEXT_CONCISE = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

Here is a related problem that may (or may not) be helpful context -- use it only if it actually helps; if it isn't relevant, ignore it and solve directly.

Be efficient: work toward the answer directly, state each key step once, and do not repeatedly re-derive, backtrack over, or restate work you have already done. Once you reach a complete, verified solution, stop -- do not continue exploring alternative approaches "just to check."

RELATED PROBLEM:
{related}

PROBLEM TO SOLVE:
{problem}"""


def call_solver(prompt: str) -> str:
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{SOLVER_BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={
                    "model": SOLVER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": CAP,
                },
                timeout=300,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or ""
            return content
        except requests.exceptions.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


ds = MathNetEasy.load(tier="hard")
query_texts = ds.queries

prev_original = {}
with open("rag_pilot_cache.jsonl") as f:
    for line in f:
        d = json.loads(line)
        prev_original[(d["condition"], d["query_id"])] = d

same_20 = json.load(open("/tmp/solver_16k_test.json"))
jobs = []
for r in same_20:
    cond, qid = r["condition"], r["query_id"]
    cid = prev_original[(cond, qid)]["retrieved_id"]
    jobs.append((cond, qid, cid))

print(f"[test] running {len(jobs)} solver calls at {CAP}-token cap, concise-solving prompt (free)...", flush=True)
records = []
t0 = time.time()
for i, (cond, qid, cid) in enumerate(jobs):
    if cid is None:
        prompt = SOLVE_PROMPT_NONE_CONCISE.format(problem=query_texts[qid])
    else:
        prompt = SOLVE_PROMPT_WITH_CONTEXT_CONCISE.format(related=ds.corpus[cid], problem=query_texts[qid])
    solution = call_solver(prompt)
    records.append({"condition": cond, "query_id": qid, "solution": solution})
    print(f"[test] {i+1}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)

json.dump(records, open("/tmp/solver_concise_16k_test.json", "w"), indent=2)
report = truncation_report(records, "solution", CAP, group_field="condition")
json.dump(report, open("results/solver_concise_16k_pretest.json", "w"), indent=2)
