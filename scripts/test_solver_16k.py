"""Free pre-check before committing to the paid narrowed-pilot rerun: does
raising SOLVER_MAX_TOKENS from 8192 to 16384 actually fix truncation, or does
GLM just use the extra room too? Solver-only (GLM, free lab endpoint) --
no grader calls, no cost. 20 calls stratified across none/dumb/gold, biased
toward queries that were truncated at 8192 last time (the hardest cases,
most informative test).
"""
import json
import sys
import time

import requests

sys.path.insert(0, "src")

from data import MathNetEasy
from llm_reranker import _read_env_var
from truncation_check import truncation_report

SEED = 42
N_PILOT = 250
SOLVER_MODEL = "glm-5.2-fp8"
SOLVER_BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")
NEW_CAP = 16384

SOLVE_PROMPT_NONE = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

PROBLEM:
{problem}"""

SOLVE_PROMPT_WITH_CONTEXT = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

Here is a related problem that may (or may not) be helpful context -- use it only if it actually helps; if it isn't relevant, ignore it and solve directly.

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
                    "max_tokens": NEW_CAP,
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
all_query_ids = ds.sample_queries(500, seed=SEED)
query_ids = all_query_ids[:N_PILOT]
query_texts = {qid: ds.queries[qid] for qid in query_ids}

solve_index = json.load(open("data/solve_solution_index.json"))
graded_query_ids = [qid for qid in query_ids if qid in solve_index and solve_index[qid]["solutions_markdown"]]

# reuse the original pilot's retrieved_id (same corpus/retrieval, only the cap changes)
prev_original = {}
with open("rag_pilot_cache.jsonl") as f:
    for line in f:
        d = json.loads(line)
        prev_original[(d["condition"], d["query_id"])] = d

# pick 7 gold, 7 dumb, 6 none -- prioritize queries that hit the 8192 cap before
def truncated_first(condition: str, source: dict, n: int) -> list[str]:
    candidates = [qid for (c, qid), rec in source.items() if c == condition and qid in graded_query_ids]
    candidates = list(dict.fromkeys(candidates))  # stable dedupe
    # sort: previously-truncated (long solution) first
    candidates.sort(key=lambda qid: -len(source[(condition, qid)]["solution"]))
    return candidates[:n]

gold_qids = truncated_first("gold", prev_original, 7)

# for dumb condition we need the retrieved_id; reuse from prev_original cache directly
dumb_qids_pool = [qid for (c, qid) in prev_original if c == "dumb" and qid in graded_query_ids]
dumb_qids_pool = list(dict.fromkeys(dumb_qids_pool))
dumb_qids_pool.sort(key=lambda qid: -len(prev_original[("dumb", qid)]["solution"]))
dumb_qids = dumb_qids_pool[:7]

none_qids_pool = [qid for (c, qid) in prev_original if c == "none" and qid in graded_query_ids]
none_qids_pool = list(dict.fromkeys(none_qids_pool))
none_qids_pool.sort(key=lambda qid: -len(prev_original[("none", qid)]["solution"]))
none_qids = none_qids_pool[:6]

jobs = []
for qid in gold_qids:
    cid = prev_original[("gold", qid)]["retrieved_id"]
    jobs.append(("gold", qid, cid))
for qid in dumb_qids:
    cid = prev_original[("dumb", qid)]["retrieved_id"]
    jobs.append(("dumb", qid, cid))
for qid in none_qids:
    jobs.append(("none", qid, None))

print(f"[test] running {len(jobs)} solver calls at {NEW_CAP}-token cap (free, GLM lab endpoint)...", flush=True)
records = []
t0 = time.time()
for i, (cond, qid, cid) in enumerate(jobs):
    if cid is None:
        prompt = SOLVE_PROMPT_NONE.format(problem=query_texts[qid])
    else:
        prompt = SOLVE_PROMPT_WITH_CONTEXT.format(related=ds.corpus[cid], problem=query_texts[qid])
    solution = call_solver(prompt)
    records.append({"condition": cond, "query_id": qid, "solution": solution})
    print(f"[test] {i+1}/{len(jobs)} done ({time.time()-t0:.0f}s)", flush=True)

json.dump(records, open("/tmp/solver_16k_test.json", "w"), indent=2)
report = truncation_report(records, "solution", NEW_CAP, group_field="condition")
json.dump(report, open("results/solver_16k_pretest.json", "w"), indent=2)
