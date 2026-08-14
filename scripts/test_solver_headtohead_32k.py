"""Head-to-head solver test at a generous, non-arbitrary cap (32,768) --
GLM 5.2 (free lab endpoint) vs DeepSeek-v4-flash (native API, real cost).
Same 10 queries x 3 conditions (none/dumb/gold) for BOTH solvers, so this is
a matched comparison, not two independent samples.

The gate that matters is NOT overall truncation -- it's whether truncation is
correlated with condition (context vs no-context), since that's what would
bias a none/dumb/gold comparison. Equal truncation across conditions is
reportable as a limitation; correlated truncation invalidates the comparison
outright, as the original GLM pilot at 8192 showed (50-53% context vs 38.6%
none).
"""
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, "src")

from data import MathNetEasy
from dumb_reranker import rerank_top10 as dumb_rerank_top10
from embed_labembed import embed_all as embed_fn
from eval import build_results, to_matrix
from llm_reranker import _read_env_var
from llm_solver_deepseek import call_solver as deepseek_call_solver
from spend_tracker import record_call, spend_line

SEED = 42
N_PILOT = 250
CAP = 32768
N_QUERIES = 10

GLM_MODEL = "glm-5.2-fp8"
GLM_BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")

SOLVE_PROMPT_NONE = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

PROBLEM:
{problem}"""

SOLVE_PROMPT_WITH_CONTEXT = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

Here is a related problem that may (or may not) be helpful context -- use it only if it actually helps; if it isn't relevant, ignore it and solve directly.

RELATED PROBLEM:
{related}

PROBLEM TO SOLVE:
{problem}"""


def call_glm(prompt: str) -> dict:
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{GLM_BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={
                    "model": GLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": CAP,
                },
                timeout=600,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            usage = data.get("usage", {})
            return {
                "text": content,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "finish_reason": choice.get("finish_reason", ""),
            }
        except requests.exceptions.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


ds = MathNetEasy.load(tier="hard")
all_query_ids = ds.sample_queries(500, seed=SEED)
query_ids = all_query_ids[:N_PILOT]
corpus_ids = list(ds.corpus.keys())
query_texts = {qid: ds.queries[qid] for qid in query_ids}
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

solve_index = json.load(open("data/solve_solution_index.json"))
graded_query_ids = [qid for qid in query_ids if qid in solve_index and solve_index[qid]["solutions_markdown"]]
test_qids = graded_query_ids[:N_QUERIES]

print("[test] loading labembed embeddings (cached)...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries")
c_cache = embed_fn(ds.corpus, cache_name="full_corpus")
query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)
dumb_reranked = dumb_rerank_top10(query_ids, results, query_texts, ds.corpus)

jobs = []
for qid in test_qids:
    jobs.append(("none", qid, None))
    jobs.append(("dumb", qid, dumb_reranked[qid][0]))
    jobs.append(("gold", qid, next(iter(strict_qrels[qid].keys()))))

print(f"[test] {len(jobs)} jobs x 2 solvers, cap={CAP}", flush=True)

write_lock = threading.Lock()


def prompt_for(cid, qid):
    if cid is None:
        return SOLVE_PROMPT_NONE.format(problem=query_texts[qid])
    return SOLVE_PROMPT_WITH_CONTEXT.format(related=ds.corpus[cid], problem=query_texts[qid])


def run_glm_job(job):
    cond, qid, cid = job
    prompt = prompt_for(cid, qid)
    t0 = time.time()
    r = call_glm(prompt)
    dt = time.time() - t0
    capped = r["completion_tokens"] >= CAP or r["finish_reason"] == "length"
    rec = {"condition": cond, "query_id": qid, "prompt_tokens": r["prompt_tokens"],
           "completion_tokens": r["completion_tokens"], "finish_reason": r["finish_reason"],
           "capped": capped, "wall_s": dt}
    with write_lock:
        print(f"[glm] {cond}/{qid}: in={r['prompt_tokens']} out={r['completion_tokens']} "
              f"finish={r['finish_reason']} wall={dt:.0f}s", flush=True)
    return rec


def run_deepseek_job(job):
    cond, qid, cid = job
    prompt = prompt_for(cid, qid)
    t0 = time.time()
    r = deepseek_call_solver(prompt, max_tokens=CAP)
    dt = time.time() - t0
    cost = record_call("deepseek-v4-flash", r["prompt_tokens"], r["completion_tokens"],
                        note=f"headtohead32k {cond}/{qid}")
    capped = r["completion_tokens"] >= CAP or r["finish_reason"] == "length"
    rec = {"condition": cond, "query_id": qid, "prompt_tokens": r["prompt_tokens"],
           "completion_tokens": r["completion_tokens"], "finish_reason": r["finish_reason"],
           "capped": capped, "wall_s": dt, "cost": cost}
    with write_lock:
        print(f"[deepseek] {cond}/{qid}: in={r['prompt_tokens']} out={r['completion_tokens']} "
              f"finish={r['finish_reason']} wall={dt:.0f}s cost=${cost:.5f}", flush=True)
    return rec


t0 = time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    glm_records = list(ex.map(run_glm_job, jobs))
print(f"[test] GLM done in {time.time()-t0:.0f}s total", flush=True)

t1 = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    deepseek_records = list(ex.map(run_deepseek_job, jobs))
print(f"[test] DeepSeek done in {time.time()-t1:.0f}s total", flush=True)

json.dump({"glm": glm_records, "deepseek": deepseek_records},
          open("results/solver_headtohead_32k.json", "w"), indent=2)


def report(name, records):
    print(f"\n=== {name} (cap={CAP}, n={len(records)}) ===")
    by_cond = {}
    for r in records:
        by_cond.setdefault(r["condition"], []).append(r)
    print(f"{'condition':10s} {'n':>3s} {'capped':>7s} {'pct':>6s} {'mean_out':>9s} {'mean_wall_s':>12s}")
    total_capped = 0
    for cond in ["none", "dumb", "gold"]:
        items = by_cond[cond]
        n = len(items)
        capped = sum(1 for r in items if r["capped"])
        total_capped += capped
        mean_out = sum(r["completion_tokens"] for r in items) / n
        mean_wall = sum(r["wall_s"] for r in items) / n
        print(f"{cond:10s} {n:3d} {capped:7d} {capped/n*100:5.1f}% {mean_out:9.0f} {mean_wall:12.1f}")
    n_total = len(records)
    print(f"{'TOTAL':10s} {n_total:3d} {total_capped:7d} {total_capped/n_total*100:5.1f}%")
    outs = sorted(r["completion_tokens"] for r in records)
    print(f"output length distribution (sorted): {outs}")
    mean_wall_all = sum(r["wall_s"] for r in records) / n_total
    print(f"mean wall time per call: {mean_wall_all:.1f}s")
    if "cost" in records[0]:
        total_cost = sum(r["cost"] for r in records)
        print(f"total cost: ${total_cost:.4f}")


report("GLM 5.2", glm_records)
report("DeepSeek-v4-flash", deepseek_records)
print(f"\n{spend_line()}")
