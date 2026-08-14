"""Settles the limitations-section question before committing to the full
run: on the 11 (condition, query_id) pairs that hit the 32,768 cap under
EITHER solver in the head-to-head test, does raising GLM's cap to 65,536
keep shrinking truncation (budget-bound residual) or do the same problems
cap out again (a hard core that doesn't converge regardless of budget)?
GLM only, free lab endpoint.
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

SEED = 42
N_PILOT = 250
CAP = 65536
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

TARGET_PAIRS = [
    ("dumb", "bra_2019_7792eb"), ("dumb", "fra_2014_eb9ec4"), ("dumb", "srb_2016_cab572"),
    ("dumb", "svn_2023_c9c438"), ("gold", "bra_2019_7792eb"), ("gold", "fra_2014_eb9ec4"),
    ("gold", "srb_2016_cab572"), ("gold", "tur_2008_22f4a1"), ("none", "bra_2019_7792eb"),
    ("none", "fra_2014_eb9ec4"), ("none", "srb_2016_cab572"),
]


def call_glm(prompt: str) -> dict:
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{GLM_BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={"model": GLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.0, "max_tokens": CAP},
                timeout=900,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            content = msg.get("content") or msg.get("reasoning_content") or ""
            usage = data.get("usage", {})
            return {"text": content, "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "finish_reason": choice.get("finish_reason", "")}
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

print("[test] loading labembed embeddings (cached)...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries")
c_cache = embed_fn(ds.corpus, cache_name="full_corpus")
query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)
dumb_reranked = dumb_rerank_top10(query_ids, results, query_texts, ds.corpus)


def cid_for(cond, qid):
    if cond == "none":
        return None
    if cond == "dumb":
        return dumb_reranked[qid][0]
    return next(iter(strict_qrels[qid].keys()))


def prompt_for(cond, qid):
    cid = cid_for(cond, qid)
    if cid is None:
        return SOLVE_PROMPT_NONE.format(problem=query_texts[qid])
    return SOLVE_PROMPT_WITH_CONTEXT.format(related=ds.corpus[cid], problem=query_texts[qid])


write_lock = threading.Lock()


def run_job(job):
    cond, qid = job
    prompt = prompt_for(cond, qid)
    t0 = time.time()
    r = call_glm(prompt)
    dt = time.time() - t0
    capped = r["completion_tokens"] >= CAP or r["finish_reason"] == "length"
    rec = {"condition": cond, "query_id": qid, "completion_tokens": r["completion_tokens"],
           "finish_reason": r["finish_reason"], "capped": capped, "wall_s": dt}
    with write_lock:
        print(f"[65k] {cond}/{qid}: out={r['completion_tokens']} finish={r['finish_reason']} "
              f"capped={capped} wall={dt:.0f}s", flush=True)
    return rec


print(f"[test] running {len(TARGET_PAIRS)} GLM calls at cap={CAP} on the previously-capped-at-32k queries...", flush=True)
with ThreadPoolExecutor(max_workers=6) as ex:
    records = list(ex.map(run_job, TARGET_PAIRS))

json.dump(records, open("results/glm_65k_residual.json", "w"), indent=2)
n_capped = sum(1 for r in records if r["capped"])
print(f"\n=== GLM 65,536 RESIDUAL TEST (n={len(records)}, all previously capped at 32k) ===")
print(f"still capped at 65k: {n_capped}/{len(records)} ({n_capped/len(records)*100:.1f}%)")
for r in records:
    print(f"  {r['condition']}/{r['query_id']}: out={r['completion_tokens']} capped={r['capped']}")
if n_capped == 0:
    print("VERDICT: all converged once given enough room -- residual truncation at 32k was budget-bound.")
elif n_capped == len(records):
    print("VERDICT: all still capped -- hard core that doesn't converge regardless of budget.")
else:
    print(f"VERDICT: mixed -- {n_capped} still cap out even at 65k, {len(records)-n_capped} converged given more room. Partial budget-bound, partial hard core.")
