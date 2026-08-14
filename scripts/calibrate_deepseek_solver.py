"""10-call calibration for the DeepSeek-v4-flash solver (native API, real cost)
before committing to the full n=250x3 utility-curve run. Reports real output
token lengths, truncation rate at the chosen cap, and a tightened cost
estimate for the full run -- gated per explicit instruction: if truncation is
under ~10%, the caller proceeds straight to the full run; if not, stop.

Solver routed via DeepSeek's native API (api.deepseek.com), not DeepInfra --
the user's own DeepSeek platform key/billing, topped up after an earlier
insufficient-balance error. Retrieval (for the dumb/gold conditions) uses
labembed embeddings, not DeepInfra either -- no DeepInfra usage anywhere
in this run. labembed's full_corpus/full_queries cache already exists
from the Aug 8-9 baseline run, so this costs nothing new on the embedding
side.
"""
import json
import sys
import time

sys.path.insert(0, "src")

from data import MathNetEasy
from dumb_reranker import rerank_top10 as dumb_rerank_top10
from embed_labembed import embed_all as embed_fn
from eval import build_results, to_matrix
from llm_solver_deepseek import call_solver
from spend_tracker import record_call, spend_line

SEED = 42
N_PILOT = 250
CAP = 8192
SOLVER_MODEL = "deepseek-v4-flash"

SOLVE_PROMPT_NONE = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

PROBLEM:
{problem}"""

SOLVE_PROMPT_WITH_CONTEXT = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

Here is a related problem that may (or may not) be helpful context -- use it only if it actually helps; if it isn't relevant, ignore it and solve directly.

RELATED PROBLEM:
{related}

PROBLEM TO SOLVE:
{problem}"""

ds = MathNetEasy.load(tier="hard")
all_query_ids = ds.sample_queries(500, seed=SEED)
query_ids = all_query_ids[:N_PILOT]
corpus_ids = list(ds.corpus.keys())
query_texts = {qid: ds.queries[qid] for qid in query_ids}
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

solve_index = json.load(open("data/solve_solution_index.json"))
graded_query_ids = [qid for qid in query_ids if qid in solve_index and solve_index[qid]["solutions_markdown"]]

print("[calib] loading labembed embeddings (cached)...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries")
c_cache = embed_fn(ds.corpus, cache_name="full_corpus")
query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)
dumb_reranked = dumb_rerank_top10(query_ids, results, query_texts, ds.corpus)

gold_qids = graded_query_ids[:4]
dumb_qids = graded_query_ids[4:7]
none_qids = graded_query_ids[7:10]

jobs = []
for qid in gold_qids:
    cid = next(iter(strict_qrels[qid].keys()))
    jobs.append(("gold", qid, cid))
for qid in dumb_qids:
    cid = dumb_reranked[qid][0]
    jobs.append(("dumb", qid, cid))
for qid in none_qids:
    jobs.append(("none", qid, None))

print(f"[calib] running {len(jobs)} {SOLVER_MODEL} solver calls via native DeepSeek API, cap={CAP} (real cost)...", flush=True)
records = []
t0 = time.time()
for i, (cond, qid, cid) in enumerate(jobs):
    if cid is None:
        prompt = SOLVE_PROMPT_NONE.format(problem=query_texts[qid])
    else:
        prompt = SOLVE_PROMPT_WITH_CONTEXT.format(related=ds.corpus[cid], problem=query_texts[qid])
    r = call_solver(prompt, max_tokens=CAP)
    cost = record_call(
        SOLVER_MODEL, r["prompt_tokens"], r["completion_tokens"],
        note=f"calibration solver call {cond}/{qid}",
        actual_cost=r.get("estimated_cost"),
    )
    capped = r["completion_tokens"] >= CAP or r["finish_reason"] == "length"
    records.append({
        "condition": cond, "query_id": qid,
        "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
        "finish_reason": r["finish_reason"], "capped": capped, "cost": cost,
    })
    print(f"[calib] {i+1}/{len(jobs)} {cond}/{qid}: in={r['prompt_tokens']} out={r['completion_tokens']} "
          f"finish={r['finish_reason']} cost=${cost:.5f} ({time.time()-t0:.0f}s)", flush=True)

n_capped = sum(1 for r in records if r["capped"])
mean_out = sum(r["completion_tokens"] for r in records) / len(records)
max_out = max(r["completion_tokens"] for r in records)
mean_in = sum(r["prompt_tokens"] for r in records) / len(records)
total_cost = sum(r["cost"] for r in records)

print(f"\n=== {SOLVER_MODEL} (native DeepSeek API) SOLVER CALIBRATION (n={len(records)}, cap={CAP}) ===")
print(f"truncated (finish_reason=length or out>=cap): {n_capped}/{len(records)} ({n_capped/len(records)*100:.1f}%)")
print(f"mean input tokens: {mean_in:.0f}   mean output tokens: {mean_out:.0f}   max output tokens: {max_out}")
print(f"actual cost, {len(records)} calls: ${total_cost:.4f}  (${total_cost/len(records)*1000:.3f} per 1000 calls... "
      f"i.e. ${total_cost/len(records):.5f}/call)")

# tightened full-run cost estimate: 630 solver jobs (same as the discarded GLM pilot's job count)
# + 630 grader-A (flash-lite) calls, using the REAL measured mean output length as candidate size
N_FULL = 630
solver_full_low = N_FULL * total_cost / len(records)
# grader input = ~615-700 fixed tokens (measured earlier from GRADE_PROMPT structure) + candidate (solver output)
fl_in, fl_out = 0.25, 1.50
grader_in = 660 + mean_out
grader_full = N_FULL * (grader_in / 1e6 * fl_in + 2 / 1e6 * fl_out)
print(f"\nTightened full-run (n={N_FULL}) estimate: solver ${solver_full_low:.3f} + grader-A(flash-lite) ~${grader_full:.3f} "
      f"= ~${solver_full_low + grader_full:.3f}  (grader-B GLM is free)")
print(spend_line())

json.dump(records, open("results/deepseek_solver_calibration.json", "w"), indent=2)
