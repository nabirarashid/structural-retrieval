"""Full 500-query x 4-config rerun with COT_PROMPT_TEMPLATE, for one judge at
a time (--judge gemini|glm). Run once per judge -- same prompt for both, so
the comparison stays valid, per the diagnostic finding that the terse prompt
was suppressing glm-5.2-fp8's reasoning specifically (scripts/test_glm_cot_prompt.py:
Hit@1 4%->10%, avg completion tokens 84->3094, on a 50-query hard-tier check).

Threaded (max_workers) since CoT calls are much slower per-call than the terse
prompt's near-instant answers -- worth the concurrency given a reasoning judge
can take ~15-20s/call.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from data import MathNetEasy
from eval import build_results, to_matrix
from llm_reranker import (
    COT_PROMPT_TEMPLATE,
    GeminiJudgeBackend,
    OpenAICompatJudgeBackend,
    _parse_choice_cot,
    _read_env_var,
    rerank_top10_llm,
)

SEED = 42
N_QUERIES = 500
RESULTS_DIR = Path("results")

WELL_KNOWN = {"imo", "usa", "apm"}
REGIONAL = {"rus", "blr", "twn", "mng", "hrv", "bra"}


def bucket(qid: str) -> str:
    prefix = qid.split("_")[0]
    if prefix in WELL_KNOWN:
        return "well_known"
    if prefix in REGIONAL:
        return "regional"
    return "other"


parser = argparse.ArgumentParser()
parser.add_argument("--judge", choices=["gemini", "glm"], required=True)
args = parser.parse_args()

if args.judge == "gemini":
    JUDGE_LABEL = "gemini-3.1-flash-lite"
    BACKEND = GeminiJudgeBackend("gemini-3.1-flash-lite", max_tokens=4096)
    MAX_WORKERS = 5
else:
    JUDGE_LABEL = "glm-5.2-fp8"
    BACKEND = OpenAICompatJudgeBackend(
        model="glm-5.2-fp8", base_url=_read_env_var("MANTIS_LLM_BASE_URL"), max_tokens=6144,
    )
    MAX_WORKERS = 6

CONFIGS = [("gemini", "easy"), ("gemini", "hard"), ("deepinfra", "easy"), ("deepinfra", "hard")]

report = []
for provider, tier in CONFIGS:
    t0 = time.time()
    print(f"\n=== COT RUN, judge={JUDGE_LABEL} ({N_QUERIES} queries): {provider} / {tier} ===", flush=True)

    if provider == "gemini":
        from embed_gemini import embed_all as embed_fn
        kwargs_q = dict(task_type="RETRIEVAL_QUERY")
        kwargs_c = dict(task_type="RETRIEVAL_DOCUMENT")
    else:
        from embed_deepinfra import embed_all as embed_fn
        kwargs_q, kwargs_c = {}, {}

    ds = MathNetEasy.load(tier=tier)
    query_ids = ds.sample_queries(N_QUERIES, seed=SEED)
    corpus_ids = list(ds.corpus.keys())
    query_texts = {qid: ds.queries[qid] for qid in query_ids}
    strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

    q_cache = embed_fn(query_texts, cache_name="full_queries", **kwargs_q)
    c_cache = embed_fn(ds.corpus, cache_name="full_corpus", **kwargs_c)
    query_matrix = to_matrix(query_ids, q_cache)
    corpus_matrix = to_matrix(corpus_ids, c_cache)
    results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)

    orig_hit1 = sum(
        1 for qid in query_ids
        if max(results[qid].items(), key=lambda kv: kv[1])[0] in strict_qrels[qid]
    ) / N_QUERIES
    orig_hit10 = sum(
        1 for qid in query_ids
        if set(strict_qrels[qid]) & {
            cid for cid, _ in sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:10]
        }
    ) / N_QUERIES

    cache_name = f"cot_{args.judge}_{provider}_{tier}"
    choices = rerank_top10_llm(
        query_ids, results, query_texts, ds.corpus, backend=BACKEND, cache_name=cache_name,
        prompt_template=COT_PROMPT_TEMPLATE, parse_fn=_parse_choice_cot, max_workers=MAX_WORKERS,
    )
    n_unparsed = sum(1 for v in choices.values() if v is None)
    llm_hit1 = sum(
        1 for qid in query_ids if choices[qid] is not None and choices[qid] in strict_qrels[qid]
    ) / N_QUERIES

    recoverable_gap = orig_hit10 - orig_hit1
    closed = llm_hit1 - orig_hit1
    share_closed = (closed / recoverable_gap) if recoverable_gap > 0 else float("nan")

    by_bucket = {}
    for b in ("well_known", "regional", "other"):
        bqids = [qid for qid in query_ids if bucket(qid) == b]
        n_b = len(bqids)
        if n_b == 0:
            by_bucket[b] = {"n": 0}
            continue
        b_orig_n = sum(1 for qid in bqids if max(results[qid].items(), key=lambda kv: kv[1])[0] in strict_qrels[qid])
        b_llm_n = sum(1 for qid in bqids if choices[qid] is not None and choices[qid] in strict_qrels[qid])
        by_bucket[b] = {
            "n": n_b,
            "orig_hit1_n": b_orig_n, "orig_hit1_pct": b_orig_n / n_b * 100,
            "llm_hit1_n": b_llm_n, "llm_hit1_pct": b_llm_n / n_b * 100,
        }

    wk_qids = [qid for qid in query_ids if bucket(qid) == "well_known"]
    rest_qids = [qid for qid in query_ids if bucket(qid) != "well_known"]
    wk_n, rest_n = len(wk_qids), len(rest_qids)
    wk_llm_n = sum(1 for qid in wk_qids if choices[qid] is not None and choices[qid] in strict_qrels[qid])
    rest_llm_n = sum(1 for qid in rest_qids if choices[qid] is not None and choices[qid] in strict_qrels[qid])
    pooled = {
        "well_known_n": wk_n, "well_known_llm_hit1_n": wk_llm_n,
        "well_known_llm_hit1_pct": wk_llm_n / wk_n * 100 if wk_n else None,
        "rest_n": rest_n, "rest_llm_hit1_n": rest_llm_n,
        "rest_llm_hit1_pct": rest_llm_n / rest_n * 100 if rest_n else None,
    }

    result = {
        "judge": JUDGE_LABEL, "prompt": "cot", "provider": provider, "tier": tier, "n_queries": N_QUERIES,
        "orig_hit1": orig_hit1, "orig_hit10": orig_hit10, "llm_reranked_hit1": llm_hit1,
        "n_unparsed_responses": n_unparsed,
        "recoverable_gap": recoverable_gap, "gap_closed": closed,
        "share_of_recoverable_gap_closed": share_closed,
        "by_contamination_bucket": by_bucket,
        "pooled_well_known_vs_rest": pooled,
        "elapsed_seconds": time.time() - t0,
    }
    report.append(result)
    print(json.dumps(result, indent=2), flush=True)

RESULTS_DIR.mkdir(exist_ok=True)
out_path = RESULTS_DIR / f"llm_reranker_full_cot_{args.judge}.json"
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n=== COT RUN SUMMARY (judge={JUDGE_LABEL}) ===")
for r in report:
    print(
        f"{r['provider']:10s} {r['tier']:5s}  "
        f"orig Hit@1={r['orig_hit1']*100:5.1f}%  "
        f"LLM-reranked Hit@1={r['llm_reranked_hit1']*100:5.1f}%  "
        f"share closed={r['share_of_recoverable_gap_closed']*100:6.1f}%  "
        f"unparsed={r['n_unparsed_responses']}"
    )
