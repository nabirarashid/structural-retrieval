import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from data import MathNetEasy
from eval import build_results, to_matrix
from dumb_reranker import rerank_top10

SEED = 42
N_QUERIES = 500
RESULTS_DIR = Path("results")

CONFIGS = [
    ("gemini", "easy"),
    ("gemini", "hard"),
    ("deepinfra", "easy"),
    ("deepinfra", "hard"),
]

report = []

for provider, tier in CONFIGS:
    t0 = time.time()
    print(f"\n=== {provider} / {tier} ===", flush=True)

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

    # original (embedding-only) Hit@1 / Hit@10, for the recoverable-gap denominator
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

    reranked = rerank_top10(query_ids, results, query_texts, ds.corpus)
    dumb_hit1 = sum(
        1 for qid in query_ids if reranked[qid][0] in strict_qrels[qid]
    ) / N_QUERIES

    recoverable_gap = orig_hit10 - orig_hit1
    closed = dumb_hit1 - orig_hit1
    share_closed = (closed / recoverable_gap) if recoverable_gap > 0 else float("nan")

    result = {
        "provider": provider,
        "tier": tier,
        "orig_hit1": orig_hit1,
        "orig_hit10": orig_hit10,
        "dumb_reranked_hit1": dumb_hit1,
        "recoverable_gap": recoverable_gap,
        "gap_closed": closed,
        "share_of_recoverable_gap_closed": share_closed,
        "elapsed_seconds": time.time() - t0,
    }
    report.append(result)
    print(json.dumps(result, indent=2), flush=True)

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "dumb_reranker_control.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n=== SUMMARY ===")
for r in report:
    print(
        f"{r['provider']:10s} {r['tier']:5s}  "
        f"orig Hit@1={r['orig_hit1']*100:5.1f}%  "
        f"dumb-reranked Hit@1={r['dumb_reranked_hit1']*100:5.1f}%  "
        f"recoverable gap={r['recoverable_gap']*100:5.1f}pp  "
        f"share closed={r['share_of_recoverable_gap_closed']*100:6.1f}%"
    )
