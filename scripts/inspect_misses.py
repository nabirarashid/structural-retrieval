import json
import sys

sys.path.insert(0, "src")

from data import MathNetEasy
from eval import to_matrix, build_results, failure_taxonomy

SEED = 42
N_QUERIES = 500

ds = MathNetEasy.load()
query_ids = ds.sample_queries(N_QUERIES, seed=SEED)
corpus_ids = list(ds.corpus.keys())
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

from embed_gemini import embed_all as embed_gemini

q_cache = embed_gemini({qid: ds.queries[qid] for qid in query_ids}, task_type="RETRIEVAL_QUERY", cache_name="full_queries")
c_cache = embed_gemini(ds.corpus, task_type="RETRIEVAL_DOCUMENT", cache_name="full_corpus")

query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)

cats, details = failure_taxonomy(query_ids, results, strict_qrels)

# pick 3 near-miss examples (the dominant failure mode) for manual inspection
examples = [d for d in details if d["category"] == "own_nm_near_miss"][:3]

for ex in examples:
    qid = ex["query_id"]
    print("=" * 100)
    print(f"QUERY ({qid}):")
    print(ds.queries[qid])
    print()
    print(f"GOLD TARGET ({ex['gold_id']}):")
    print(ds.corpus[ex["gold_id"]])
    print()
    ranked = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:10]
    for rank, (cid, score) in enumerate(ranked, 1):
        marker = " <-- GOLD" if cid == ex["gold_id"] else (" <-- TOP-1 (WRONG)" if rank == 1 else "")
        print(f"  #{rank} [{score:.4f}] {cid}{marker}")
        print(f"       {ds.corpus[cid][:200]}")
    print()
