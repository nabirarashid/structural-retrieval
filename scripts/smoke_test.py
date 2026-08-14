import sys
sys.path.insert(0, "src")

from data import MathNetEasy
from embed_gemini import embed_all as embed_gemini
from embed_deepinfra import embed_all as embed_deepinfra
from eval import build_results, run_beir_eval, hit_at_k, failure_taxonomy, to_matrix

SEED = 42
N_QUERIES = 20
SMOKE_POOL_SIZE = 500

print("Loading dataset...")
ds = MathNetEasy.load()

query_ids = ds.sample_queries(N_QUERIES, seed=SEED)
pool = ds.small_smoke_pool(query_ids, seed=SEED, target_size=SMOKE_POOL_SIZE)
corpus_ids = list(pool.keys())
print(f"smoke: {len(query_ids)} queries, {len(corpus_ids)} corpus items")

query_texts = {qid: ds.queries[qid] for qid in query_ids}
lenient_qrels = ds.lenient_qrels_for(query_ids)
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

providers = [
    ("gemini", embed_gemini, dict(task_type="RETRIEVAL_QUERY"), dict(task_type="RETRIEVAL_DOCUMENT")),
    ("deepinfra", embed_deepinfra, {}, {}),
]

for provider_name, embed_fn, kwargs_q, kwargs_c in providers:
    print(f"\n=== {provider_name} ===")
    q_cache = embed_fn(query_texts, cache_name="smoke_queries_v2", **kwargs_q)
    c_cache = embed_fn(pool, cache_name="smoke_corpus_v2", **kwargs_c)

    query_matrix = to_matrix(query_ids, q_cache)
    corpus_matrix = to_matrix(corpus_ids, c_cache)
    results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix)

    strict_recall = run_beir_eval(strict_qrels, {k: dict(v) for k, v in results.items()})
    lenient_recall = run_beir_eval(lenient_qrels, {k: dict(v) for k, v in results.items()})
    strict_hit = hit_at_k(query_ids, results, strict_qrels)
    lenient_hit = hit_at_k(query_ids, results, lenient_qrels)

    print("STRICT  Recall@k (BEIR):", {k: v for k, v in strict_recall.items() if "Recall" in k})
    print("STRICT  Hit@k          :", strict_hit)
    print("LENIENT Recall@k (BEIR, penalized by 3 gold items -- see note):",
          {k: v for k, v in lenient_recall.items() if "Recall" in k})
    print("LENIENT Hit@k (any sibling counts -- the intended comparison):", lenient_hit)

    cats, details = failure_taxonomy(query_ids, results, strict_qrels)
    print("failure categories (strict rank-1 misses):", dict(cats))

print("\nSmoke test complete.")
