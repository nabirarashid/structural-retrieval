import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from data import MathNetEasy
from eval import build_results, run_beir_eval, hit_at_k, failure_taxonomy, to_matrix

SEED = 42
N_QUERIES = 500
RESULTS_DIR = Path("results")

parser = argparse.ArgumentParser()
parser.add_argument("--provider", choices=["gemini", "deepinfra", "labembed"], required=True)
args = parser.parse_args()

if args.provider == "gemini":
    from embed_gemini import embed_all as embed_fn
    embed_kwargs_q = dict(task_type="RETRIEVAL_QUERY")
    embed_kwargs_c = dict(task_type="RETRIEVAL_DOCUMENT")
elif args.provider == "labembed":
    from embed_labembed import embed_all as embed_fn
    embed_kwargs_q = {}
    embed_kwargs_c = {}
else:
    from embed_deepinfra import embed_all as embed_fn
    embed_kwargs_q = {}
    embed_kwargs_c = {}

t0 = time.time()
print(f"[{args.provider}/hard] loading dataset...", flush=True)
ds = MathNetEasy.load(tier="hard")  # same corpus/queries as easy, hard qrels

query_ids = ds.sample_queries(N_QUERIES, seed=SEED)  # identical sample, same seed
corpus_ids = list(ds.corpus.keys())
print(f"[{args.provider}/hard] {len(query_ids)} queries (seed={SEED}, same sample as easy run), "
      f"{len(corpus_ids)} corpus items", flush=True)

query_texts = {qid: ds.queries[qid] for qid in query_ids}
lenient_qrels = ds.lenient_qrels_for(query_ids)  # tier-agnostic: any of 3 siblings
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}  # hard tier's ::eq::hard target

print(f"[{args.provider}/hard] loading embeddings (should be fully cached from the easy-tier run "
      f"-- same corpus/query text, no new API calls expected)...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries", **embed_kwargs_q)
c_cache = embed_fn(ds.corpus, cache_name="full_corpus", **embed_kwargs_c)

query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)

strict_recall = run_beir_eval(strict_qrels, {k: dict(v) for k, v in results.items()})
lenient_recall = run_beir_eval(lenient_qrels, {k: dict(v) for k, v in results.items()})
strict_hit = hit_at_k(query_ids, results, strict_qrels)
lenient_hit = hit_at_k(query_ids, results, lenient_qrels)
cats, details = failure_taxonomy(query_ids, results, strict_qrels)

RESULTS_DIR.mkdir(exist_ok=True)
out = {
    "provider": args.provider,
    "tier": "hard",
    "seed": SEED,
    "n_queries": N_QUERIES,
    "corpus_size": len(corpus_ids),
    "strict_recall_beir": strict_recall,
    "lenient_recall_beir": lenient_recall,
    "strict_hit": strict_hit,
    "lenient_hit": lenient_hit,
    "failure_categories": dict(cats),
    "failure_details": details,
    "elapsed_seconds": time.time() - t0,
}
out_path = RESULTS_DIR / f"baseline_{args.provider}_hard.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"[{args.provider}/hard] DONE in {out['elapsed_seconds']:.0f}s. Saved to {out_path}", flush=True)
print(f"[{args.provider}/hard] STRICT Hit@k:  {strict_hit}", flush=True)
print(f"[{args.provider}/hard] LENIENT Hit@k: {lenient_hit}", flush=True)
print(f"[{args.provider}/hard] Failure categories: {dict(cats)}", flush=True)
