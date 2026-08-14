import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

from data import MathNetEasy, parse_corpus_id
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
print(f"[{args.provider}] loading dataset...", flush=True)
ds = MathNetEasy.load()

query_ids = ds.sample_queries(N_QUERIES, seed=SEED)
corpus_ids = list(ds.corpus.keys())
print(f"[{args.provider}] {len(query_ids)} queries (seed={SEED}), {len(corpus_ids)} corpus items (FULL, no subsampling)", flush=True)

query_texts = {qid: ds.queries[qid] for qid in query_ids}
lenient_qrels = ds.lenient_qrels_for(query_ids)
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

print(f"[{args.provider}] embedding {len(query_texts)} queries...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries", **embed_kwargs_q)

print(f"[{args.provider}] embedding {len(ds.corpus)} corpus items (this is the long part)...", flush=True)
c_cache = embed_fn(ds.corpus, cache_name="full_corpus", **embed_kwargs_c)

print(f"[{args.provider}] embedding done in {time.time()-t0:.0f}s, scoring...", flush=True)
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
out_path = RESULTS_DIR / f"baseline_{args.provider}.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"[{args.provider}] DONE in {out['elapsed_seconds']:.0f}s. Saved to {out_path}", flush=True)
print(f"[{args.provider}] STRICT Hit@k:  {strict_hit}", flush=True)
print(f"[{args.provider}] LENIENT Hit@k: {lenient_hit}", flush=True)
print(f"[{args.provider}] Failure categories: {dict(cats)}", flush=True)
