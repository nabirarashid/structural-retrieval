"""Follow-up to the GLM CoT truncation finding (63.3% of the full run hit the
6,144-token cap without concluding -- see the correction banner in
results/llm_reranker_cot_full_comparison.md). Tests whether a larger budget
(16,384) PLUS an explicit concise-reasoning instruction (COT_CONCISE_PROMPT_TEMPLATE)
gets truncation down to a usable rate, on a small 30-query sample before
committing to a full rerun. Per the decision gate: <~10% truncation -> rerun
properly at full scale; otherwise GLM may not be usable as a CoT judge, and
that itself is worth reporting.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, "src")

from data import MathNetEasy
from eval import build_results, to_matrix
from embed_gemini import embed_all as embed_fn
from llm_reranker import COT_CONCISE_PROMPT_TEMPLATE, _build_prompt, _parse_choice_cot, _read_env_var

SEED = 42
N_TEST = 30
MAX_TOKENS = 16384
MAX_WORKERS = 6
base_url = _read_env_var("MANTIS_LLM_BASE_URL")


def call_glm(prompt: str) -> tuple[str, dict]:
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={
                    "model": "glm-5.2-fp8",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": MAX_TOKENS,
                },
                timeout=300,
            )
            resp.raise_for_status()
            d = resp.json()
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or ""
            return content, d["usage"]
        except requests.exceptions.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(30, 2**attempt))


ds = MathNetEasy.load(tier="hard")
all_query_ids = ds.sample_queries(500, seed=SEED)
query_ids = all_query_ids[:N_TEST]
corpus_ids = list(ds.corpus.keys())
query_texts = {qid: ds.queries[qid] for qid in all_query_ids}
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

q_cache = embed_fn(query_texts, cache_name="full_queries", task_type="RETRIEVAL_QUERY")
c_cache = embed_fn(ds.corpus, cache_name="full_corpus", task_type="RETRIEVAL_DOCUMENT")
qm = to_matrix(all_query_ids, q_cache)
cm = to_matrix(corpus_ids, c_cache)
full_results = build_results(all_query_ids, qm, corpus_ids, cm, top_n=200)
results = {qid: full_results[qid] for qid in query_ids}

orig_hit1_n = sum(1 for qid in query_ids if max(results[qid].items(), key=lambda kv: kv[1])[0] in strict_qrels[qid])
orig_hit10_n = sum(
    1 for qid in query_ids
    if set(strict_qrels[qid]) & {cid for cid, _ in sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:10]}
)
print(f"[baseline] orig Hit@1: {orig_hit1_n}/{N_TEST}  orig Hit@10: {orig_hit10_n}/{N_TEST}", flush=True)


def process(qid):
    top10 = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:10]
    top10_ids = [cid for cid, _ in top10]
    cand_texts = [ds.corpus[cid] for cid in top10_ids]
    prompt = _build_prompt(query_texts[qid], cand_texts, COT_CONCISE_PROMPT_TEMPLATE)
    text, usage = call_glm(prompt)
    choice = _parse_choice_cot(text, 10)
    return {
        "query_id": qid,
        "text": text,
        "usage": usage,
        "choice": choice,
        "top10_ids": top10_ids,
        "hit": choice is not None and top10_ids[choice - 1] in strict_qrels[qid],
    }


t0 = time.time()
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    records = list(ex.map(process, query_ids))
elapsed = time.time() - t0

completion_tokens = [r["usage"]["completion_tokens"] for r in records]
n_capped = sum(1 for c in completion_tokens if c >= MAX_TOKENS)
n_unparsed = sum(1 for r in records if r["choice"] is None)
n_hits = sum(1 for r in records if r["hit"])
n_final_answer_marker = sum(1 for r in records if "FINAL ANSWER" in r["text"].upper())

with open("results/glm_concise_cot_test.json", "w") as f:
    json.dump(
        {
            "n": N_TEST, "max_tokens_cap": MAX_TOKENS,
            "orig_hit1_n": orig_hit1_n, "orig_hit10_n": orig_hit10_n,
            "hits": n_hits, "unparsed": n_unparsed, "capped": n_capped,
            "used_final_answer_marker": n_final_answer_marker,
            "mean_completion_tokens": sum(completion_tokens) / len(completion_tokens),
            "elapsed_seconds": elapsed,
            "records": [{k: v for k, v in r.items() if k != "top10_ids"} for r in records],
        },
        f, indent=2,
    )

print(f"\n=== concise-CoT, 16,384 cap, n={N_TEST} ===")
print(f"Hit@1: {n_hits}/{N_TEST} ({n_hits/N_TEST*100:.1f}%)")
print(f"unparsed: {n_unparsed}/{N_TEST}")
print(f"hit the {MAX_TOKENS}-token cap: {n_capped}/{N_TEST} ({n_capped/N_TEST*100:.1f}%)")
print(f"used FINAL ANSWER marker: {n_final_answer_marker}/{N_TEST}")
print(f"mean completion tokens: {sum(completion_tokens)/len(completion_tokens):.1f}")
print(f"elapsed: {elapsed:.0f}s")
