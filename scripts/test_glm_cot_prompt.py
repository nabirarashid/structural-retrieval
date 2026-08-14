"""Diagnostic: does glm-5.2-fp8 actually engage its reasoning capability on the
reranker task if explicitly told to think step-by-step before answering, vs the
production prompt's "ONLY the number, no explanation" instruction (which in the
full run produced ~2 completion tokens/call and occasional raw `<arg_value>N`
artifacts -- looks like the terse instruction is short-circuiting deliberation,
possibly routing through a tool-call-like template path).

Same 50-query subsample convention as the earlier judge validation script
(first 50 of the fixed 500-query, seed=42 sample) so results are directly
comparable to prior numbers. Both prompt variants are freshly called here
(not read from the production cache) so completion-token accounting is
apples-to-apples -- the production cache never logged usage stats.
"""
import re
import statistics
import sys
import time

import requests

sys.path.insert(0, "src")

from data import MathNetEasy
from eval import build_results, to_matrix
from embed_gemini import embed_all as embed_fn
from llm_reranker import PROMPT_TEMPLATE as TERSE_TEMPLATE
from llm_reranker import _read_env_var

SEED = 42
N_VALIDATION = 50
BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")
MODEL = "glm-5.2-fp8"
MAX_RETRIES = 4

COT_TEMPLATE = """You are given a math competition problem ("ANCHOR") and 10 candidate \
problems, labeled 1 through 10. Exactly one candidate relies on the same underlying \
mathematical technique or method as the anchor -- the same core idea you would actually use \
to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared \
language, shared story framing (e.g. both about chessboards, or both in the same language), or \
shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to \
the anchor and still use a completely different technique, and a candidate can look nothing like \
the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve \
each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Think step by step. First state the core technique needed to solve the ANCHOR. Then, for each \
candidate in turn, briefly note the technique it actually needs and whether it matches the \
anchor's. After going through all 10, conclude.

End your response with your final answer on its own line, in exactly this format:
FINAL ANSWER: <candidate number>"""


def build_prompt(template, anchor, candidates):
    block = "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(candidates))
    return template.format(anchor=anchor, candidates=block)


def call_glm(prompt, max_tokens):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": max_tokens,
                },
                timeout=300,
            )
            resp.raise_for_status()
            d = resp.json()
            msg = d["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or ""
            return content, d["usage"], d["choices"][0]["finish_reason"]
        except requests.exceptions.RequestException:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(min(30, 2**attempt))


def parse_terse(text, n):
    matches = re.findall(r"\d+", text)
    for raw in reversed(matches):
        val = int(raw)
        if 1 <= val <= n:
            return val
    return None


def parse_cot(text, n):
    markers = re.findall(r"final answer:?\s*(\d+)", text, re.IGNORECASE)
    for raw in reversed(markers):
        val = int(raw)
        if 1 <= val <= n:
            return val
    return parse_terse(text, n)  # fall back if the model didn't use the marker


ds = MathNetEasy.load(tier="hard")
all_query_ids = ds.sample_queries(500, seed=SEED)
query_ids = all_query_ids[:N_VALIDATION]
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
print(f"[baseline] orig Hit@1: {orig_hit1_n}/{N_VALIDATION}  orig Hit@10: {orig_hit10_n}/{N_VALIDATION}", flush=True)

for label, template, parser in [("terse", TERSE_TEMPLATE, parse_terse), ("cot", COT_TEMPLATE, parse_cot)]:
    hits = 0
    unparsed = 0
    comp_tokens, reasoning_tokens = [], []
    finish_reasons = []
    examples = []
    t0 = time.time()
    for i, qid in enumerate(query_ids, 1):
        top10 = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:10]
        top10_ids = [cid for cid, _ in top10]
        cand_texts = [ds.corpus[cid] for cid in top10_ids]
        prompt = build_prompt(template, query_texts[qid], cand_texts)
        text, usage, finish_reason = call_glm(prompt, max_tokens=4096)
        choice = parser(text, 10)
        comp_tokens.append(usage["completion_tokens"])
        reasoning_tokens.append(usage.get("reasoning_tokens", 0))
        finish_reasons.append(finish_reason)
        if choice is None:
            unparsed += 1
        else:
            chosen_id = top10_ids[choice - 1]
            if chosen_id in strict_qrels[qid]:
                hits += 1
        if len(examples) < 2:
            examples.append((qid, text))
        if i % 10 == 0:
            print(f"[{label}] {i}/{N_VALIDATION}", flush=True)

    print(f"\n=== {label} ===")
    print(f"Hit@1: {hits}/{N_VALIDATION} ({hits/N_VALIDATION*100:.1f}%)  unparsed: {unparsed}")
    print(f"avg completion_tokens: {statistics.mean(comp_tokens):.1f}  avg reasoning_tokens: {statistics.mean(reasoning_tokens):.1f}")
    print(f"finish_reasons: {set(finish_reasons)}")
    print(f"elapsed: {time.time()-t0:.0f}s")
    for qid, text in examples:
        print(f"--- example ({qid}) ---")
        print(text[:600])
    print()
