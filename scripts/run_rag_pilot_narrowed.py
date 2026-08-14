"""Narrowed follow-up to the 6-condition RAG pilot (results/rag_pilot.md), which
was inconclusive at n=88 split six ways -- no pairwise comparison reached
p<0.05 despite a 12.5-point spread in the raw numbers. Per direction: three
conditions with real power beats six with none, and none-vs-dumb is the
actual question (does deliberately bad retrieval fall below the no-retrieval
floor?). gold is kept as the oracle ceiling for reference.

  none  -- zero-shot, no retrieved context
  dumb  -- dumb_reranker.py's top-1 (deliberately bad retrieval)
  gold  -- the true ::eq::hard equivalent (oracle)

n=250 (first 250 of the same fixed 500-query hard-tier sample used throughout
this project, seed=42) -- the first 88 gradeable queries here are the same
queries already graded in the original pilot, so this both extends power and
is directly comparable to the original run's none/dumb/gold numbers, not a
fresh independent sample.

Same solver (glm-5.2-fp8), same grader (gemini-3-flash-preview, thinking
disabled), same problem-only-context design and its stated limitations --
see results/rag_pilot.md for the full methodology notes, which apply
unchanged here.
"""
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, "src")

from data import MathNetEasy
from dumb_reranker import rerank_top10 as dumb_rerank_top10
from eval import build_results, to_matrix
from embed_deepinfra import embed_all as embed_fn
from llm_reranker import GeminiJudgeBackend, _HttpError, _read_env_var

SEED = 42
N_PILOT = 250
RESULTS_DIR = Path("results")
SOLVE_INDEX_PATH = Path("data/solve_solution_index.json")
PILOT_CACHE_PATH = Path("rag_pilot_narrowed_cache.jsonl")

SOLVER_MODEL = "glm-5.2-fp8"
SOLVER_BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")
SOLVER_MAX_TOKENS = 8192
SOLVER_WORKERS = 6

GRADER_MODEL = "gemini-3-flash-preview"
GRADER_MAX_TOKENS = 32
GRADER_WORKERS = 5

SOLVE_PROMPT_NONE = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

PROBLEM:
{problem}"""

SOLVE_PROMPT_WITH_CONTEXT = """Solve the following mathematics olympiad problem. Give a complete, rigorous solution.

Here is a related problem that may (or may not) be helpful context -- use it only if it actually helps; if it isn't relevant, ignore it and solve directly.

RELATED PROBLEM:
{related}

PROBLEM TO SOLVE:
{problem}"""

GRADE_PROMPT = """You are grading a candidate solution to a mathematics olympiad problem against a reference solution.

PROBLEM:
{problem}

REFERENCE SOLUTION:
{reference}

CANDIDATE SOLUTION:
{candidate}

Score the candidate solution's correctness on a scale from 0 to 7, where 7 means fully correct \
(or containing only minor errors that don't affect the core reasoning) and 0 means completely \
incorrect or no meaningful progress. Judge whether the candidate's mathematical reasoning and \
final conclusion are consistent with the reference, not writing style or presentation. The \
reference may be in a different language than the candidate -- grade the mathematical content, \
not the language.

Respond with ONLY the integer score (0-7). No explanation, no other text."""


def _parse_int_in_range(text: str, lo: int, hi: int) -> int | None:
    for raw in reversed(re.findall(r"\d+", text)):
        val = int(raw)
        if lo <= val <= hi:
            return val
    return None


def call_solver(prompt: str) -> str:
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{SOLVER_BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={
                    "model": SOLVER_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "max_tokens": SOLVER_MAX_TOKENS,
                },
                timeout=300,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            content = msg.get("content") or ""
            if not content.strip():
                content = msg.get("reasoning_content") or ""
            return content
        except requests.exceptions.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(30, 2**attempt))


grader_backend = GeminiJudgeBackend(GRADER_MODEL, max_tokens=GRADER_MAX_TOKENS, disable_thinking=True)


def call_grader(prompt: str) -> str:
    for attempt in range(6):
        try:
            return grader_backend.call(prompt)
        except _HttpError as e:
            if e.status in (429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(30, 2**attempt))
                continue
            raise


ds = MathNetEasy.load(tier="hard")
all_query_ids = ds.sample_queries(500, seed=SEED)
query_ids = all_query_ids[:N_PILOT]
corpus_ids = list(ds.corpus.keys())
query_texts = {qid: ds.queries[qid] for qid in query_ids}
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

if not SOLVE_INDEX_PATH.exists():
    raise RuntimeError(f"{SOLVE_INDEX_PATH} missing -- run scripts/build_solve_solution_index.py first")
solve_index = json.load(open(SOLVE_INDEX_PATH))
graded_query_ids = [qid for qid in query_ids if qid in solve_index and solve_index[qid]["solutions_markdown"]]
print(f"[pilot] {len(graded_query_ids)}/{N_PILOT} queries have a matched reference solution", flush=True)

print("[pilot] loading DeepInfra embeddings (cached)...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries")
c_cache = embed_fn(ds.corpus, cache_name="full_corpus")
query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)

dumb_reranked = dumb_rerank_top10(query_ids, results, query_texts, ds.corpus)

CONDITIONS = ["none", "dumb", "gold"]


def retrieved_id(condition: str, qid: str) -> str | None:
    if condition == "none":
        return None
    if condition == "dumb":
        return dumb_reranked[qid][0]
    if condition == "gold":
        return next(iter(strict_qrels[qid].keys()))
    raise ValueError(condition)


cache: dict[tuple, dict] = {}
if PILOT_CACHE_PATH.exists():
    with open(PILOT_CACHE_PATH) as f:
        for line in f:
            d = json.loads(line)
            cache[(d["condition"], d["query_id"])] = d

write_lock = threading.Lock()
out_f = open(PILOT_CACHE_PATH, "a")


def process(condition: str, qid: str) -> None:
    key = (condition, qid)
    if key in cache:
        return
    cid = retrieved_id(condition, qid)
    if cid is None:
        prompt = SOLVE_PROMPT_NONE.format(problem=query_texts[qid])
    else:
        prompt = SOLVE_PROMPT_WITH_CONTEXT.format(related=ds.corpus[cid], problem=query_texts[qid])

    solution = call_solver(prompt)

    ref_solutions = solve_index[qid]["solutions_markdown"]
    reference = "\n\n---\n\n".join(ref_solutions)
    grade_prompt = GRADE_PROMPT.format(problem=query_texts[qid], reference=reference, candidate=solution)
    raw_score = call_grader(grade_prompt)
    score = _parse_int_in_range(raw_score, 0, 7)
    correct = score is not None and score >= 6

    record = {
        "condition": condition, "query_id": qid, "retrieved_id": cid,
        "solution": solution, "raw_score": raw_score, "score": score, "correct": correct,
    }
    with write_lock:
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        cache[key] = record


jobs = [(cond, qid) for cond in CONDITIONS for qid in graded_query_ids if (cond, qid) not in cache]
print(f"[pilot] {len(cache)} cached, {len(jobs)} to run (solver + grader per job)", flush=True)

t0 = time.time()
done = 0
with ThreadPoolExecutor(max_workers=SOLVER_WORKERS) as ex:
    futures = [ex.submit(process, cond, qid) for cond, qid in jobs]
    for fut in futures:
        fut.result()
        done += 1
        if done % 25 == 0 or done == len(jobs):
            print(f"[pilot] {done}/{len(jobs)} done ({time.time()-t0:.0f}s elapsed)", flush=True)
out_f.close()

summary = {}
for cond in CONDITIONS:
    n = len(graded_query_ids)
    n_correct = sum(1 for qid in graded_query_ids if cache[(cond, qid)]["correct"])
    n_unparsed = sum(1 for qid in graded_query_ids if cache[(cond, qid)]["score"] is None)
    summary[cond] = {"n": n, "n_correct": n_correct, "pct_correct": n_correct / n * 100, "n_unparsed_grade": n_unparsed}

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "rag_pilot_narrowed.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== NARROWED RAG PILOT SUMMARY (n=%d graded queries) ===" % len(graded_query_ids))
for cond in CONDITIONS:
    s = summary[cond]
    print(f"{cond:8s}  {s['n_correct']:3d}/{s['n']:3d}  ({s['pct_correct']:5.1f}%)  unparsed_grade={s['n_unparsed_grade']}")
