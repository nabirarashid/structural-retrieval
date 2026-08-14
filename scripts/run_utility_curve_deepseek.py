"""Full n=250x3 (none/dumb/gold) RAG utility-curve run, DeepSeek-v4-flash
solver (native API) -- option 2 after the GLM run stalled 4x under sustained
shared-lab-endpoint load (6.5h for 49% completion vs ~1.6h predicted).
GLM's partial run (utility_curve_glm_cache.jsonl, 307/630 done: 210 none,
97 dumb, 0 gold) is kept as-is, NOT reused or topped up here -- it's a
separate side comparison (GLM vs DeepSeek accuracy on identical problems,
scoped to the overlapping solved jobs) once this run covers the same grid.

Two fixes vs the GLM run:
  1. Jobs submitted INTERLEAVED across conditions (round-robin qid, cycling
     none/dumb/gold), not condition-by-condition -- a stall now yields a
     balanced partial dataset instead of a confounded one.
  2. Progress reporting uses as_completed(), not ordered fut.result() --
     the GLM run's progress counter silently froze at 270/630 while 37 more
     jobs actually finished in the background, because it blocked on
     futures in submission order. as_completed() reports true completion
     order so a stall is visible immediately, not five hours later.

Two graders, no tiebreaker, per instruction -- report disagreement, don't
resolve it:
  Grader A: gemini-3.1-flash-lite (real cost)
  Grader B: glm-5.2-fp8 (free)

Pre-approved spend: ~$5.72 estimated (DeepSeek solver $3.65 + grading
$2.07). Safety-net hard stop at $9 new spend (just under the user's stated
$10 ceiling for this decision) -- not an interrupting $3-increment stop,
since this batch was explicitly pre-approved as a whole; $9 is a backstop
against the estimate being badly wrong, not a routine checkpoint.

Truncation/drift and slowdown checks kept from the GLM run, wall baseline
adjusted to DeepSeek's measured ~197s (head-to-head test).
"""
import json
import os
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, "src")

from data import MathNetEasy
from dumb_reranker import rerank_top10 as dumb_rerank_top10
from embed_labembed import embed_all as embed_fn
from eval import build_results, to_matrix
from llm_reranker import _HttpError, _read_env_var
from llm_solver_deepseek import call_solver as deepseek_call_solver
from spend_tracker import HardStopExceeded, SessionSpendGuard, record_call, spend_line

SEED = 42
N_PILOT = 250
CACHE_PATH = "utility_curve_cache/utility_curve_deepseek_cache.jsonl"

SOLVER_MODEL = "deepseek-v4-flash"
SOLVER_CAP = 32768
SOLVER_WORKERS = 6
WALL_BASELINE_S = 197
SLOWDOWN_WINDOW = 20
SLOWDOWN_FACTOR = 3.0
DRIFT_THRESHOLD_PTS = 15
DRIFT_MIN_N = 20

GRADER_A_MODEL = "gemini-3.1-flash-lite"
GRADER_A_KEY = _read_env_var("GEMINI_API_KEY")
GRADER_B_MODEL = "glm-5.2-fp8"
GRADER_MAX_TOKENS = 32
SPEND_GUARD_LIMIT = 9.0

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


def _parse_int_in_range(text, lo, hi):
    import re
    for raw in reversed(re.findall(r"\d+", text)):
        val = int(raw)
        if lo <= val <= hi:
            return val
    return None


def call_grader_a(prompt: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GRADER_A_MODEL}:generateContent"
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{url}?key={GRADER_A_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.0, "maxOutputTokens": GRADER_MAX_TOKENS}},
                timeout=120,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0].get("content", {})
                parts = content.get("parts")
                text = parts[0]["text"] if parts else ""
                usage = data.get("usageMetadata", {})
                return {"text": text, "prompt_tokens": usage.get("promptTokenCount", 0),
                        "completion_tokens": usage.get("candidatesTokenCount", 0)}
            raise _HttpError(resp.status_code, resp.text)
        except (_HttpError, requests.exceptions.RequestException) as e:
            status = getattr(e, "status", None)
            if attempt == 5 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(30, 2 ** attempt))


GLM_BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")


def call_grader_b(prompt: str) -> str:
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{GLM_BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={"model": GRADER_B_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.0, "max_tokens": GRADER_MAX_TOKENS},
                timeout=120,
            )
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            return msg.get("content") or msg.get("reasoning_content") or ""
        except requests.exceptions.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


ds = MathNetEasy.load(tier="hard")
all_query_ids = ds.sample_queries(500, seed=SEED)
query_ids = all_query_ids[:N_PILOT]
corpus_ids = list(ds.corpus.keys())
query_texts = {qid: ds.queries[qid] for qid in query_ids}
strict_qrels = {qid: ds.qrels_strict[qid] for qid in query_ids}

solve_index = json.load(open("data/solve_solution_index.json"))
graded_query_ids = [qid for qid in query_ids if qid in solve_index and solve_index[qid]["solutions_markdown"]]
print(f"[run] {len(graded_query_ids)}/{N_PILOT} queries have a matched reference solution", flush=True)

print("[run] loading labembed embeddings (cached)...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries")
c_cache = embed_fn(ds.corpus, cache_name="full_corpus")
query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)
dumb_reranked = dumb_rerank_top10(query_ids, results, query_texts, ds.corpus)

CONDITIONS = ["none", "dumb", "gold"]


def retrieved_id(condition, qid):
    if condition == "none":
        return None
    if condition == "dumb":
        return dumb_reranked[qid][0]
    return next(iter(strict_qrels[qid].keys()))


cache = {}
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH) as f:
        for line in f:
            d = json.loads(line)
            cache[(d["condition"], d["query_id"])] = d
print(f"[run] {len(cache)} cached from a previous run, resuming", flush=True)

write_lock = threading.Lock()
out_f = open(CACHE_PATH, "a")

guard = SessionSpendGuard(limit=SPEND_GUARD_LIMIT)
stop_event = threading.Event()

wall_times = deque(maxlen=SLOWDOWN_WINDOW)
cond_done = {c: 0 for c in CONDITIONS}
cond_capped = {c: 0 for c in CONDITIONS}
stats_lock = threading.Lock()


def check_drift():
    if cond_done["none"] < DRIFT_MIN_N:
        return
    none_rate = cond_capped["none"] / cond_done["none"] * 100
    for c in ("dumb", "gold"):
        if cond_done[c] >= DRIFT_MIN_N:
            rate = cond_capped[c] / cond_done[c] * 100
            if rate - none_rate > DRIFT_THRESHOLD_PTS:
                print(f"[run] *** WARNING: TRUNCATION DRIFT *** {c} truncation ({rate:.1f}%, n={cond_done[c]}) "
                      f"exceeds none ({none_rate:.1f}%, n={cond_done['none']}) by >{DRIFT_THRESHOLD_PTS}pts", flush=True)


def check_slowdown():
    if len(wall_times) < SLOWDOWN_WINDOW:
        return
    mean_recent = sum(wall_times) / len(wall_times)
    if mean_recent > WALL_BASELINE_S * SLOWDOWN_FACTOR:
        print(f"[run] *** WARNING: SLOWDOWN *** trailing-{SLOWDOWN_WINDOW} mean solver wall time "
              f"{mean_recent:.0f}s exceeds {SLOWDOWN_FACTOR}x baseline ({WALL_BASELINE_S}s)", flush=True)


def process(condition: str, qid: str) -> None:
    if stop_event.is_set():
        return
    key = (condition, qid)
    if key in cache:
        return
    cid = retrieved_id(condition, qid)
    prompt = SOLVE_PROMPT_NONE.format(problem=query_texts[qid]) if cid is None else \
        SOLVE_PROMPT_WITH_CONTEXT.format(related=ds.corpus[cid], problem=query_texts[qid])

    t0 = time.time()
    sol = deepseek_call_solver(prompt, max_tokens=SOLVER_CAP)
    dt = time.time() - t0
    solver_cost = record_call(SOLVER_MODEL, sol["prompt_tokens"], sol["completion_tokens"],
                               note=f"solver {condition}/{qid}")
    capped = sol["completion_tokens"] >= SOLVER_CAP or sol["finish_reason"] == "length"

    with stats_lock:
        wall_times.append(dt)
        cond_done[condition] += 1
        if capped:
            cond_capped[condition] += 1
        check_drift()
        check_slowdown()

    ref_solutions = solve_index[qid]["solutions_markdown"]
    reference = "\n\n---\n\n".join(ref_solutions)
    grade_prompt = GRADE_PROMPT.format(problem=query_texts[qid], reference=reference, candidate=sol["text"])

    ga = call_grader_a(grade_prompt)
    a_cost = record_call(GRADER_A_MODEL, ga["prompt_tokens"], ga["completion_tokens"],
                          note=f"grader-a {condition}/{qid}")
    score_a = _parse_int_in_range(ga["text"], 0, 7)
    correct_a = score_a is not None and score_a >= 6

    gb_text = call_grader_b(grade_prompt)
    score_b = _parse_int_in_range(gb_text, 0, 7)
    correct_b = score_b is not None and score_b >= 6

    record = {
        "condition": condition, "query_id": qid, "retrieved_id": cid,
        "solution": sol["text"], "completion_tokens": sol["completion_tokens"],
        "finish_reason": sol["finish_reason"], "capped": capped, "wall_s": dt,
        "solver_cost": solver_cost, "grader_a_cost": a_cost,
        "score_a": score_a, "correct_a": correct_a, "raw_a": ga["text"],
        "score_b": score_b, "correct_b": correct_b, "raw_b": gb_text,
        "agree": correct_a == correct_b,
    }
    with write_lock:
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        cache[key] = record

    try:
        guard.check()
    except HardStopExceeded as e:
        print(f"[run] *** HARD STOP *** {e}", flush=True)
        stop_event.set()


# Interleaved job order: round-robin across conditions per query, not
# condition-by-condition -- a stall yields a balanced partial dataset.
jobs = []
for qid in graded_query_ids:
    for cond in CONDITIONS:
        if (cond, qid) not in cache:
            jobs.append((cond, qid))

print(f"[run] {len(cache)} cached, {len(jobs)} to run (interleaved order)", flush=True)
print(f"[run] {spend_line()}", flush=True)

t_start = time.time()
done = 0
with ThreadPoolExecutor(max_workers=SOLVER_WORKERS) as ex:
    futures = {ex.submit(process, cond, qid): (cond, qid) for cond, qid in jobs}
    for fut in as_completed(futures):
        fut.result()
        done += 1
        if done % 30 == 0 or done == len(jobs):
            elapsed = time.time() - t_start
            print(f"[run] {done}/{len(jobs)} done ({elapsed/60:.1f}min elapsed, "
                  f"est. total {elapsed/max(done,1)*len(jobs)/60:.0f}min) -- {spend_line()}", flush=True)
        if stop_event.is_set():
            print(f"[run] stopping early at {done}/{len(jobs)} due to hard stop", flush=True)
            break
out_f.close()

print(f"\n=== DEEPSEEK UTILITY CURVE FINAL RESULTS (n={len(graded_query_ids)} graded queries/condition) ===")
for cond in CONDITIONS:
    items = [cache[(cond, qid)] for qid in graded_query_ids if (cond, qid) in cache]
    n = len(items)
    if n == 0:
        continue
    n_capped = sum(1 for r in items if r["capped"])
    n_correct_a = sum(1 for r in items if r["correct_a"])
    n_correct_b = sum(1 for r in items if r["correct_b"])
    n_agree = sum(1 for r in items if r["agree"])
    print(f"{cond:8s} n={n:3d}  truncated={n_capped}/{n} ({n_capped/n*100:.1f}%)  "
          f"acc_A={n_correct_a}/{n} ({n_correct_a/n*100:.1f}%)  acc_B={n_correct_b}/{n} ({n_correct_b/n*100:.1f}%)  "
          f"agree={n_agree}/{n} ({n_agree/n*100:.1f}%)")

all_items = [cache[(cond, qid)] for cond in CONDITIONS for qid in graded_query_ids if (cond, qid) in cache]
overall_agree = sum(1 for r in all_items if r["agree"]) / len(all_items) * 100 if all_items else 0
print(f"\nOverall grader agreement rate: {overall_agree:.1f}%")
print(f"Total wall time: {(time.time()-t_start)/3600:.2f}h")
print(spend_line())

summary = {
    cond: {
        "n": len([1 for qid in graded_query_ids if (cond, qid) in cache]),
        "truncated": sum(1 for qid in graded_query_ids if (cond, qid) in cache and cache[(cond, qid)]["capped"]),
        "correct_a": sum(1 for qid in graded_query_ids if (cond, qid) in cache and cache[(cond, qid)]["correct_a"]),
        "correct_b": sum(1 for qid in graded_query_ids if (cond, qid) in cache and cache[(cond, qid)]["correct_b"]),
        "agree": sum(1 for qid in graded_query_ids if (cond, qid) in cache and cache[(cond, qid)]["agree"]),
    }
    for cond in CONDITIONS
}
json.dump(summary, open("results/utility_curve_deepseek.json", "w"), indent=2)
