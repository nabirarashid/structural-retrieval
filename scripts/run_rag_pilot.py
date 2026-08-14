"""RAG quality-curve pilot ("Option A"): does downstream solving accuracy fall
BELOW the no-retrieval baseline at the low end of retrieval quality? MathNet
only tested none/embedding/gold; this fills in the middle with two reranked
points and a deliberately-bad point (dumb lexical reranker, already shown
elsewhere in this project to actively favor near-miss decoys).

100 queries (hard tier, seed=42, first 100 of the fixed 500-query sample),
six retrieval conditions, all reranking/selecting from the SAME DeepInfra
(Qwen3-Embedding-8B) top-10 candidate pool so they form one coherent ladder:

  none              -- zero-shot, no retrieved context (MathNet's control)
  dumb              -- dumb_reranker.py's top-1 (pure lexical; a genuinely
                        bad-retrieval condition MathNet never tested)
  baseline          -- raw embedding cosine top-1, no reranking
  glm_reranked      -- glm-5.2-fp8 CoT-prompt judge's top-1 pick
  gemini_reranked   -- gemini-3.1-flash-lite CoT-prompt judge's top-1 pick
  gold              -- the true ::eq::hard equivalent (oracle)

Context is problem-only (no solution) -- see results/rag_pilot.md for why:
MathNet-Retrieve corpus items are LLM-paraphrased and have no reliable join
back to MathNet-Solve's official solutions, so attaching a solution to an
arbitrary retrieved candidate isn't something we can do faithfully. This
still tests the core question (does retrieval quality shape downstream
accuracy, can bad retrieval hurt below baseline) -- just without the
solution-augmentation ingredient from MathNet's exact Expert-RAG setup.

Solver: glm-5.2-fp8 via the lab endpoint (free, unlimited).
Grader: gemini-3-flash-preview (GPT-5 unavailable -- no OpenAI key in this
project), 0-7 score binarized at >=6, mirroring MathNet's stated Solve rubric
as closely as an available judge model allows. MathNet's RAG-specific human/
4-LLM-grader rubric isn't fully restated in the paper text, so exact
comparability to their published RAG numbers is approximate, not exact.
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
N_PILOT = 100
RESULTS_DIR = Path("results")
LLM_CACHE_DIR = Path("llm_reranker_cache")
SOLVE_INDEX_PATH = Path("data/solve_solution_index.json")
PILOT_CACHE_PATH = Path("rag_pilot_cache.jsonl")

SOLVER_MODEL = "glm-5.2-fp8"
SOLVER_BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")
SOLVER_MAX_TOKENS = 8192
SOLVER_WORKERS = 6

GRADER_MODEL = "gemini-3-flash-preview"
# gemini-3-flash-preview is a thinking model that (a) can burn its whole token
# budget on hidden thought tokens before writing visible output, and (b)
# doesn't reliably respect a terse "answer only" instruction even when it
# does write visible text -- both confirmed via direct API tests, including
# one grading call that ran past 2048 tokens of open-ended reasoning without
# ever stating a score. Fix: disable_thinking=True (thinkingConfig.
# thinkingBudget=0) below, which produced a clean single-token reply in
# testing. Small budget is fine now that thinking is off.
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


# ---- assemble the six conditions ----

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
print(f"[pilot] {len(graded_query_ids)}/{N_PILOT} queries have a matched reference solution -- "
      f"only these are used (excluded: {sorted(set(query_ids) - set(graded_query_ids))})", flush=True)

print("[pilot] loading DeepInfra embeddings (cached)...", flush=True)
q_cache = embed_fn(query_texts, cache_name="full_queries")
c_cache = embed_fn(ds.corpus, cache_name="full_corpus")
query_matrix = to_matrix(query_ids, q_cache)
corpus_matrix = to_matrix(corpus_ids, c_cache)
results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)

dumb_reranked = dumb_rerank_top10(query_ids, results, query_texts, ds.corpus)


def load_choices(cache_name: str) -> dict[str, str | None]:
    path = LLM_CACHE_DIR / f"{cache_name}.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["query_id"]] = d["chosen_id"]
    return out


glm_choices = load_choices("cot_glm_deepinfra_hard")
gemini_choices = load_choices("cot_gemini_deepinfra_hard")

CONDITIONS = ["none", "dumb", "baseline", "glm_reranked", "gemini_reranked", "gold"]


def retrieved_id(condition: str, qid: str) -> str | None:
    if condition == "none":
        return None
    if condition == "dumb":
        return dumb_reranked[qid][0]
    if condition == "baseline":
        return max(results[qid].items(), key=lambda kv: kv[1])[0]
    if condition == "glm_reranked":
        return glm_choices.get(qid)
    if condition == "gemini_reranked":
        return gemini_choices.get(qid)
    if condition == "gold":
        return next(iter(strict_qrels[qid].keys()))
    raise ValueError(condition)


# ---- cache (resumable) ----

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
        if done % 20 == 0 or done == len(jobs):
            print(f"[pilot] {done}/{len(jobs)} done ({time.time()-t0:.0f}s elapsed)", flush=True)
out_f.close()

# ---- summarize ----

summary = {}
for cond in CONDITIONS:
    n = len(graded_query_ids)
    n_correct = sum(1 for qid in graded_query_ids if cache[(cond, qid)]["correct"])
    n_unparsed = sum(1 for qid in graded_query_ids if cache[(cond, qid)]["score"] is None)
    summary[cond] = {"n": n, "n_correct": n_correct, "pct_correct": n_correct / n * 100, "n_unparsed_grade": n_unparsed}

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "rag_pilot.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== RAG PILOT SUMMARY (n=%d graded queries) ===" % len(graded_query_ids))
for cond in CONDITIONS:
    s = summary[cond]
    print(f"{cond:16s}  {s['n_correct']:3d}/{s['n']:3d}  ({s['pct_correct']:5.1f}%)  unparsed_grade={s['n_unparsed_grade']}")
