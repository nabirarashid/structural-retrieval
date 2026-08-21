# Appendix material for the LaTeX pass

Extracted directly from source scripts and result files, zero API calls, no edits to
`results/paper.md`. Sections A–D below match the request exactly: verbatim judge prompts, two
worked examples, per-cell failure-taxonomy tables, and the utility grader prompt.

---

## A. Verbatim judge prompts

All three judges (Gemini-j, GLM-j, Haiku-j) use **identical prompt wording within each domain** —
Haiku-j imports the same template object the other two judges use rather than reimplementing it
(`scripts/task2_haiku_reranker_full.py` does `from llm_reranker import PROMPT_TEMPLATE as
MATH_PROMPT_TEMPLATE` and `from step5_llm_reranker import PROMPT_TEMPLATE as TRAJ_PROMPT_TEMPLATE`),
so there is one terse prompt per domain, not three. Text below is the true string sent to the API
(Python's `\`-newline continuation in the source is a line-wrap artifact only; it inserts nothing
into the actual string — reproduced here as the continuous text that results).

`{anchor}` and `{candidates}` are `.format()`-substituted with the anchor problem/task text and the
10 numbered candidate texts before sending.

### A.1 Math domain — terse prompt (Gemini-j, GLM-j, Haiku-j)

Source: `src/llm_reranker.py`, `PROMPT_TEMPLATE`.

```
You are given a math competition problem ("ANCHOR") and 10 candidate problems, labeled 1 through 10. Exactly one candidate relies on the same underlying mathematical technique or method as the anchor -- the same core idea you would actually use to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared language, shared story framing (e.g. both about chessboards, or both in the same language), or shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to the anchor and still use a completely different technique, and a candidate can look nothing like the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Respond with ONLY the candidate number (1-10). No explanation, no other text.
```

### A.2 Math domain — chain-of-thought prompt (Gemini-j, GLM-j only; not used for Haiku-j)

Source: `src/llm_reranker.py`, `COT_PROMPT_TEMPLATE`. Same task framing as A.1, with an added
step-by-step reasoning instruction and a structured final-answer marker.

```
You are given a math competition problem ("ANCHOR") and 10 candidate problems, labeled 1 through 10. Exactly one candidate relies on the same underlying mathematical technique or method as the anchor -- the same core idea you would actually use to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared language, shared story framing (e.g. both about chessboards, or both in the same language), or shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to the anchor and still use a completely different technique, and a candidate can look nothing like the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Think step by step. First state the core technique needed to solve the ANCHOR. Then, for each candidate in turn, briefly note the technique it actually needs and whether it matches the anchor's. After going through all 10, conclude.

End your response with your final answer on its own line, in exactly this format:
FINAL ANSWER: <candidate number>
```

### A.3 Math domain — concise chain-of-thought variant (bonus; diagnostic-only, not one of the three headline judge runs)

Not part of the requested terse/CoT pair, but included since it's directly part of the CoT-truncation
story referenced in paper §4/§8: this variant was tested on a 30-query sample
(`scripts/test_glm_concise_cot.py`) after A.2 caused 63.3% of GLM-j's full CoT run to truncate
mid-reasoning (§8 incident 1); it never replaced A.2 in a full run. Source: `src/llm_reranker.py`,
`COT_CONCISE_PROMPT_TEMPLATE`.

```
You are given a math competition problem ("ANCHOR") and 10 candidate problems, labeled 1 through 10. Exactly one candidate relies on the same underlying mathematical technique or method as the anchor -- the same core idea you would actually use to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared language, shared story framing (e.g. both about chessboards, or both in the same language), or shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to the anchor and still use a completely different technique, and a candidate can look nothing like the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Think step by step, but be CONCISE -- this is a triage judgment, not a full solution write-up. State the core technique needed for the ANCHOR in one sentence. Then for each candidate, in one short phrase each, name its technique and say match or no match -- do not re-derive or fully solve any candidate. If you are torn between two candidates, pick the better one directly rather than re-litigating the comparison at length.

End your response with your final answer on its own line, in exactly this format:
FINAL ANSWER: <candidate number>
```

### A.4 Trajectory domain — terse prompt (Gemini-j, GLM-j, Haiku-j)

Source: `scripts/step5_llm_reranker.py`, `PROMPT_TEMPLATE`. No chain-of-thought variant exists for
this domain — CoT was only run in mathematics (confirmed: no CoT template anywhere in
`scripts/step5_llm_reranker.py`, and paper §5 never mentions it).

```
You are given an agent task instruction ("ANCHOR") and 10 candidate task trajectories, labeled 1 through 10. Exactly one candidate follows the same underlying PROCEDURE as the anchor -- the same transformation type (e.g. simple placement, clean-then-place, heat-then-place, cool-then-place, examine-with-light, two-object placement) -- even though it may involve a completely different object.

IGNORE which specific object is involved when deciding: matching object names, matching receptacles, or similar surface phrasing do NOT mean same procedure. A candidate can involve the exact same object as the anchor and still follow a different procedure, and a candidate can involve a completely different object and still follow the exact same procedure.

Focus only on: what sequence of actions / transformation type would you need to perform for each. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Respond with ONLY the candidate number (1-10). No explanation, no other text.
```

---

## B. Two worked examples

### B.1 Math domain — flipped-inequality near-miss (`sau_2012_bb6608`)

Source: `results/strict_misses_examples.md`, Example 1, reproduced verbatim.

**QUERY** (`sau_2012_bb6608`):
> Determine all positive integers $n$ such that the inequality
> $$\sqrt{x-1} + \sqrt{x-2} + \cdots + \sqrt{x-n} < x$$
> holds for every real number $x \ge n$.

**GOLD TARGET** (`sau_2012_bb6608::eq::easy`):
> Find every natural number $m \ge 1$ such that for any real number $x \ge m$, the inequality
> $$\sum_{k=1}^m \sqrt{x-k} < x$$
> is satisfied.

**Top-10 retrieved (embedding cosine similarity):**

| Rank | Score | ID | Note |
|---|---|---|---|
| 1 | 0.8599 | `sau_2012_bb6608::nm::0` | **WRONG top-1** — identical text, `<` flipped to `>` (opposite-direction inequality, different problem) |
| 2 | 0.8396 | `sau_2012_bb6608::nm::2` | sum replaced with product |
| 3 | 0.8311 | `sau_2012_bb6608::eq::medium` | sibling reformulation |
| 4 | 0.8208 | `sau_2012_bb6608::eq::easy` | **GOLD** — ranked 4th |
| 5 | 0.8157 | `sau_2012_bb6608::nm::1` | square roots changed to cube roots |
| 6-10 | 0.75-0.79 | various other-base items | unrelated problems, lower similarity |

(These are the exact numbers paper §4 cites: "a character-identical problem with a single flipped
inequality wins at cosine 0.860 while the true renamed-variable equivalent ranks fourth at 0.821.")

### B.2 Trajectory domain — SIBLING miss (`easy_9`)

`strict_misses_examples.md` covers the math domain only (verified: it contains exactly 3 examples,
all math), so no trajectory example could be copied from it. This one is newly extracted here,
directly and only from already-committed local files — `trajectory_reranker_cache/step5_llm_reranker_cache_n118.jsonl`
(GLM-j's chosen candidate on this query), `results/task1_expanded_tier_labels.json` (query text and
gold/SIBLING sets), and `results/agentinstruct_task_type_labels.json` (corpus task descriptions) —
zero API calls.

**QUERY** (`easy_9`, EASY tier, task_type 1 `pick_and_place_simple`, target object `remotecontrol`,
goal receptacle `sofa`):
> Place a remote control on the sofa

**GLM-j's chosen candidate — WRONG, a SIBLING miss** (`alfworld_286`):
> put a remotecontrol in armchair.

Same object (`remotecontrol`), same task type, different receptacle (`armchair` vs. the query's
`sofa`) — a literal same-object duplicate. This is exactly the SIBLING failure mode: the judge
picked the trajectory that shares the query's literal object instead of one requiring the same
procedure over a *different* object, which is what STRICT scoring requires.

**A representative STRICT gold trajectory** (`alfworld_118`, in `easy_9`'s STRICT gold set):
> put some newspaper on sofa.

Different object (`newspaper` vs. `remotecontrol`), same task type (`pick_and_place_simple`), same
receptacle (`sofa`) as the query — the structural match STRICT rewards, and the one the judge did
not pick.

---

## C. Per-cell failure-taxonomy tables (verbatim from `results/FINAL_NUMBERS.md`)

### C.1 Math domain, 4 cells (baseline retrieval failure structure, dominant miss categories)

Verbatim from `results/FINAL_NUMBERS.md` (§1, "Failure taxonomy"):

> **Failure taxonomy** (`strict_misses_examples.md`, `baseline_results.md`): dominant miss category
> `own_nm_near_miss` — Gemini Easy 420, Gemini Hard 420, Qwen3-DeepInfra Easy 446, Hard 446;
> `sibling_eq_variant` — Gemini Easy 13, Hard 74, Qwen3-DeepInfra Easy 7, Hard 50

| Embedder | Tier | `own_nm_near_miss` | `sibling_eq_variant` |
|---|---|---|---|
| Gemini-emb | Easy | 420 | 13 |
| Gemini-emb | Hard | 420 | 74 |
| Qwen-emb (DeepInfra) | Easy | 446 | 7 |
| Qwen-emb (DeepInfra) | Hard | 446 | 50 |

### C.2 Trajectory domain, 9 cells (LLM-reranker miss taxonomy: SIBLING/NEAR_MISS/OTHER/unparsed)

Verbatim from `results/FINAL_NUMBERS.md` (§2), Gemini-j/GLM-j (6 cells):

> Failure taxonomy, pooled misses (SIBLING/NEAR_MISS/OTHER/unparsed): labembed-Gemini
> 85.1%/13.5%/1.4%/0.0% (n=74 miss); labembed-GLM 88.3%/11.7%/0.0%/0.0% (n=60); gemini-Gemini
> 85.7%/14.3%/0.0%/0.0% (n=70); gemini-GLM 80.0%/14.0%/2.0%/4.0% (n=50); MiniLM-Gemini
> 74.3%/20.3%/5.4%/0.0% (n=74); MiniLM-GLM 72.3%/7.7%/9.2%/10.8% (n=65)

Verbatim from `results/FINAL_NUMBERS.md` (§2), Haiku-j (3 cells):

> - labembed-Qwen3-8B: Hit@1 17.8%→39.0%, share closed 46.3% [30.2,63.0], taxonomy (n_miss=72): SIBLING 84.7%, NEAR_MISS 11.1%, OTHER 2.8%, unparsed 1.4%
> - gemini-embedding-001: Hit@1 15.3%→39.8%, share closed 43.9% [30.4,57.9], taxonomy (n_miss=71): SIBLING 85.9%, NEAR_MISS 12.7%, OTHER 1.4%, unparsed 0.0%
> - MiniLM-L6-v2: Hit@1 9.3%→32.2%, share closed 48.2% [32.2,64.8], taxonomy (n_miss=80): SIBLING 75.0%, NEAR_MISS 17.5%, OTHER 5.0%, unparsed 2.5%

All 9 cells as one table:

| Embedder | Judge | n_miss | SIBLING | NEAR_MISS | OTHER | unparsed |
|---|---|---|---|---|---|---|
| labembed-Qwen3-8B | Gemini-j | 74 | 85.1% | 13.5% | 1.4% | 0.0% |
| labembed-Qwen3-8B | GLM-j | 60 | 88.3% | 11.7% | 0.0% | 0.0% |
| labembed-Qwen3-8B | Haiku-j | 72 | 84.7% | 11.1% | 2.8% | 1.4% |
| gemini-embedding-001 | Gemini-j | 70 | 85.7% | 14.3% | 0.0% | 0.0% |
| gemini-embedding-001 | GLM-j | 50 | 80.0% | 14.0% | 2.0% | 4.0% |
| gemini-embedding-001 | Haiku-j | 71 | 85.9% | 12.7% | 1.4% | 0.0% |
| MiniLM-L6-v2 | Gemini-j | 74 | 74.3% | 20.3% | 5.4% | 0.0% |
| MiniLM-L6-v2 | GLM-j | 65 | 72.3% | 7.7% | 9.2% | 10.8% |
| MiniLM-L6-v2 | Haiku-j | 80 | 75.0% | 17.5% | 5.0% | 2.5% |

---

## D. Utility grader prompt (verbatim)

Source: `scripts/run_utility_curve_deepseek.py`, `GRADE_PROMPT`. Used identically for both graders
(Grader A = `gemini-3.1-flash-lite`, Grader B = `glm-5.2-fp8`) — a single shared `grade_prompt`
string is built once per query/condition and passed to both `call_grader_a` and `call_grader_b`, so
there is one grader prompt, not two.

`{problem}`, `{reference}`, and `{candidate}` are `.format()`-substituted with the query text, the
concatenated reference solution(s), and the solver's candidate solution text before sending.

```
You are grading a candidate solution to a mathematics olympiad problem against a reference solution.

PROBLEM:
{problem}

REFERENCE SOLUTION:
{reference}

CANDIDATE SOLUTION:
{candidate}

Score the candidate solution's correctness on a scale from 0 to 7, where 7 means fully correct (or containing only minor errors that don't affect the core reasoning) and 0 means completely incorrect or no meaningful progress. Judge whether the candidate's mathematical reasoning and final conclusion are consistent with the reference, not writing style or presentation. The reference may be in a different language than the candidate -- grade the mathematical content, not the language.

Respond with ONLY the integer score (0-7). No explanation, no other text.
```
