# RAG quality-curve pilot ("Option A"): does bad retrieval hurt below the no-retrieval floor?

> ## ⚠️ CORRECTION (added after this doc was written) — solver output is systematically truncated
>
> **49.6% of the 528 cached solver answers behind the table below (262/528, across all 6
> conditions) hit `SOLVER_MAX_TOKENS=8192` mid-derivation, not at a natural stopping point.** This
> was discovered the same way the GLM-CoT-judge truncation was (see `llm_reranker_cot_full_comparison.md`)
> — by checking real per-response token counts against the cap, after the fact. The "verbosity
> doesn't explain it" check in this doc (§Results, point 2) compared mean solution length in
> *characters* across conditions and found them within ~8% of each other — but that check doesn't
> catch truncation, because a cut-off response and a complete response can be similar length in
> characters while one of them never reached a conclusion.
>
> **Per-condition truncation rate and its effect on accuracy** (truncated vs. complete answers,
> same condition): `none` 34/88 truncated (38.6%; accuracy 50.0% truncated vs. 66.7% complete),
> `dumb` 46/88 (52.3%; 65.2% vs. 81.0%), `baseline` 47/88 (53.4%; 51.1% vs. 75.6%),
> `glm_reranked` 44/88 (50.0%; 56.8% vs. 86.4%), `gemini_reranked` 44/88 (50.0%; 47.7% vs. 77.3%),
> `gold` 47/88 (53.4%; 46.8% vs. 82.9%). **Every condition's truncated answers score 15–35 points
> worse than its complete answers** — truncation is not neutral noise, it's suppressing the
> reported accuracy in every row of the table below, and context-bearing conditions truncate
> noticeably more (50–53%) than `none` (38.6%), which is itself a confound correlated with the
> experimental variable.
>
> **What this means for the table and conclusions below:** the six numbers in §Results are
> real outputs from a real pipeline, but roughly half of the underlying answers in every
> condition never reached a conclusion — so none of the absolute percentages, and none of the
> "not statistically distinguishable" verdict, should be read as a clean measurement of what these
> six conditions actually do. A fix (raising the cap, and/or an explicit concise-solving
> instruction — the same class of fix that worked for the GLM CoT judge) is being tested before
> any rerun. **Do not cite this doc's Results table as a settled finding; it is a diagnosed,
> not-yet-corrected data-quality problem, not a null result.**

## Opening: what MathNet already showed, and what it didn't test

MathNet-RAG (§4.3.3 of the paper) compares exactly three retrieval conditions: Zero-Shot (no
retrieval), Embed-RAG (`gemini-embedding-001` top-1), and Expert-RAG (human-paired gold). No
intermediate quality levels, and no condition designed to be deliberately bad.

But the paper's own results contain a below-baseline effect they didn't set out to find. Direct
quote: *"the table also shows that stronger retrieved context does not guarantee gains for every
model... Embed-RAG is less reliable and can fall below Zero Shot, as seen for Grok-4.1-Fast,
Gemini-3-Pro, and oLMO-3-Think"* under LLM grading. They treat this as an incidental
observation inside a three-point comparison, not as a designed experimental condition — there's no
retrieval condition in their design that's *worse* than realistic embedding retrieval, so they
never systematically characterize how far below baseline accuracy can go, or what it takes to get
there.

That's the opening this pilot targets: build a deliberately bad retrieval condition (not just a
mediocre one) and place it on the same quality ladder as no-retrieval, realistic-embedding, and
gold, to see whether the sub-baseline effect MathNet stumbled into is a general property of bad
retrieval, or a coincidence of their specific three-point design.

## Design

100 queries, hard tier, seed=42 (first 100 of this project's fixed 500-query sample). Six
conditions, all reranking/selecting from the same DeepInfra (Qwen3-Embedding-8B) top-10 candidate
pool so they form one coherent ladder rather than six unrelated setups:

| Condition | What it is | Expected quality |
|---|---|---|
| `none` | Zero-shot, no retrieved context | Floor / MathNet's control |
| `dumb` | Top-1 from the pure-lexical reranker (`src/dumb_reranker.py`) | Deliberately bad — this project already showed dumb reranking actively favors near-miss decoys over gold |
| `baseline` | Raw embedding cosine top-1, no reranking | Realistic mediocre retrieval, MathNet's Embed-RAG analog |
| `glm_reranked` | glm-5.2-fp8 CoT-judge top-1 | Improved retrieval |
| `gemini_reranked` | gemini-3.1-flash-lite CoT-judge top-1 | Improved retrieval |
| `gold` | True `::eq::hard` equivalent | Oracle, MathNet's Expert-RAG analog |

Solver: **glm-5.2-fp8** via the lab endpoint (free, unlimited). Grader: **gemini-3-flash-preview**
(see caveats below).

## Methodology deviations from MathNet's exact setup — stated up front, not discovered later

1. **Context is problem-only, not problem+solution.** MathNet-Retrieve's corpus (what gets
   retrieved) has no solution field at all — confirmed by inspecting the raw BEIR export and the
   `ShadenA/MathNet-Retrieve` HF dataset directly. Worse, corpus items are LLM-paraphrased
   reformulations of anchor problems (more heavily disguised at hard tier specifically), so even a
   fuzzy text-similarity join back to MathNet-Solve's official solutions would be least reliable
   exactly where this pilot is focused. Attaching a solution to an arbitrary retrieved candidate
   isn't something this project's data supports doing faithfully, so this pilot omits it. This is
   a real deviation from MathNet's Expert-RAG design (which paired retrieved problems with their
   official solutions) — the pilot tests "does the presence of a related problem help/hurt," not
   "does a related problem plus its worked solution help/hurt."

2. **Grader is gemini-3-flash-preview, not GPT-5.** No OpenAI API key is available in this
   project. The 0–7 scale, binarized at ≥6 = correct, is preserved exactly as MathNet's stated
   Solve rubric describes it. The judge model itself is substituted — same pattern as the
   reranker-judge work earlier in this project, documented rather than silently worked around.

3. **Reference solutions for the 100 query problems were obtained via a direct text join to
   MathNet-Solve**, not distributed with MathNet-Retrieve. Query IDs in MathNet-Retrieve are
   unsuffixed base IDs (unlike corpus items, which always carry an `::eq::`/`::nm::` suffix) —
   spot-checking confirmed queries are verbatim-or-near-verbatim original MathNet-Solve problems,
   unlike the deliberately paraphrased corpus items, so an exact/normalized text match against all
   ~27,800 downloaded MathNet-Solve rows (`ShadenA/MathNet`, per-country parquet shards) is
   reliable here specifically. **99/100 queries matched a MathNet-Solve row** (97 exact substring,
   2 via whitespace-normalized match); 1 query (`pol_33dd51`) had no match at all. Of the 99
   matches, **11 have an empty `solutions_markdown` field** (that MathNet-Solve row only carries
   `answers_markdown` — a final answer, no worked solution — likely an answer-type rather than
   proof-type problem), so no reference solution exists to grade against even though the join
   succeeded. Excluding those too, the pilot runs on **88 queries**, not 100.

4. **MathNet's own RAG-specific grading rubric (the "four LLM graders" + human expert protocol
   used for Table 5/7) is not fully restated in the paper text** — only the base Solve rubric
   (0–7, binarize at ≥6) is spelled out in detail. This pilot uses that base rubric as the closest
   documented approximation. Treat comparability to MathNet's *published RAG numbers specifically*
   as approximate, not exact — comparability to the *shape* of their finding (does retrieval
   quality change downstream accuracy, can it go below baseline) is what this pilot targets, not
   literal numeric parity with Table 5.

## Results

88 graded queries, solver glm-5.2-fp8, grader gemini-3-flash-preview (0–7, binarize ≥6).

| Condition | Correct | n | % correct |
|---|---|---|---|
| `none` (no retrieval) | 53 | 88 | 60.2% |
| `dumb` (deliberately bad) | 64 | 88 | **72.7%** |
| `baseline` (raw embedding top-1) | 55 | 88 | 62.5% |
| `glm_reranked` | 63 | 88 | 71.6% |
| `gemini_reranked` | 55 | 88 | 62.5% |
| `gold` (oracle) | 56 | 88 | 63.6% |

At face value this is not the shape anyone would have predicted: `dumb` — the condition built to be
deliberately bad, retrieving the candidate a pure-lexical scorer likes best from the same top-10
pool — scores *highest*, above even `gold`. `none` is the *lowest* of all six. Before reading
anything into that, three checks:

**1. None of it is statistically distinguishable at this sample size.** Paired McNemar's exact test
(same 88 queries across every condition, so this is the right test, not an independent-samples
comparison):

| Comparison | A-only | B-only | McNemar p |
|---|---|---|---|
| none vs dumb | 9 | 20 | 0.061 |
| none vs glm_reranked | 6 | 16 | 0.053 |
| glm_reranked vs gemini_reranked | 13 | 5 | 0.096 |
| dumb vs gold | 18 | 10 | 0.185 |
| dumb vs baseline | 18 | 9 | 0.122 |
| none vs baseline | 11 | 13 | 0.839 |
| none vs gold | 9 | 12 | 0.664 |
| baseline vs gold | 10 | 11 | 1.000 |

Nothing clears the conventional 0.05 threshold. Two comparisons (none vs dumb, none vs
glm_reranked) sit just above it and are worth a larger follow-up, but as this pilot stands, **every
pairwise gap in the six-point table is consistent with noise.** A 12.5-point spread (60.2% vs
72.7%) looks large as a bar chart and isn't backed by significance at n=88.

**2. Verbosity doesn't explain it.** Mean solution length by condition (chars): none 12,391 / dumb
13,345 / baseline 13,427 / glm_reranked 12,459 / gemini_reranked 12,878 / gold 13,272 — all within
~8% of each other. If the grader were just rewarding longer answers, `dumb`'s lead wouldn't track
with it being *shorter* on average than `baseline` and `gold`, both of which scored lower.

**3. Grader score distribution looks bimodal-but-sane, with a caveat.** Score histograms per
condition are dominated by 0 and 7 (e.g. dumb: 0×15, 7×64, everything else in single digits;
similar shape everywhere) — the grader is making decisive calls, not scattering noise across the
0–7 range, which argues against `disable_thinking=True` having produced garbage judgments. The
caveat: **6 is almost never used** (0 across five of six conditions, 3 for gold) — in practice this
grader is closer to a binary correct/incorrect call than a genuine 0–7 scale, meaning the
"binarize at ≥6" step is doing very little work beyond "is it a 7." A judge with visible reasoning
enabled might grade more granularly, at the cost of the token-budget/reliability problems that
motivated disabling it here (see methodology notes above).

## Answering the question: does accuracy fall below the no-retrieval baseline at the low end?

**Not established by this pilot, in either direction.** The point estimate for the one condition
built to be deliberately bad (`dumb`) is numerically *above* `none`, not below it — the opposite of
what MathNet's incidental Embed-RAG-below-Zero-Shot observation would predict for a worse-than-baseline
condition. But that specific comparison (none vs dumb) sits at p=0.061 — a trend, not a finding.
The honest summary: **this pilot did not reproduce a below-baseline effect, and did not clearly rule
one out either.** At n=88 split six ways, the design doesn't have the power to resolve a 10-15 point
gap from noise. Scaling to the full 500-query sample (or concentrating power on fewer, sharper
comparisons — e.g. just `none` vs `dumb` at n=300+ — instead of six simultaneous conditions) would
be the natural next step if this line is worth pursuing further, rather than treating the current
six-point spread as a real curve.

## Reproducibility

`scripts/build_solve_solution_index.py` (one-time query→solution join, `data/solve_solution_index.json`),
`scripts/run_rag_pilot.py` (main pilot, resumable via `rag_pilot_cache.jsonl`). Raw output:
`results/rag_pilot.json`.
