# Full comparison: terse vs. CoT prompt, both judges, all 4 configs

> ## ⚠️ CORRECTION (added after this doc was written) — GLM-CoT numbers are compromised
>
> **63.3% of the 2,000 responses in the GLM CoT full run (`llm_reranker_full_cot_glm.json`) hit
> the 6,144-token cap** — truncated mid-deliberation, not concluded. The parser still extracted
> *a* number from almost all of them (only 1 of 1,266 truncated responses failed to parse
> entirely), which is why this doc reports "0 unparsed" and reads the run as clean data. **Parsed
> successfully is not the same as reached a real conclusion.** A representative example: one cached
> response ends mid-sentence — *"...Since the problem says 'exactly one,' and both 4 and 5"* — and
> the parser extracted "5" as if that were a decided answer.
>
> **Do not cite the GLM-CoT numbers below** — the share-of-gap-closed figures for GLM CoT and the
> GLM-CoT contamination test results (both candidate sets) are downstream of this truncation and
> should be treated as unreliable, not as a real measurement of GLM's judgment quality.
>
> **Unaffected, still citable:** all Gemini CoT numbers (max 978/4,096 tokens observed, never
> truncated) and all terse-prompt numbers for both judges (GLM terse median was 1 token — genuinely
> terse, not truncated-long).
>
> See `results/glm_cot_diagnostic.md` for the same correction and the follow-up investigation
> (concise-reasoning prompt + larger token budget, tested at small scale before deciding whether to
> rerun or report GLM as unusable for CoT-style judging).

Completes the second-judge / prompt-artifact investigation. GLM's CoT run took much longer than
the 50-query diagnostic implied (hard-tier configs ran ~3+ hours each vs. easy-tier's ~1 hour,
apparently hard problems draw out much longer reasoning traces) — total wall time across all 4
GLM CoT configs was ~8.7 hours, vs. Gemini CoT's ~33 minutes total.

## Share of recoverable gap closed — all 8 combinations

| Config | Gemini, terse | Gemini, CoT | GLM, terse | GLM, CoT |
|---|---|---|---|---|
| Gemini-embed / Easy | 20.6% | **44.7%** | 10.1% | 6.3% |
| Gemini-embed / Hard | **44.4%** | 22.7% | 10.5% | 16.6% |
| DeepInfra-embed / Easy | 35.6% | **55.4%** | 12.0% | 14.3% |
| DeepInfra-embed / Hard | 41.0% | 26.7% | 18.1% | **21.9%** |

Every one of the 8 judge×prompt×config combinations is positive — the direction of the core
finding (LLM reranking beats raw embedding Hit@1 when told to ignore surface form) is fully
robust. But magnitude is highly judge- and prompt-dependent, and CoT doesn't uniformly help:

- **Gemini**: CoT roughly *doubles* easy-tier gains (20.6%→44.7%, 35.6%→55.4%) but *cuts hard-tier
  gains by a third to a half* (44.4%→22.7%, 41.0%→26.7%). Reasoning traces read as high-quality
  (correct, detailed derivations), but on hard tier the model sometimes explicitly favors a
  candidate it itself calls "a direct restatement" of the anchor — plausible mechanism: giving the
  model room to reason may let it drift toward the most recognizable/lexically-similar sibling
  (an easier reformulation of the same base problem) rather than the harder-to-recognize hard-tier
  variant, i.e. CoT may reintroduce exactly the surface-form bias the prompt is trying to suppress.
  Unconfirmed — would need per-query auditing of which suffix got picked to establish this
  causally.
- **GLM**: CoT helps 3 of 4 configs modestly (both hard-tier configs, deepinfra-embed/easy) but
  *hurts* gemini-embed/easy (10.1%→6.3%). GLM's absolute numbers remain well below Gemini's in
  every single comparable cell, terse or CoT.

**Verdict on reranking gains, full picture**: direction replicates across every judge/prompt
combination tested (8/8 positive) — strong, robust evidence this isn't an artifact of one model or
one prompt. Magnitude does not replicate at all; it swings by 2-4x depending on judge, and by up
to 2x depending on prompt within the same judge, occasionally reversing which tier benefits more.
Report the direction as solid; never report a specific magnitude number as "the" effect size
without naming the exact judge and prompt it came from.

## Contamination: pooled well_known(n=57) vs rest(n=443), hard tier, all 8 combinations

| Candidate set | Judge | Prompt | gap (pts) | two-prop-z p | Fisher exact p |
|---|---|---|---|---|---|
| Gemini-embed | gemini-3.1-flash-lite | terse | +19.8 | **0.0011** | **0.0018** |
| Gemini-embed | gemini-3.1-flash-lite | cot | +13.5 | **0.0038** | **0.0091** |
| Gemini-embed | glm-5.2-fp8 | terse | +1.4 | 0.676 | 0.560 |
| Gemini-embed | glm-5.2-fp8 | cot | +3.5 | 0.393 | 0.462 |
| DeepInfra-embed | gemini-3.1-flash-lite | terse | +8.1 | 0.040 | 0.073 |
| DeepInfra-embed | gemini-3.1-flash-lite | cot | +9.5 | **0.0033** | **0.0087** |
| DeepInfra-embed | glm-5.2-fp8 | terse | +5.6 | 0.037 | 0.054 |
| DeepInfra-embed | glm-5.2-fp8 | cot | +2.7 | 0.355 | 0.318 |

**This changes the read from the terse-only data.** Earlier (terse-only), the DeepInfra-embed gap
appearing under both judges at similar magnitude looked like genuine cross-model corroboration.
With the full CoT data in hand, that doesn't hold up: GLM's DeepInfra-embed result actually gets
*weaker* under CoT (+5.6pt→+2.7pt, p rises from borderline to clearly non-significant), while
Gemini's gets *stronger* under CoT on the same candidate set (+8.1pt→+9.5pt, now solidly
significant). Across all 4 GLM combinations (2 candidate sets × 2 prompts), none reach
significance (p ranges 0.32–0.68). Across all 4 Gemini combinations, all reach significance
(p ranges 0.001–0.009).

**Superseded by the framing below** — the "0/4 GLM-judge combinations significant" claim in this
paragraph only holds if the 2 GLM-CoT rows are counted, and those are exactly the rows compromised
by the 63.3%-truncation bug (see correction banner at top). ~~this well_known-recognition effect is
specific to the Gemini judge and does not appear under the GLM judge, under any prompt or candidate
set tested~~. Kept for the record of how the read evolved; do not cite this paragraph's verdict.

**Final framing (decided 2026-08-11 — GLM CoT will not be rerun, 8.7hr cost not justified for a
secondary question; GLM terse is used as the cross-judge data point instead):** Contamination is
found **robustly under the Gemini judge across both prompts** (4/4 combinations significant, p
0.001–0.009). Restricted to the only uncompromised GLM data — the terse-prompt rows — **GLM shows
the same direction in both candidate sets**, reaching borderline significance on DeepInfra-embed
(+5.6pt, two-prop-z p=0.037, Fisher p=0.054) but not on Gemini-embed (+1.4pt, p=0.676/0.560). GLM
CoT is excluded from the verdict entirely, not folded in as a null result — its 2 rows above are
truncation artifacts, not evidence of absence. Report as: **contamination found robustly under
Gemini across both prompts; GLM terse shows the same direction at borderline significance; GLM CoT
excluded due to 63.3% truncation.** Do not report "absent under GLM" — that conflates a broken
measurement with a negative one.

## Reproducibility

`scripts/run_llm_reranker_full_cot.py --judge {gemini,glm}`, seed=42, `COT_PROMPT_TEMPLATE`,
temperature=0.0. Raw output: `results/llm_reranker_full_cot_gemini.json`,
`results/llm_reranker_full_cot_glm.json`. Per-query responses cached in
`llm_reranker_cache/cot_{gemini,glm}_{provider}_{tier}.jsonl`.
