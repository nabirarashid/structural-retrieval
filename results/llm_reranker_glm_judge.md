# Second judge: glm-5.2-fp8 — does it reproduce the gemini-3.1-flash-lite results?

## Setup

Identical protocol to the gemini-3.1-flash-lite run: same prompt (unedited), same 500 queries
(seed=42), same 4 configs, temperature 0.0. Only the judge model changed, via the `JudgeBackend`
swap built for exactly this purpose. Endpoint: an internal MIT CSAIL lab GPU host (address withheld,
not publicly reachable), served via sglang, confirmed via `/v1/models` to be serving `glm-5.2-fp8`.
URL lives in `.env` as `MANTIS_LLM_BASE_URL`, never hardcoded in any published file. No API key
required (dummy Bearer accepted).

**A behavioral quirk found before running at scale, and worth stating up front because it bears on
how to read the results below:** glm-5.2-fp8 is a reasoning model — it returns its answer in a
`reasoning_content` field, separate from `content`, which stayed empty on every call regardless of
instructions (`OpenAICompatJudgeBackend` was updated to fall back to `reasoning_content` when
`content` is empty). But on this specific task, the model used almost no visible reasoning: ~2
completion tokens per call, versus 288 reasoning tokens on an unrelated riddle sanity-check in the
same session. It answered near-instantly (~1.4–2.7s/call) rather than deliberating. So this is not
a case of the model running out of budget mid-thought (max_tokens was set generously, to 4096) —
it appears to answer this specific task quickly regardless of budget. This means the comparison
below is judge-model vs. judge-model, but it is *not* a clean "more reasoning effort vs. less"
comparison — GLM's weaker performance here (see below) could reflect genuinely weaker judgment on
this task, or could reflect that it isn't engaging its reasoning capability the way the prompt
intends. Both are plausible; this run can't distinguish them. Also fixed along the way: the retry
logic only caught HTTP-error responses, not connection-level failures — a transient read-timeout
crashed the deepinfra/hard config partway through (resumed cleanly from cache, no data lost, only
~10 min of reruns).

## Question 1: does GLM reproduce the reranking gains?

**Yes in direction, no in magnitude.** All four configs are positive — a real, reproducible signal
that *something* about ignoring surface form recovers hits an embedding-only ranking misses — but
GLM's share of the recoverable gap closed is 2–4x smaller than gemini-3.1-flash-lite's in every
config.

| Config | gemini-3.1-flash-lite | glm-5.2-fp8 | Ratio |
|---|---|---|---|
| Gemini-embed / Easy | 20.6% | 10.1% | 0.49x |
| Gemini-embed / Hard | 44.4% | 10.5% | 0.24x |
| DeepInfra-embed / Easy | 35.6% | 12.0% | 0.34x |
| DeepInfra-embed / Hard | 41.0% | 18.1% | 0.44x |

**Honest read:** the direction of the effect (LLM reranking beats raw embedding similarity when
told to ignore surface form) replicates cleanly across two independently-sourced judge models —
that's real corroboration that this isn't a gemini-3.1-flash-lite-specific artifact. But the
*size* of the effect is clearly judge-dependent, and given the reasoning-effort caveat above, it's
not established whether that's because GLM is worse at identifying shared mathematical technique,
or because this deployment isn't actually using its reasoning capability on this task the way the
low completion-token counts suggest. Report the direction as reproduced; do not report the
magnitude as reproduced.

## Question 2: does GLM reproduce the well_known > regional/other contamination pattern?

**Partially — present for DeepInfra-embedding candidates under both judges, absent for
Gemini-embedding candidates under GLM.** Pooled well_known (n=57) vs rest (n=443), hard tier only
(same grouping as the gemini-judge pooled test):

| Candidate set | Judge | well_known Hit@1 | rest Hit@1 | gap (pts) | two-prop-z p | Fisher exact p |
|---|---|---|---|---|---|---|
| Gemini-embed | gemini-3.1-flash-lite | 42.1% (24/57) | 22.4% (99/443) | +19.8 | **0.0011** | **0.0018** |
| Gemini-embed | glm-5.2-fp8 | 7.0% (4/57) | 5.6% (25/443) | +1.4 | 0.676 | 0.560 |
| DeepInfra-embed | gemini-3.1-flash-lite | 15.8% (9/57) | 7.7% (34/443) | +8.1 | 0.040 | 0.073 |
| DeepInfra-embed | glm-5.2-fp8 | 8.8% (5/57) | 3.2% (14/443) | +5.6 | 0.037 | 0.054 |

**Honest read:** this is a genuinely mixed result, not a clean yes/no.

- On the Gemini-embedding candidate set, the strong well_known effect found with the
  gemini-3.1-flash-lite judge (p=0.001, the most statistically solid contamination result in this
  project) essentially **disappears** with the GLM judge (gap shrinks from +19.8pt to +1.4pt,
  p=0.68 — indistinguishable from noise). If this were the only comparison, the honest conclusion
  would be "the effect was judge-specific."
- On the DeepInfra-embedding candidate set, *both* judges show a well_known > rest gap of similar,
  borderline-significant size (+8.1pt / p=0.04–0.07 for Gemini judge, +5.6pt / p=0.04–0.05 for GLM
  judge) — genuine, if modest, cross-judge agreement.

So the contamination effect is not purely a one-judge artifact (it shows up under both judges on
one of the two candidate sets), but it's also not a robustly-reproduced universal effect (it's
much weaker or absent for the other candidate set under the second judge). The most defensible
statement: **there is cross-judge-corroborated evidence of a well_known recognition effect
specifically on the DeepInfra-embedding candidate set at hard tier; the much stronger effect seen
on the Gemini-embedding candidate set with the gemini-3.1-flash-lite judge does not clearly survive
switching judges, and should not be reported as judge-independent.**

## Combined verdict

| Question | Answer |
|---|---|
| Does a second, independent judge reproduce the reranking gains? | Direction: yes. Magnitude: no — GLM's gains are 2–4x smaller across all 4 configs. |
| Does a second, independent judge reproduce the contamination pattern? | Partially — corroborated on the DeepInfra-embedding candidate set, not on the Gemini-embedding candidate set. |

Neither result should be reported as fully judge-independent. The reranking-gains direction is the
most solid cross-judge finding in this project; the contamination pattern is real evidence, not
noise, but weaker and more conditional than the single-judge numbers suggested on their own — this
is exactly the kind of overclaim a second judge is supposed to catch.

## Reproducibility

`scripts/run_llm_reranker_full_glm.py`, seed=42, `glm-5.2-fp8` @ temperature=0.0 via
`MANTIS_LLM_BASE_URL`, identical prompt to both prior runs. Raw output:
`results/llm_reranker_full_glm.json`. Per-query judge responses cached verbatim in
`llm_reranker_cache/full_glm_{provider}_{tier}.jsonl`.
