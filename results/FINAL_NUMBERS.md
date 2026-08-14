# Final numbers digest

Flat, citable list of every number this project's paper draft would cite, one line each, with a
95% CI where one was computed. Source file/script named for every line so a number can be checked
back against raw output. Compiled 2026-08-14, after the last planned experiment (the n=118
trajectory reranker, `results/task_traj_reranker_n118.json`) landed. **This digest is the
authoritative numbers source — where it disagrees with `RESULTS_SUMMARY.md` or
`RETRIEVAL_WRITEUP.md`, this digest wins; discrepancies are listed immediately below.**

Note on scope: `paper_draft_v3.md`, referenced as already existing, was not found anywhere in this
repository or searched locations — this digest was compiled from `results/` and the project's raw
caches directly, not reconciled against that file. If it exists elsewhere, check its numbers against
this digest before using it.

## Discrepancies found

- **`paper_draft_v3.md` mislabels the MiniLM validation-gate domain — this digest corrects it.**
  The paper's §3 states "A MiniLM gate for the mathematics domain failed (MAP 0.630 versus the
  published 0.795)... it appears only in the trajectory domain, whose gate it passed." Only one
  MiniLM gate file exists anywhere in this codebase (`results/minilm_validation_gate.json`,
  `scripts/validate_minilm_reproduction.py`), and its own docstring, query IDs (`easy_1`..`hard_15`,
  the original-40 trajectory set), and target numbers (0.7945/0.842/0.746/0.791 overall/E/M/H) are
  unambiguously the **trajectory** paper's own published Table 2 MAP, not MathNet's. There is no
  math-domain MiniLM gate file anywhere in this repository. Likely a sentence-construction slip in
  the paper draft (both numbers it cites belong to one trajectory-domain check, not two checks in
  two domains) — corrected below and moved to the Trajectories section, where it belongs and where
  it was in fact adopted (MiniLM-L6-v2 is one of the three trajectory embedders used throughout §2).
  **This is exactly the kind of check this digest exists to catch — flag to the author before the
  LaTeX pass.**
- Everything else: the math-domain and Task 2A/2B/2C numbers below were re-pulled directly from
  their source JSON files and match `RESULTS_SUMMARY.md`/`RETRIEVAL_WRITEUP.md` exactly (both were
  written from the same source files). §3 Utility has no prior writeup to disagree with
  (`RETRIEVAL_WRITEUP.md` explicitly excludes it; no RAG/utility-curve `.md` was ever written for
  the final DeepSeek-solver run) — its numbers are new, not discrepant. One number in §3 is not
  stated anywhere else: the final utility-curve run's solver truncation rate is **31%**
  (64–66/210 per condition) even at a 32,768-token cap — down from 50% at 16,384 and 80% at 8,192
  on the earlier pretests, but **not resolved**. Accuracy numbers in §3 should be read with this
  caveat, consistent with this project's standing truncation-disclosure rule.

---

## 1. Math domain (MathNet-Retrieve, n=500, seed=42)

**Baseline retrieval** (`results/baseline_{gemini,deepinfra}[_hard].json`; CIs: `task2b_bootstrap_cis.json`):
- Gemini-embedding-001, Easy: STRICT Hit@1 12.2%, Hit@5 89.8% [87.0,92.4], Hit@10 97.6% [96.2,98.8]
- Gemini-embedding-001, Hard: STRICT Hit@1 **0.0%** [0.0,0.0], Hit@5 10.0% [7.4,12.8], Hit@10 55.4% [51.0,59.8]
- Qwen3-Embedding-8B (DeepInfra), Easy: STRICT Hit@1 8.6%, Hit@5 86.8% [83.8,89.6], Hit@10 95.2% [93.2,97.0]
- Qwen3-Embedding-8B (DeepInfra), Hard: STRICT Hit@1 **0.0%** [0.0,0.0], Hit@5 2.8% [1.4,4.4], Hit@10 21.0% [17.6,24.6]
- LENIENT Hit@10: 100% (Gemini), 99.0% (DeepInfra) — both tiers, since LENIENT is tier-invariant
- Gate check vs. paper Table 4 (Easy, Gemini): ours 12.2/89.8/97.6 vs. paper's 11.36/90.68/96.93 (Strict R@1/R@5/R@10) — within ~1pt

**Failure taxonomy** (`strict_misses_examples.md`, `baseline_results.md`): dominant miss category
`own_nm_near_miss` — Gemini Easy 420, Gemini Hard 420, Qwen3-DeepInfra Easy 446, Hard 446;
`sibling_eq_variant` — Gemini Easy 13, Hard 74, Qwen3-DeepInfra Easy 7, Hard 50

**Dumb (lexical) reranker control** (`dumb_reranker_control.json`/`.md`) — hurts or is flat, never helps:
- Gemini Easy: 12.2%→4.4% (**−9.1%** share of gap)
- Gemini Hard: 0.0%→0.2% (+0.4%)
- Qwen3-DeepInfra Easy: 8.6%→4.4% (**−4.8%**)
- Qwen3-DeepInfra Hard: 0.0%→0.0% (0.0%)

**Lexical distance check** (`lexical_distance_check.json`/`.md`) — anchor↔gold vs. anchor↔false-positive similarity, share where FP is more similar:
- Gemini Easy: n=439, gold 0.652, FP 0.908, FP-more-similar **95.7%**
- Gemini Hard: n=500, gold 0.367, FP 0.883, FP-more-similar **99.8%**
- Qwen3-DeepInfra Easy: n=457, gold 0.656, FP 0.911, FP-more-similar **95.2%**
- Qwen3-DeepInfra Hard: n=500, gold 0.367, FP 0.892, FP-more-similar **99.4%**

**LLM reranker, terse prompt, share of recoverable gap closed — all 8 cells, with CIs**
(`llm_reranker_full.json`, `llm_reranker_full_glm.json`; CIs `task2b_bootstrap_cis.json`):
- Gemini judge / Gemini-embed / Easy: 20.6% [14.7,26.3]
- Gemini judge / Gemini-embed / Hard: 44.4% [37.5,51.3]
- Gemini judge / DeepInfra-embed / Easy: 35.6% [29.9,41.1]
- Gemini judge / DeepInfra-embed / Hard: 41.0% [29.5,53.3]
- GLM judge / Gemini-embed / Easy: 10.1% [5.0,15.0]
- GLM judge / Gemini-embed / Hard: 10.5% [6.9,14.4]
- GLM judge / DeepInfra-embed / Easy: 12.0% [7.3,16.6]
- GLM judge / DeepInfra-embed / Hard: 18.1% [10.5,26.7]

**LLM reranker, Gemini CoT prompt — 4 cells, no CI computed** (`llm_reranker_full_cot_gemini.json`,
`llm_reranker_cot_full_comparison.md`; Task 2B explicitly scoped CoT numbers out of the CI pass):
- Gemini-embed / Easy: 44.7%
- Gemini-embed / Hard: 22.7%
- DeepInfra-embed / Easy: 55.4%
- DeepInfra-embed / Hard: 26.7%

**GLM-CoT exclusion** (`glm_cot_diagnostic.md`, `llm_reranker_cot_full_comparison.md` correction
banners) — **63.3% (1,266/2,000) of GLM CoT responses truncated**, not concluded; parser still
extracted a candidate from all but 1 of the 1,266 truncated responses ("parses cleanly" ≠
"concluded"). Concise-reasoning-prompt + 16,384-token-cap follow-up: 10.0% (3/30) on a 30-query
sample — real improvement, never scaled to a full rerun (would cost ~8.7h wall time for a secondary
question). GLM-CoT numbers are excluded from every citable result in this project.

**Contamination gap, well_known (n=57) vs. rest (n=443), hard tier, terse prompt, with bootstrap CIs**
(`results/task2c_contamination_cis.json`):
- Gemini judge / Gemini-embed: well_known 42.1%, rest 22.3%, gap **+19.8pt** [+6.7,+33.2] — only cell whose CI clears zero
- Gemini judge / DeepInfra-embed: well_known 15.8%, rest 7.7%, gap +8.1pt [−1.4,+18.6]
- GLM judge / Gemini-embed: well_known 7.0%, rest 5.6%, gap +1.4pt [−5.0,+8.8]
- GLM judge / DeepInfra-embed: well_known 8.8%, rest 3.2%, gap +5.6pt [−1.2,+13.9]

**Deployment divergence, Qwen3-Embedding-8B: DeepInfra vs. the lab deployment** (`deepinfra_vs_labembed.json`/`.md`, `deepinfra_vs_labembed_paired.json`):
- Raw cosine similarity, 500 identical texts: mean **0.9947**, median 0.9950 (both already unit-norm)
- McNemar exact test, paired 500-query retrieval outcomes:
  - Easy Hit@1: DeepInfra-only 1, Lab-only 1, p=1.000
  - Easy Hit@5: DeepInfra-only 5, Lab-only 10, p=0.302
  - Easy Hit@10: DeepInfra-only 2, Lab-only 3, p=1.000
  - Hard Hit@5: DeepInfra-only 2, Lab-only 1, p=1.000
  - Hard Hit@10: both-hit 88, DeepInfra-only **17**, Lab-only **1**, p=**0.00014**

---

## 2. Trajectory domain (procedural memory / AgentInstruct-ALFWorld, n=118)

**Baseline retrieval, STRICT, all 3 embedders, definition (i) frozen, with CIs**
(`results/task1_expanded_full_results.json`, `task2b_bootstrap_cis.json` → `trajectory_pooled_n118`):
- Pooled chance: Hit@1 15.3%, Hit@5 55.4%, Hit@10 79.1% (hypergeometric, per-query gold-set size)
- labembed-Qwen3-8B: Hit@1 17.8% [11.0,25.4], Hit@5 45.8%, Hit@10 63.6%, MAP 0.453 — above chance
- gemini-embedding-001: Hit@1 15.3% [9.3,22.0], Hit@5 50.8%, Hit@10 71.2%, MAP 0.489 — at chance
- MiniLM-L6-v2: Hit@1 9.3% [4.2,14.4], Hit@5 32.2%, Hit@10 56.8%, MAP 0.289 — below chance
- old_40 subset: labembed 10.0%/25.0%/42.5%, gemini 5.0%/22.5%/45.0%, MiniLM 2.5%/15.0%/42.5% (Hit@1/5/10)
- new_78 subset: labembed 21.8%/56.4%/74.4%, gemini 20.5%/65.4%/84.6%, MiniLM 12.8%/41.0%/64.1%

**Tier-definition robustness, Task 2A** (`results/task2a_tier_robustness.json`):
- (i) frozen (diff. object), chance Hit@1 15.3%: labembed 17.8% (above), gemini 15.3% (at), MiniLM 9.3% (below)
- (ii) harsher (diff. object **and** diff. receptacle), chance Hit@1 14.1%: labembed **11.0%**, gemini **8.5%**, MiniLM **6.8%** — all three **below chance**
- (iii) lenient (any object, reference), chance Hit@1 16.9%: labembed 75.4%, gemini 74.6%, MiniLM 50.0%

**Dumb (lexical) reranker control — helps, opposite of math** (`task1_expanded_full_results.json`, CIs `task2b_bootstrap_cis.json`):
- labembed: full_mix Hit@1 29.7%, share closed **+25.9%** [11.3,41.2]
- gemini-embedding-001: full_mix Hit@1 35.6%, share closed **+36.4%** [23.4,50.0]
- MiniLM-L6-v2: full_mix Hit@1 24.6%, share closed **+32.1%** [18.5,47.1]

**Verb/noun mechanism ablation, old_40 subset** (`results/mechanism_ablation_results.json`):
- labembed: orig Hit@1 10.0%; full_mix share closed 15.4%; verb_only Hit@1 15.0%, share closed 15.4%; noun_only Hit@1 10.0%, share closed **0.0%**
- gemini-embedding-001: orig Hit@1 5.0%; full_mix share closed 31.3%; verb_only Hit@1 12.5%, share closed 18.8%; noun_only Hit@1 7.5%, share closed 6.3%
- MiniLM-L6-v2: orig Hit@1 2.5%; full_mix share closed 18.8%; verb_only Hit@1 7.5%, share closed 12.5%; noun_only Hit@1 7.5%, share closed 12.5%
- Verb overlap alone drives most of the lexical reranker's gain; noun/object overlap alone contributes little to nothing (0% for labembed) to a lot less than the full mix in every embedder.

**Verb-first vs. adjective-first surface-form slice** (`task1_expanded_full_results.json`, task_types 3/4/5):
- labembed: verb (n=40) 15.0%→27.5% (+12.5pt); adjective (n=22) 18.2%→36.4% (+18.2pt)
- gemini-embedding-001: verb 12.5%→22.5% (+10.0pt); adjective 13.6%→31.8% (+18.2pt)
- MiniLM-L6-v2: verb 2.5%→22.5% (+20.0pt); adjective 13.6%→45.5% (+18.2pt)

**LLM reranker, n=118, both judges, all 3 embedders, with bootstrap CIs**
(`results/task_traj_reranker_n118.json`; 708 calls, net new spend $0.30; truncation 1.1% capped /
1.3% unparsed, isolated to GLM, diagnosed as GLM's known terse-prompt profile, proceeded per the
<=10% threshold):
- labembed / Gemini: Hit@1 37.3% (orig 17.8%), share closed 42.6% [27.3,59.3]
- labembed / GLM: Hit@1 49.2% (orig 17.8%), share closed **68.5%** [50.9,87.5]
- gemini-embed / Gemini: Hit@1 40.7% (orig 15.3%), share closed 45.5% [31.9,59.7]
- gemini-embed / GLM: Hit@1 57.6% (orig 15.3%), share closed **75.8%** [60.7,91.4]
- MiniLM / Gemini: Hit@1 37.3% (orig 9.3%), share closed 58.9% [42.1,76.8]
- MiniLM / GLM: Hit@1 44.9% (orig 9.3%), share closed **75.0%** [56.9,93.2]
- Judge ranking reverses vs. math domain: GLM closes 1.3–1.6x more gap than Gemini here, every embedder, non-overlapping CIs
- old_40 vs new_78 split: labembed Gemini 61.5%/36.6%, labembed GLM 69.2%/68.3%, gemini-embed Gemini 62.5%/40.0%, gemini-embed GLM 75.0%/76.0%, MiniLM Gemini 56.3%/60.0%, MiniLM GLM 75.0%/75.0%
- Failure taxonomy, pooled misses (SIBLING/NEAR_MISS/OTHER/unparsed): labembed-Gemini 85.1%/13.5%/1.4%/0.0% (n=74 miss); labembed-GLM 88.3%/11.7%/0.0%/0.0% (n=60); gemini-Gemini 85.7%/14.3%/0.0%/0.0% (n=70); gemini-GLM 80.0%/14.0%/2.0%/4.0% (n=50); MiniLM-Gemini 74.3%/20.3%/5.4%/0.0% (n=74); MiniLM-GLM 72.3%/7.7%/9.2%/10.8% (n=65)
- SIBLING (literal same-object duplicate) dominates every cell (72–88%) — structural mirror of math's `own_nm_near_miss`

**Query-set expansion**: 40 original + 78 new (from 134 `hkust-nlp/agentboard` ALFWorld `valid_unseen`
episodes, 78 unique goal strings) = 118 total, vs. ~150 target — shortfall disclosed, not patched
(no `valid_seen`-equivalent found at a comparably lightweight source).

**Label-verification audit**: 60 new-query labels + 60 corpus/trajectory labels human-reviewed
(`results/task1_new_query_review_sample.json`, `results/agentinstruct_label_review_sample.json`,
60 items each) — **zero disagreements** with the rule-based classifier's assigned task_type/object/receptacle.

**MiniLM validation gate (corrected domain — see Discrepancies)** (`results/minilm_validation_gate.json`,
`scripts/validate_minilm_reproduction.py`; original-40 query set, reproducing the source trajectory
paper's own MiniLM pipeline and released judgments exactly, including their non-standard
normalize-by-found-not-total AP formula): overall MAP 0.630 vs. their published 0.7945; by tier
Easy 0.577 vs. 0.842, **Medium 0.753 vs. 0.746 (within 0.007)**, Hard 0.544 vs. 0.791. Medium
closely reproduces; Easy/Hard diverge because their pipeline judges retrieval fresh each run while
this reproduction scores against their released, frozen relevance-judgment pool — anything retrieved
outside the originally keyword-matched/judged candidates scores as not-relevant here even if it's a
legitimate match. MiniLM-L6-v2 was adopted as one of the three trajectory-domain embedders
throughout this document on this basis (see baseline retrieval above).

---

## 3. Utility curve (RAG: does retrieved context help a solver?), n=210, 3 conditions

**Setup**: DeepSeek-v4-flash solver (native API), Grader A = gemini-3.1-flash-lite (real cost),
Grader B = glm-5.2-fp8 (free), no tiebreaker — disagreement reported, not adjudicated. Target
n=250, landed at n=210/condition (630 jobs total) after filtering to queries with a usable reference
solution (`scripts/run_utility_curve_deepseek.py`, `utility_curve_deepseek_cache.jsonl`,
`results/utility_curve_deepseek.json`).

**Truncation rate, solver, 32,768-token cap — still substantial, not resolved:**
- none: 64/210 (30.5%) capped
- dumb: 66/210 (31.4%) capped
- gold: 65/210 (31.0%) capped
- Progression across this project's cap increases: 8,192 cap → ~50–80% truncated (original pilot / worst-case pretest) → 16,384 cap → 50% truncated (paired pretest) → **32,768 cap → ~31% truncated (this final run)**. Each increase helped; none eliminated the problem. Accuracy numbers below should be read with this caveat.

**Accuracy (Grader A / Grader B), by condition:**
- none: correct_a 146/210 (69.5%), correct_b 147/210 (70.0%)
- dumb: correct_a 143/210 (68.1%), correct_b 141/210 (67.1%)
- gold: correct_a 143/210 (68.1%), correct_b 145/210 (69.0%)

**Grader agreement (A and B gave the same correct/incorrect verdict):**
- none: 207/210 (98.6%)
- dumb: 202/210 (96.2%)
- gold: 206/210 (98.1%)

**Headroom recovery / regression, paired by query_id, Grader A:**
- none→gold: of 64 wrong-under-none, **10 recovered (10/64, 15.6%)**; of 146 right-under-none, **13 regressed (13/146, 8.9%)** — net −3 (matches 146−3=143)
- none→dumb: of 64 wrong-under-none, 8 recovered (8/64, 12.5%); of 146 right-under-none, 11 regressed (11/146, 7.5%) — net −3
- dumb→gold: of 67 wrong-under-dumb, 8 recovered (8/67); of 143 right-under-dumb, 8 regressed (8/143) — net 0

**McNemar exact test, paired, Grader A (both comparisons requested):**
- none vs. dumb: recovered 8, regressed 11, **p=0.6476** — not significant
- none vs. gold: recovered 10, regressed 13, **p=0.6776** — not significant
- (Grader B, for robustness: none vs. dumb p=0.3075; none vs. gold p=0.8318 — also not significant either grader)
- **Neither retrieval condition significantly changes solver accuracy vs. no context, under either grader.** Retrieval headroom exists (10–13 queries move in each direction) but nets out to noise at n=210.

**Original 6-condition pilot (superseded, correction banner in `results/rag_pilot.md`)** —
kept for reference only, not citable as a clean result:
- 528 cached solver answers, 6 conditions, 8,192-token cap: **49.6% truncated overall**
- By condition: `none` 38.6% vs. 50–53% for every context-bearing condition
- Truncated answers score 15–35 points worse than complete answers, every condition
- Gold condition: raw reported 63.6% accuracy, **82.9%** restricted to non-truncated answers only

**GLM solver run — stalled, abandoned, not reused** (`utility_curve_glm_cache.jsonl`, 307/630 jobs:
210 `none` + 97 `dumb` + 0 `gold`): stalled 4x under sustained shared-lab-endpoint load (6.5h for 49%
completion vs. ~1.6h predicted). Kept as a partial cache, explicitly not topped up or reused —
DeepSeek run used instead. No GLM-vs-DeepSeek solver comparison was ever computed on the overlap.

---

## 4. Integrity incidents (8), one line each

1. **GLM-CoT reranker truncation (math domain):** 63.3% (1,266/2,000) of responses truncated at the
   6,144-token cap, not concluded — parser still extracted an answer from all but 1, so "0 unparsed"
   looked clean until real token counts were checked. Excluded from all citable results.
2. **RAG-pilot solver truncation (original 6-condition pilot):** 49.6% of 528 cached answers
   truncated at 8,192 tokens; gold's reported 63.6% accuracy was 82.9% restricted to complete
   answers — a pipeline-wide bug, not a gold-specific one.
3. **Gemini hidden-thinking-token quirk (found twice — reranker judge, then RAG grader):** hidden
   thought tokens burned the output budget with no visible text; fixed both times via
   `thinkingConfig.thinkingBudget=0`.
4. **Billing scare / spend reconciliation:** user's actual Gemini spend hit $9.85 CAD against a $15
   cap, above every prior estimate; live run stopped immediately; spend reconstructed via real
   cache-file call counts + real sampled token sizes (not guesses) to $6.32–$6.99 USD (~$9.58–$9.79
   CAD), closing almost the entire gap.
5. **Solver-cap increases insufficient on their own:** 8,192→16,384 tokens still left 50% truncated
   on a worst-case-biased paired pretest (down from 80%); 16,384→32,768 in the final run still left
   ~31% truncated (§3 above) — truncation tracks problem difficulty, not just budget.
6. **Task 3 third-judge infeasibility:** Claude Haiku 4.5 ruled out (no Anthropic API credential
   anywhere in the project); DeepSeek-v4-flash as reranker judge showed unbounded reasoning — still
   `finish_reason=length`, empty `content`, at 24,000 tokens / 227 seconds on one query. Third judge
   dropped entirely; math-domain judge comparison stayed at 2 judges.
7. **GLM solver run stall (utility curve):** 6.5 hours for 49% completion vs. ~1.6h predicted, under
   sustained shared-lab-endpoint load — abandoned in favor of a DeepSeek-solver rerun rather than
   waited out.
8. **Contamination framing tightened by bootstrap CIs (Task 2C):** the original two-proportion-z
   framing read as "positive in most cells"; resampling showed only 1 of 4 judge/candidate-set cells
   actually clears zero (Gemini judge / Gemini-embed, [+6.7,+33.2]) — the other 3 are directionally
   consistent but not individually significant at n=57 well-known queries.

*(Not counted among the 8, but disclosed: a project-venv dependency drift this session —
`numpy`/`transformers` silently upgraded, breaking the MiniLM embedder — was root-cause-fixed by
pinning back to a mutually-compatible set rather than upgrading further; see 2026-08-14 entry in
`JOURNEY_LOG.md`.)*

---

## 5. Spend, final totals per provider (`results/SPEND.json`)

- Gemini: **$8.45**
- DeepSeek: **$3.49**
- Claude: **$0.00** (no Anthropic credential ever existed in this project; nothing was ever spent here)
- **Total: $11.94**, across 1,972 recorded API calls
