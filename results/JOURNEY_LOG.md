# Journey Log

Reconstructed from `results/` file timestamps and their content. One dated entry per working
day. Appended to at the end of every session going forward — this is the running project diary,
not a results doc; look in `results/*.md` for the actual findings.

---

## 2026-08-07 — Baseline retrieval, first controls

**Done**
- Set up the project as a standalone repo (`embedding-benchmark/`, separate from the Mantis
  product codebase), with `.gitignore` committed first specifically to protect `.env`/secrets.
- Ran the full gated Step 1 baseline: 500 queries, full 117,088-item corpus, both
  `gemini-embedding-001` and `Qwen3-Embedding-8B` (DeepInfra), easy tier
  (`baseline_gemini.json` 13:46, `baseline_gemini_hard.json` 15:04, `baseline_deepinfra.json`
  16:32, `baseline_deepinfra_hard.json` 16:35).
- Wrote up `baseline_results.md` (17:56) — STRICT/LENIENT Hit@k, tier-invariance proof, failure
  taxonomy, gate-check against the MathNet paper's Table 4.
- Saved 3 verbatim qualitative rank-1-miss examples (`strict_misses_examples.md`, 17:57).
- Ran the dumb lexical reranker control (`dumb_reranker_control.json`/`.md`, 18:07–18:30) — a
  confound check *before* touching any smart reranker.
- Ran the direct lexical-distance check (§2b), `lexical_distance_check.json`/`.md` (18:30) —
  95–99.8% of misses have the false positive more lexically similar than gold.
- Validated the LLM reranker pipeline at 50 queries (`llm_reranker_validation.json`, 18:38)
  before committing to the expensive full run.

**Decided**
- Memory-safe `VectorCache` (memmap-backed) design, after catching that the naive
  JSONL-in-RAM approach would need 5–11GB per provider at full corpus scale.
- Custom `hit_at_k()` binary metric, after catching that standard BEIR Recall@k mechanically
  under-scores lenient (multi-gold) matches.
- Narrowed the dumb-reranker-control conclusion to "rules out lexical fingerprint specifically"
  (not generation artifacts generally) per explicit correction.

**Blocked**
- Nothing carried over — each gate passed before proceeding to the next step.

**Next**
- Hard tier, both providers; investigate why our numbers ran higher than the paper's cited
  aggregate (resolved next session — turned out to be a wrong reference comparison).

---

## 2026-08-08 – 08-09 — Hard tier, the lab embedding deployment, first full reranker run

**Done**
- Fetched the MathNet paper directly (arXiv 2604.18584) and verified the gate-check against the
  paper's actual Easy-tier row rather than its prose-cited aggregate — resolved an apparent
  mismatch that turned out to be an inconsistency in the paper itself, not our numbers.
- Ran DeepInfra vs the lab deployment raw embedding cosine-similarity comparison on 500 identical
  texts (`deepinfra_vs_labembed.json`, Aug 8 19:36) — mean cosine 0.9947, both already
  unit-norm, so not a normalization difference.
- Ran the full terse-prompt LLM reranker: 500 queries × 4 configs, `gemini-3.1-flash-lite` judge
  (`llm_reranker_full.json` 19:51, write-up `llm_reranker_full.md` 23:32) — all positive, plus
  the pooled well_known-vs-rest contamination test with counts.
- Built `src/embed_labembed.py`, confirmed the lab endpoint serves the identical model ID,
  and ran its own full baseline (`baseline_labembed.json`/`_hard.json`, Aug 9 00:16).
- Extended the DeepInfra-vs-lab-deployment comparison to a **paired** McNemar test on actual
  retrieval outcomes, not just raw cosine similarity (`deepinfra_vs_labembed_paired.json`,
  `.md`, Aug 9 11:33) — found the ~0.5% embedding drift is retrieval-irrelevant almost
  everywhere, but produces a real, significant divergence specifically at hard-tier Hit@10
  (p=0.00014).
- Added GLM 5.2 (`glm-5.2-fp8`, lab endpoint) as a second reranker judge, terse prompt
  (`llm_reranker_full_glm.json`, `llm_reranker_glm_judge.md`, Aug 9 22:42–22:44) — direction
  replicated, magnitude didn't (2–4x smaller gains than Gemini).

**Decided**
- `.env` secrets (the lab deployment URL, later the LLM endpoint) go in `.env` only, never
  hardcoded — repo goes public eventually.
- Judge model made pluggable (`JudgeBackend` interface) specifically so a second/third judge
  could be swapped in without touching prompt or parsing logic.

**Blocked**
- Nothing project-blocking; caught and fixed my own bugs along the way (a stale skip-condition
  in an earlier smoke test, an unhandled connection-level exception in the retry logic).

**Next**
- Contamination framing: is judge-dependence a refutation or expected (different training
  data)? Revisit once a CoT-prompt version exists to compare against.

---

## 2026-08-10 — CoT prompt, full second-judge comparison

**Done**
- Diagnosed and fixed a Gemini-3 "thinking" quirk (hidden thought tokens burn budget with no
  visible output) via `thinkingConfig.thinkingBudget=0` — became a permanent fix in
  `GeminiJudgeBackend`.
- 50-query GLM terse-vs-CoT diagnostic (`glm_cot_diagnostic.md`, originally written 12:10) —
  found CoT prompting engages real reasoning GLM's terse instruction was suppressing.
- Ran the full CoT-prompt comparison: both judges, all 4 configs
  (`llm_reranker_full_cot_gemini.json` 12:36, `llm_reranker_full_cot_glm.json` 20:50,
  write-up `llm_reranker_cot_full_comparison.md`, originally written 20:52) — Gemini CoT
  clean throughout; GLM CoT's token/truncation behavior not yet checked at the time.
- Added the contamination-is-judge-dependent-by-training-data framing per explicit request —
  DeepInfra-embed candidate set showed corroborating signal under both judges even though the
  Gemini-embed candidate set's effect vanished under GLM.

**Decided**
- Judge-dependent contamination magnitude is expected (different training data → different
  competition memorization), not evidence the effect is fake — but should be reported per-judge,
  not pooled into one number.

**Blocked**
- Nothing yet identified — the truncation problem in the GLM CoT run existed in this data from
  the start but wasn't discovered until the next session (real token counts weren't checked at
  the time; only "0 unparsed" was verified, which turned out to be the wrong signal).

**Next**
- (Retroactive, from next session) real per-response token accounting on the GLM CoT run.

---

## 2026-08-10 (evening) – 08-11 (early hours) — RAG pilot v1 (single grader)

**Done**
- Read the MathNet paper's RAG section directly: 3 conditions (Zero-Shot/Embed-RAG/Expert-RAG),
  no deliberately-bad condition, incidental below-baseline observation for 3 solver models.
- Designed and built the 6-condition quality-curve pilot (none/dumb/baseline/glm_reranked/
  gemini_reranked/gold), GLM 5.2 solver, `gemini-3-flash-preview` grader.
- Hit and resolved two real data gaps along the way: MathNet-Retrieve corpus items have no
  solution field and can't be reliably joined to MathNet-Solve (paraphrased text) — resolved by
  going problem-only context, per explicit direction; query IDs, unlike corpus items, turned out
  to be near-verbatim original MathNet-Solve text, so a direct text join for *reference
  solutions* worked (99/100 matched, 88/100 had a usable worked solution) —
  `data/solve_solution_index.json` (21:22 Aug 10).
- Fixed a second Gemini "thinking" issue specific to the grader (this time it ignored terse
  instructions even with a budget) via the same `disable_thinking` fix.
- Ran the full 88-query × 6-condition pilot (`rag_pilot_cache.jsonl`, `rag_pilot.json`/`.md`,
  01:11–01:13 Aug 11) — survived a mid-run directory move that broke the venv's baked-in
  absolute paths (fixed `activate` script, resumed from cache) and set up a proper crash/
  progress Monitor after under-verifying an earlier status update.

**Decided**
- Report the pilot's result honestly as **inconclusive** — no pairwise comparison reached
  p<0.05 at n=88 despite a 12.5-point spread in the raw numbers — rather than reading a
  "dumb > gold" headline into noise.

**Blocked**
- Six-way split at n=88 doesn't have the power to resolve the effect sizes in play. (Addressed
  same day — see below.)

**Next**
- Narrow to fewer conditions at higher n; the real question is `none` vs `dumb`.

---

## 2026-08-11 — Grading-cost design, correction notices, truncation follow-up, pilot narrowing, billing audit, solver-truncation bug

**Done**
- Designed a two-grader-plus-Claude-adjudication grading scheme for the eventual full RAG
  curve; validated DeepSeek-V3.2 as Grader B (no hidden-reasoning issue, confirmed on both a
  trivial and a real grading prompt) and produced a real-measured cost estimate (not a guess)
  for all three providers before spending anything.
- Discovered, via real per-response tokenization (GLM's own `/tokenize` endpoint), that
  **63.3% of the full GLM CoT run's 2,000 responses were truncated at the 6,144-token cap** —
  parsed-successfully was being mistaken for concluded. Pulled a concrete example (cut off
  mid-sentence, parser grabbed a number from unresolved deliberation).
- Added correction banners to `llm_reranker_cot_full_comparison.md` and
  `glm_cot_diagnostic.md` stating plainly that GLM-CoT numbers should not be cited, and that
  Gemini CoT and all terse-prompt numbers are unaffected.
- Built a concise-reasoning CoT prompt variant (`COT_CONCISE_PROMPT_TEMPLATE`) and a 16,384-cap,
  30-query truncation-rate test — result 10.0% (3/30), a real improvement but not enough to
  justify an 8.7-hour rerun for a secondary question.
- Started this file.
- Hit a billing scare (user's Gemini spend at $9.85 CAD against a $15 cap, well above every
  estimate given so far) — stopped the actively-spending narrowed pilot (`TaskStop`) immediately,
  before any further investigation, then reconstructed the full Gemini spend with zero new
  billed calls: exact call counts from cache-file line counts (3,100 flash-lite reranker calls +
  625 flash-preview grader calls + 118,108 embedding items) plus real token measurements sampled
  directly from the cached prompts/responses. First pass: $6.12–$6.47 (didn't fully close the
  gap). Second pass, after measuring the RAG-pilot grader's real input size (it re-feeds the
  solver's full completion — mean 5,496 tokens/call, much bigger than first assumed): revised to
  $6.32–$6.99 (mid $6.65), which converts to ≈$9.58–$9.79 CAD — closes almost the entire gap.
  Confirmed one real cache-duplication instance (520-item smoke-test embedding overlap, ~$0.01);
  no other re-run-without-cache-hit found.
- Built `src/spend_tracker.py` + `results/SPEND.json` — real code-enforced spend tracking
  (`record_call()` from actual API usage, `SessionSpendGuard` hard-stops a run at $3 of new
  spend) — per explicit instruction that a doc to remember to open isn't a control.
- Free gold-condition diagnosis (per explicit priority: "if the oracle condition is broken,
  every condition below it is uninterpretable"): pulled 5 grader-marked-wrong gold cases,
  confirmed the retrieved context was a legitimate correct-equivalent problem in the first case
  (ruling out "gold retrieval is broken"), then found via GLM's free `/tokenize` endpoint that
  all 5 solver answers hit exactly the 8,192-token `SOLVER_MAX_TOKENS` cap. Extended the check to
  **all 528 cached solver answers across all 6 original-pilot conditions: 49.6% truncated
  overall**, every condition affected (38.6% for `none` vs 50–53% for every context-bearing
  condition), and truncated answers score 15–35 points worse than complete ones in every
  condition. This is a pipeline-wide solver-truncation bug, not a gold-specific problem — the
  oracle's reported 63.6% is dragged down by truncation; restricted to non-truncated answers only,
  gold hits 82.9%.
- Built `src/truncation_check.py` as a standing, reusable check (GLM `/tokenize`, free) — to be
  run on any generation output before reporting results from it, per explicit instruction (this
  is the third silent truncation bug this project has hit: GLM CoT judge, GLM CoT diagnostic,
  now the RAG solver — all three parsed cleanly and looked fine until token counts were checked).

**Decided**
- Don't rerun GLM CoT at all — use the clean GLM terse results as the cross-judge data point.
  Final framing: "contamination found robustly under Gemini across both prompts; GLM terse shows
  the same direction at borderline significance; GLM CoT excluded due to 63.3% truncation."
  Applied to both docs.
- Narrow the RAG pilot to `none` vs `dumb` vs `gold` at n=250 before any 500-query scale-up.
- Going forward: nothing over $1 runs without an estimate + confirmation; hard stop every $3 of
  new spend; running spend total reported at the top of every message.
- Raise `SOLVER_MAX_TOKENS` (8192→16,384) and discard/rerun the narrowed pilot's first 97 jobs,
  gated on a free 20-call pretest first (<~10% truncation to proceed).
- Truncation-rate reporting is now a standing requirement after any generation run, not an
  ad-hoc check triggered by suspicion.

**Blocked (superseded below — see the continuation)**
- Narrowed RAG pilot (n=250, 3 conditions) stopped mid-run at 97/630 jobs when the billing issue
  came up; cache preserved but must be discarded, not resumed, once the cap is raised (those 97
  are truncated at the old 8192 cap).
- Claude tiebreaker grading path still blocked on model/design decisions; Haiku 4.5 call-count
  cap not yet built (no tiebreaker pipeline exists yet to cap).
- Grader-A swap test (flash-lite vs flash-preview vs DeepSeek agreement) explicitly on hold
  until the solver-cap rerun lands — one thing at a time, per explicit instruction.

**Next (superseded below — see the continuation)**
- 20-call free pretest at the 16,384 cap; if truncation is still >~10%, stop and reconsider
  whether GLM is usable as a solver at all rather than doubling the cap again.
- If the pretest passes: rerun the narrowed pilot at the new cap (~$2.25, approved), reporting
  truncation rate per condition alongside the accuracy results.
- Then: Grader-A swap test, Claude tiebreaker design/cap.

---

## 2026-08-11 (continued, overnight) — billing reconciliation delivered, 16k pretest failed, retrieval writeup drafted

**Done**
- Recomputed the Gemini spend reconciliation with real per-model precision (exact cache-file line
  counts + real prompt/response sizes sampled directly from cached data, no guesses): flash-lite
  (terse+CoT reranker+validation, 3,100 calls) $2.85–$3.26; flash-preview (RAG pilot grader only,
  625 calls) $1.87–$1.89; gemini-embedding-001 (118,108 items) $1.61–$1.84. Total $6.32–$6.99
  (mid $6.65) ≈ $9.58–$9.79 CAD — closes almost the entire gap against the user's reported $9.85
  CAD (the earlier $6.30 seed had undercounted the grader's real input size — it re-feeds the
  solver's full completion, averaging 5,496 tokens/call, not a small prompt). Updated
  `results/SPEND.json`'s reconstruction with this tighter number and methodology note.
  Confirmed Aug 10-11 (CoT reranker + pilot grading) = ~68% of total spend, matching the user's
  own billing-dashboard observation.
- Ran the approved (~$2.25) 20-call free pretest at 16,384 tokens before the paid rerun, per
  explicit gate: **failed** — 50% truncated on a worst-case-biased sample (paired same-query
  comparison: 80% truncated at the old 8,192 cap → 50% at 16,384 — a real but insufficient
  improvement, several answers hit the new cap exactly too). Did **not** run the paid rerun.
  Flagged the likely diagnosis: GLM appears to fill whatever budget it's given on hard problems
  rather than converging near a natural stopping point — same failure class as the CoT judge
  before an explicit concise instruction fixed it (63.3%→10.0%), not a pure budget problem.
- Built and ran a concise-solving-prompt variant (`scripts/test_solver_concise_16k.py`) on the
  same 20 queries at 16,384 tokens, to test whether an explicit brevity instruction succeeds
  where a bigger cap alone didn't — result pending as this entry is written.
- Drafted `results/RETRIEVAL_WRITEUP.md` — a standalone, professor-facing writeup of everything on
  the retrieval/reranking side (validation gate, tier-invariance proof, failure taxonomy, dumb
  reranker + lexical distance controls, LLM reranker gains, CoT effects, contamination analysis,
  deployment divergence, explicit limitations and "what this does not show" sections). Deliberately
  excludes the RAG pilot/utility curve, which is still blocked on the solver-truncation bug.
- Added a correction banner to `results/rag_pilot.md` itself (previously only the two GLM-CoT docs
  had one) — its six-condition table is downstream of the same truncation bug (49.6% overall,
  per-condition rates and the truncated-vs-complete accuracy gap included in the banner) and
  should not be cited as a settled finding.
- Computed (no API calls — pure arithmetic on the existing pricing table) a cost estimate for
  tomorrow's planned utility-curve rerun: solver=DeepSeek-V3.2, Grader A=Gemini (flash-lite if
  quality holds), Grader B=GLM 5.2 (free), no Claude tiebreaker, n=250×3 conditions (~630
  gradeable jobs). Estimate driven mostly by DeepSeek's unmeasured real output length.

**Decided**
- Overnight work restricted to free-only actions (no paid runs) per explicit instruction — the
  $3 hard stop doesn't help an unattended run, so the constraint is enforced by not starting
  anything paid at all tonight.
- New role assignment for the utility curve, chosen so no model grades its own solutions: solver
  DeepSeek-V3.2, Grader A Gemini, Grader B GLM (free — grading is short output, so GLM's
  verbosity problem doesn't apply there), no Claude tiebreaker — report grader disagreement rate
  directly instead of adjudicating it.

**Blocked**
- Solver-truncation fix still unresolved — 16,384 cap alone insufficient (50% truncated); concise-
  prompt result pending. Until resolved, no RAG-pilot rerun (narrowed or original) should proceed.
- Grader-A swap / utility-curve rerun: cost estimate computed but the run itself needs the user's
  go-ahead (explicitly deferred to tomorrow) and depends on the solver-truncation question above
  being resolved for the same underlying pipeline.

**Next**
- Report the concise-prompt truncation rate once the background test completes.
- If concise-prompt truncation is low enough: consider whether the narrowed pilot rerun should use
  the concise-prompt solver rather than just a bigger cap.
- Get go-ahead on the utility-curve cost estimate, then run n=250×3 with the new solver/grader
  role assignment.

---

## 2026-08-12 – 08-13 — Second domain (trajectories), robustness tasks, third-judge attempt

**Done**
- Task 1: expanded the trajectory-domain query set from 40 to 118 (steps 1–5: `hkust-nlp/agentboard`
  ALFWorld `valid_unseen` split, 78 unique new goal strings against a ~150 target — reported as a
  real data-availability shortfall, not silently patched; steps 6–7: full pipeline rerun on all 118,
  three embedders (`labembed-Qwen3-8B`, `gemini-embedding-001`, `MiniLM-L6-v2`), pooled/old/new
  breakdown, plus a verb-first-vs-adjective-first surface-form slice within task types 3/4/5).
  `results/task1_expanded_tier_labels.json`, `task1_expanded_full_results.json`.
- Task 2A: tier-definition robustness for the trajectory domain's near-chance pooled result — tested
  a harsher gold definition (different object **and** different receptacle). All three embedders,
  including both production models, score *below* random chance under this definition —
  strengthens rather than weakens the original finding. `results/task2a_tier_robustness.json`.
- Task 2B: 10,000-resample bootstrap 95% CIs on headline numbers in both domains (math baseline +
  terse reranker share-closed; trajectory pooled Hit@k + dumb-reranker share-closed).
  `results/task2b_bootstrap_cis.json`.
- Task 2C: bootstrap CIs on the math-domain contamination gap (well_known vs rest, hard tier, both
  judges, both candidate sets). Found only 1 of 4 judge/candidate-set cells has a CI that clears
  zero — tightens the §3.6 contamination framing from "found under both judges" to "confidently
  established in one cell, directionally consistent but not individually significant in the other
  three." `results/task2c_contamination_cis.json`.
- Task 3: attempted a third math-domain reranker judge. Claude Haiku 4.5 was ruled out immediately —
  no Anthropic API credential exists anywhere in this project (confirmed by checking `.env` for
  every plausible key name); the client and pilot script built for it were discarded once this was
  confirmed rather than run against nonexistent credentials. DeepSeek (`deepseek-v4-flash` native
  API) was then probed directly: despite behaving as a fast, terse grader elsewhere in this project,
  as a *reranker judge* it engaged genuinely open-ended reasoning — still `finish_reason=length`
  with empty `content` at 24,000 tokens / 227 seconds on a single representative query, no
  convergence in sight. Reported to the user as an infeasibility finding (not a token-cap tuning
  problem); user confirmed dropping the third judge rather than running a scaled-down sample.
- Wrote `results/RESULTS_SUMMARY.md` — the consolidated top-level document tying together the math
  domain (condensed pointer into `RETRIEVAL_WRITEUP.md` + the new CI additions), the full new
  trajectory-domain writeup, the Task 3 negative finding, and a cross-domain synthesis section
  (§5: both domains show these embedding models track literal surface content over abstract
  structure when the two are separable, but whether a cheap lexical control helps or hurts depends
  on whether the dataset's surface variation is adversarial (math, hurts) or incidental
  (trajectories, helps) — not a domain-general property of lexical reranking).

**Decided**
- Report contamination with the tightened Task 2C framing everywhere going forward, not the
  looser "found under both judges" framing from before bootstrap CIs existed.
- Task 3 third-judge robustness check is abandoned, not deferred — the existing two-judge
  (Gemini + GLM) comparison is final for this project; no third data point will be sought later
  under the current model/credential constraints.

**Blocked**
- Nothing carried forward — this was the final planned task before re-freezing results.

**Next**
- None planned; `results/RESULTS_SUMMARY.md` is the final deliverable for this project's current
  scope.

---

## 2026-08-14 — Final gap-fill: n=118 trajectory reranker, then the numbers digest

**Done**
- Discovered and fixed real environment drift blocking this session: the project `.venv` had numpy
  silently upgraded to 2.5.1 (incompatible with the pinned `torch==2.2.2`) and
  `transformers`/`sentence-transformers` drifted to versions requiring `torch>=2.4` — broke the
  MiniLM embedder entirely (`_lzma` missing on system Python; torch/numpy ABI mismatch on the venv).
  Restored a mutually-compatible pinned set (`numpy<2`, `scipy<1.13`, `transformers==4.40.0`,
  `sentence-transformers==2.7.0`) rather than upgrading `torch` — root-cause fix, not a version bump
  forward, since nothing about the task called for newer libraries.
- Ran the last open cell: the LLM reranker on the trajectory domain's full n=118 query set, both
  judges (`gemini-3.1-flash-lite`, `glm-5.2-fp8`), terse prompt, all three embedders' top-10 sets,
  708 total judge calls. Reused 236 of the original n=40 pilot's cached calls after verifying
  byte-identical query IDs/tiers/rankings; the MiniLM top-10s shifted on 4 of those (library-version
  floating-point ties, not a data change) and were discarded and re-queried rather than trusted
  stale. Net new spend $0.30 (pre-estimated $1–2). `scripts/task_traj_reranker_n118.py`,
  `results/task_traj_reranker_n118.json`.
- Truncation/finish_reason audit before reporting (standing rule): 1.1% capped / 1.3% unparsed,
  entirely isolated to GLM, hand-diagnosed as GLM's known terse-prompt profile (occasional
  unsolicited reasoning trace, occasional out-of-range digit) rather than a new failure mode — well
  under the project's 10% stop threshold, so reporting proceeded.
- Found a genuine cross-domain surprise: **GLM closes 1.3–1.6x more of the recoverable gap than
  Gemini in the trajectory domain, on every embedder** — the opposite ranking from the math domain,
  where Gemini's gains are consistently 2–4x larger than GLM's. Failure-mode taxonomy (reusing the
  tier file's pre-computed SIBLING/NEAR_MISS/OTHER partition) shows the two domains agree on *why*
  rerankers fail even though they disagree on *which judge is stronger*: 72–88% of every judge's
  misses in the trajectory domain pick the literal same-object duplicate over an unrelated
  distractor — the structural mirror of the math domain's dominant surface-favoring failure mode.
- Updated `results/RESULTS_SUMMARY.md` (§3.6, new) and its cross-domain synthesis (§5) with these
  findings; this is now the final version of that document.

**Decided**
- Report the judge-magnitude reversal explicitly rather than letting the math domain's "GLM is the
  weaker judge" framing stand unqualified — it does not generalize to the second domain, which is
  itself a finding worth stating plainly rather than picking one domain's framing as canonical.

**Next**
- Write `results/FINAL_NUMBERS.md`, a flat one-line-per-number digest of every citable number
  across both domains plus the utility-curve/RAG pilot work and the project's integrity incidents,
  for the paper draft. Final task of this project.

*(A `results/FINAL_NUMBERS.md`-writing session and a separate "repo publication pass" session —
secrets audit, mantiscluster→labembed anonymization, LICENSE/README/requirements.txt, clean-clone
test — both happened between this entry and the next; no journal entry was written for either at
the time. Flagged here rather than silently left unindexed; backfill if useful later.)*

---

## 2026-08-17 — Four-task gap-fill: math CI fix, utility-null recheck, Haiku third judge, fetch scripts

**Done**
- Task 0 (free verification, reported before anything else ran): reconciled the paper's "1.8%"
  n=118 truncation figure against the digest's "1.1%/1.3%" — both correct, the paper's is the union
  (13/708, 4 records both capped and unparsed) while the digest reported the two individual rates.
  Confirmed both math-domain embedders have bootstrap CIs on easy-tier Hit@1 in the underlying data
  (`task2b_bootstrap_cis.json`) — the paper's Table 1 is missing the Qwen-emb bracket, not missing
  data; same gap existed in `FINAL_NUMBERS.md`'s own first draft, fixed there too.
- Task 1 (free): recomputed the utility-curve null on the subset complete (non-truncated) in all
  three conditions (n=127/210). Accuracy jumps to near-ceiling (97–100%) — in the `none` condition,
  *exactly* the 146 non-truncated queries were the 146 scored correct, no exceptions. The null still
  holds (no significant McNemar result either grader), but the honest framing shifted: the 69.5%
  headline accuracy is close to a truncation proxy, not a solving-ability measure, and there's
  essentially no headroom left once truncation is controlled for — a materially different story than
  "truncation doesn't matter."
- Task 2: Claude Haiku 4.5 as a third reranker judge, both domains. Confirmed `ANTHROPIC_API_KEY`
  had been added to the parent `.env` (not yet synced to the repo-root copy from the portability fix
  two sessions ago — re-synced). 10-call pilot first (`scripts/task2_haiku_pilot.py`): clean,
  `stop_reason=end_turn` throughout, no capping, $0.0014/call measured. Pre-estimated the full scope
  (2,354 calls) from one real probe per domain at ~$5.46 — well under the $15 stop threshold — then
  ran it (`scripts/task2_haiku_reranker_full.py`; actual cost $5.30). Truncation audit: 0.10% math,
  0.85% trajectories, both far under threshold. Finding: **Haiku is the outlier in the math domain**
  (tier-inverted relative to both other judges — dramatically stronger on easy, dramatically weaker
  on hard, the opposite pattern from Gemini and GLM) **and closely corroborates Gemini in the
  trajectory domain, where GLM is the outlier instead.** No judge is "the odd one out" in general —
  each domain has a different outlier, and a different judge each time. This makes the paper's
  current §8/§9 framing ("one [judge] was attempted and abandoned") stale — flagged in
  `FINAL_NUMBERS.md`'s Discrepancies section, paper itself not touched per instruction.
- Task 3: built and verified three fetch scripts, all free, no credentials needed (public sources).
  Found `ShadenA/MathNet-Retrieve` as a **separate** Hugging Face dataset repo from the raw MathNet
  corpus (`ShadenA/MathNet`) — resolves the gap flagged two sessions ago, where the raw corpus alone
  had no equivalence/near-miss pairing fields and reconstructing them via the original LLM-paraphrase
  pipeline wouldn't have reproduced the exact text. All 9 files verified byte-exact (SHA256) against
  this project's existing `data/`. Along the way, discovered the trajectory corpus's `/tmp` scratch
  copy had actually disappeared since the last session — a routine cleanup cleared it, including the
  cloned repo's own `.git/config` — confirming exactly the fragility the fetch script exists to fix.
  Restored it via the user-supplied real source (`github.com/qpiai/Proced_mem_bench`), verified
  count (336 trajectories) and spot-checked content (`alfworld_0`'s task description byte-identical
  to what was used throughout this project) before trusting it for the Haiku trajectory run. Built
  the ALFWorld valid_unseen fetch (`hkust-nlp/agentboard`) with a full text cross-check, not just a
  hash — 78/78 stored query texts confirmed present in the fetched goal set. Final end-to-end proof:
  re-ran the n=118 trajectory reranker against freshly-fetched data and got a byte-identical
  `results/task_traj_reranker_n118.json` with zero new API calls.
- Updated `results/FINAL_NUMBERS.md` (Task 0 reconciliation, Task 1 subset, Task 2 both domains plus
  a new stale-paper-section flag) and `README.md` (fetch scripts documented in Data + Reproduction,
  credential table, "known gap" language retired now that it's closed).

**Decided**
- Report the utility-null's complete-answers-only recheck as a reframing, not a resolution — "the
  null holds because there's no headroom left" is a different claim than "the null holds despite
  truncation," and conflating them would overstate what the subset check actually shows.
- Do not touch `paper_draft_v3.md` — flag every place its current text is now stale (MiniLM gate
  domain from the prior session, third-judge status from this one, missing Table 1 bracket) in
  `FINAL_NUMBERS.md`'s Discrepancies section instead, per explicit instruction that integration is a
  separate, later step.

**Next**
- None from this task list. Paper integration (folding the digest's corrections and the new Haiku
  results into `paper_draft_v3.md`) is the user's next step, explicitly deferred.
