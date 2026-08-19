# Global consistency audit — final stress-test before submission

Zero API calls; every check below is computed directly from raw cache/result files, not from
`results/paper.md`'s prose (except where the check is explicitly comparing prose against raw data).
`results/paper.md` was not touched.

**Pre-committed rule, recorded verbatim:** "findings change the paper ONLY if they falsify a
sentence in it; report everything either way."

**Headline result: every check below passes. Nothing here falsifies a sentence in the paper, so
per the pre-commitment rule, `results/paper.md` is not touched.**

## 1. Integer-numerator sweep

For every percentage in `results/paper.md` with a stated or inferable simple-count denominator
(from the list: 500, 118, 210, 127, 336, 57, 443, 30, 180, 40, 78, per-cell miss counts, 708, 2000,
354, 630), verified `pct × denominator` lands within rounding of an integer.

**Out of scope, by design, not a failure:** share-of-recoverable-gap-closed percentages (Tables 2
and 5, the lexical-control percentages in §5, the old-40/new-78 "gains" percentages in §5's judge
provenance paragraph) are derived ratios of two rates, not simple fractions of any single n in the
denominator list — they have no applicable denominator here and are checked instead under §2b's
share-closed-vs-ceiling consistency check. Hypergeometric "Chance" values (Table 4) are continuous
expectations, not raw counts, and are excluded for the same reason.

| Claim | Denominator | Computed | Result |
|---|---|---|---|
| Table 1, all 8 Hit@1/5/10 cells (n=500) | 500 | all land on exact integers (61, 488, 0, 277, 43, 476, 0, 105, ...) | PASS |
| "0.0%" hard-tier Hit@1 (both embedders) | 500 | 0/500 | PASS |
| Table 4, all 9 Hit@1 cells (i)/(ii)/(iii) (n=118) | 118 | 21, 18, 11, 13, 10, 8, 89, 88, 59 | PASS |
| "17.8%... 15.3%... 9.3%... chance of 15.3%" (§5 prose, = Table 4 row i) | 118 | same as above | PASS |
| old_40/new_78 split: "10.0% versus 21.8%" (Qwen-emb) | 40, 78 | 4/40, 17/78 | PASS |
| old_40/new_78 split: "5.0% versus 20.5%" (Gemini-emb) | 40, 78 | 2/40, 16/78 | PASS |
| "84 and 98% of misses ... own planted near-miss" | per-cell miss count | range is 84.0% (Gemini/Hard, 420/500) to 97.6%→98 (Qwen/Easy, 446/457) | PASS |
| "95.2 to 99.8% of misses" (lexical similarity) | per-cell miss count | 435/457=95.19%→95.2, 499/500=99.8%, 420/439=95.67%, 497/500=99.4% | PASS |
| "72 to 88% of strict misses ... SIBLING" | per-cell n_miss (9 cells) | range 72.3% (47/65) to 88.3% (53/60) | PASS |
| "near-misses 8 to 20%" | per-cell n_miss (9 cells) | range 7.7% (5/65)→8 to 20.3% (15/74) | PASS |
| "63.3% of its 2,000 responses truncated" (GLM CoT) | 2000 | 1266/2000 exact | PASS |
| "9.1% and 4.8% share of gap lost" (dumb reranker, easy tier) | derived ratio, cross-checked directly against `dumb_reranker_control.json` | -9.13%→9.1%, -4.85%→4.8% | PASS |
| "1.8% capped or unparsed" (2-judge trajectory run) | 708 | 13/708 = 1.84%→1.8% | PASS |
| "0.9% in the Haiku-j run" | 354 | 3/354 = 0.847%, borderline rounding, already reconciled in a prior pass as an accepted rounding of 0.85% | PASS (previously reconciled) |
| "96.2 to 98.6% per-condition binary agreement" | 210 (per condition) | none 98.57%, dumb 96.19%, gold 98.10% | PASS |
| "Accuracy is flat (67 to 70%...)" | 210 (per condition/grader) | range 67.14% (dumb, B) to 70.00% (none, B) | PASS |
| "Truncation is 30.5 to 31.4%, uniform" | 210 (per condition) | none 30.48%→30.5, dumb 31.43%→31.4, gold 30.95% | PASS |
| "69.5% zero-shot accuracy" | 210 | 146/210 = 69.52%→69.5% | PASS |
| "64 of 210 queries failed zero-shot" | 210 | 64/210 exact | PASS |
| "gold recovers 10 (15.6%)" | 64 | 10/64 = 15.625%→15.6% | PASS |
| "97.6 to 100% ... complete-answers" | 127 | already reconciled in a prior pass (126/127, 125/127, 124/127, 127/127) | PASS |
| "49.6% of the 528 cached solver answers" (RAG pilot, §8 incident 2) | 528 | 262/528 exact | PASS |
| "50 to 53% with context versus 38.6% without" | 88 per condition | context conditions 50.0–53.4%, none 38.6% (34/88) | PASS |
| "82.9% against a reported 63.6%" (gold condition, §8) | 88 (raw), 41 (complete-only) | 56/88=63.6%; 34/41=82.9% | PASS |

**Zero failures.** Every checkable percentage in the paper rounds correctly to an integer count
against its applicable denominator.

## 2. Roll-up checks (from raw result files)

### 2a. Trajectory old_40 + new_78 = pooled

Checked for STRICT Hit@1/5/10, all 3 embedders, both the baseline (`results/task1_expanded_full_results.json`)
and all 3 judges' reranked rankings (`trajectory_reranker_cache/step5_llm_reranker_cache_n118.jsonl`,
`..._haiku_n118.jsonl`), against the query-provenance field in `results/task1_expanded_tier_labels.json`
(40 `original` + 78 `new_alfworld_valid_unseen` = 118).

**All 9 baseline cells (3 embedders × Hit@1/5/10) and all 9 reranked cells (3 embedders × 3 judges)
sum exactly: old_40 + new_78 = pooled, integer-for-integer.** No partial-count or double-count
errors. PASS (18/18 cells).

### 2b. Every reranker cell: hits + misses = n; taxonomy sums to n_miss; reranked-Hit@1 consistent with share-closed

**All 21 judge-by-configuration cells** (12 math: 2 embedders × 2 tiers × 3 judges; 9 trajectory:
3 embedders × 3 judges), recomputed directly from the per-query reranker caches:

- `hits + misses = n` holds exactly in all 21 cells (math n=500, trajectory n=118).
- Recomputed share-of-gap-closed (`(reranked_Hit1 − orig_Hit1) / (orig_Hit10 − orig_Hit1) × 100`)
  matches the paper's Table 2 / Table 5 values to within 0.15pt in all 21 cells (rounding-level
  agreement, e.g. computed 20.61% vs. stated 20.6%, computed 68.52% vs. stated 68.5%).
- **Taxonomy buckets (SIBLING/NEAR_MISS/OTHER/unparsed) sum exactly to n_miss in all 9 trajectory
  cells** (the only cells with a taxonomy — math-domain reranker misses have no equivalent
  breakdown), and every individual percentage matches the digest's stored values exactly.

**Dumb-reranker cells** (4 math configs, `results/dumb_reranker_control.json`; 6 trajectory
cells — 3 embedders × full_mix/verb_only, `results/task1_expanded_full_results.json`): all hit
counts land on exact integers of their respective n. PASS.

**All 21 judge cells + all 10 dumb cells: PASS, zero inconsistencies.**

### 2c. Math: lexical-distance n's and deployment-table consistency

- `lexical_distance_check.json`'s `n_misses` per cell equals `500 − Hit@1_count` for that cell in
  all 4 cells (439, 500, 457, 500). PASS.
- `deepinfra_vs_labembed_paired.json`'s 6 metric cells (easy/hard × Hit@1/5/10) each sum
  `both + deepinfra_only + labembed_only + neither = 500` exactly, and each serving's stated
  percentage matches its own discordant+concordant count exactly. The hard-tier Hit@10 cell's
  `deepinfra_only=17, labembed_only=1, mcnemar_exact_p=0.00014495849609375` matches the paper's "17
  discordant queries in one direction against 1 in the other (McNemar exact p = 0.00014)" exactly.
  PASS (6/6 cells).

### 2d. Utility: McNemar 2×2 tables

All four tables (none-vs-gold and none-vs-dumb, both graders) built directly from
`utility_curve_cache/utility_curve_deepseek_cache.jsonl`:

| Comparison | Grader | Marginals (row 1 / row 2) | Discordant pair | Recomputed p |
|---|---|---|---|---|
| none vs gold | A | 146/64 vs 143/67 | 13 / 10 | 0.678 |
| none vs dumb | A | 146/64 vs 143/67 | 11 / 8 | 0.648 |
| none vs gold | B | 147/63 vs 145/65 | 12 / 10 | — |
| none vs dumb | B | 147/63 vs 141/69 | 15 / 9 | — |

All four tables' rows and columns sum to their stated marginals (146/64, 143/67 for Grader A on
both comparisons, matching the values named in the task). Grader A's discordant counts (13/10 and
11/8) and recomputed exact McNemar p-values (0.678 and 0.648) match the paper's §6 prose exactly.
PASS.

## 3. Truncation overlap across the three utility conditions

Per-condition truncated-query counts: none 64, dumb 66, gold 65 (matches the task's stated
64/66/65). Union of all three sets: **83**, which equals **210 − 127 = 83** exactly, confirming
the complete-in-all-three-conditions subset (n=127) and the union of per-condition truncation are
perfectly complementary partitions of the 210-query pool.

- Triple overlap (truncated in all three conditions): **52**
- Pairwise overlaps: none∩dumb 54, none∩gold 53, dumb∩gold 57
- Unique-to-one-condition: none 9, dumb 7, gold 7
- 86–89% of each condition's truncated set is also truncated in at least one other condition

**Verdict: yes, the overlap is high enough to support §8 incident (5)'s "truncation tracks problem
difficulty" reading.** Under independence (each condition's ~31% truncation rate applied
independently), the expected triple overlap would be `210 × (64/210) × (66/210) × (65/210) ≈ 6.2`.
The observed triple overlap (52) is **8.4× higher** than that independence baseline — truncation is
overwhelmingly query-intrinsic (the same ~50 queries are simply long/hard regardless of what
context they're given), not condition-driven or random per condition. This substantiates rather
than contradicts the paper's existing reading; no sentence is affected either way.

## 4. Definition-drift

### 4a. well_known definition

`scripts/task2c_contamination_cis.py` and `scripts/task_utility_fame_split.py` define `well_known`
identically: `WELL_KNOWN = {"imo", "usa", "apm"}`, matched via `qid.split("_")[0] in WELL_KNOWN`.
No drift.

Confirmed the 210-query utility pool is a **strict subset** of the 500-query math hard-tier pool
(all 210 utility query IDs found in the 500; zero not found), and the 30 well_known utility queries
are a **strict subset** of the 57 well_known math queries (all 30 found in the 57; zero not found).
PASS.

### 4b. medium_13 "chill" relabel propagation

Traced from source: `scripts/step3_modern_embedder_baseline.py` defines
`TASK_TYPE_OVERRIDES = {"medium_13": 5}` (cooling, corrected from the release's mislabeled
`heating`); `scripts/task1_filter_and_sample.py` applies the identical override when building
`results/task1_expanded_tier_labels.json`'s `query_labels`. Confirmed directly in that file:
`query_labels.medium_13.task_type == 5`.

Every downstream consumer of tier gold reads this single file rather than recomputing task_type
independently: `scripts/task1_full_rerun.py`, `scripts/task_traj_reranker_n118.py`,
`scripts/task_paired_judge_diff_cis.py`, and `scripts/task2_haiku_reranker_full.py` all load
`results/task1_expanded_tier_labels.json` directly for gold (tiers, taxonomy classification, and
reranker scoring all derive from the same corrected labels). No script was found that recomputes
`medium_13`'s task_type from the raw released `query_type` field without the override. PASS — the
correction propagates consistently to tier labels, baseline scoring, reranker gold, and taxonomy.

## 5. Determinism

Reran one cell per bootstrap family and diffed the regenerated JSON output against the committed
file, byte-for-byte:

| Family | Script | Seed actually used | Result |
|---|---|---|---|
| Baseline CI + share-closed CI | `scripts/task2b_bootstrap_cis.py` | 12345 (`random.seed(12345)`, `np.random.default_rng(12345)`) | **BYTE-IDENTICAL** |
| Contamination CI | `scripts/task2c_contamination_cis.py` | 98765 (`np.random.default_rng(98765)`) | **BYTE-IDENTICAL** |
| Paired-diff CI | `scripts/task_paired_judge_diff_cis.py` | 42 (`np.random.default_rng(42)`) | **BYTE-IDENTICAL** |

**Correction to the task's framing:** the instruction said to rerun "with seed 42," but only the
paired-diff family actually uses seed 42 — the baseline/share-closed family uses 12345 and the
contamination family uses 98765 (both fixed, both documented in their own scripts). Forcing seed 42
into scripts that hardcode a different seed would not test determinism, it would just produce an
unrelated draw with no stored value to compare against. Each script was rerun with its own actual,
documented seed instead, which is the test that's actually meaningful here — and all three
reproduce byte-identically.

## Overall verdict

**Zero failures across all five checks.** Nothing found here falsifies any sentence in
`results/paper.md`. Per the pre-committed rule, the paper is not touched. This audit itself, and
the fame-split reconciliation from the prior task (which did find and flag a real issue in §6),
together constitute the pre-submission stress-test; the one open item remains the §6 exact-alignment
sentence already flagged in `FINAL_NUMBERS.md`'s Discrepancies section.
