# Utility experiment: famous-vs-obscure composition

**Internal robustness check, pre-committed as paper-external.** Zero API calls; computed entirely
from `utility_curve_cache/utility_curve_deepseek_cache.jsonl` and cross-checked against
`results/utility_curve_deepseek.json`. Script: `scripts/task_utility_fame_split.py`.

**Decision rule, recorded verbatim as given before any data was computed:** "this split enters the
paper ONLY if it falsifies a sentence currently in paper_final.md. Otherwise it is meeting-prep
material only. No paper wording changes based on direction." (Note: `results/paper_final.md` was
renamed to `results/paper.md` in the prior commit; the check below was run against the current file
at that new path, which is the same content the rule refers to.)

`well_known` uses the same definition as Task 2C (`scripts/task2c_contamination_cis.py`): query_id
prefix (text before the first `_`) in `{imo, usa, apm}`.

## 1. All 210 utility queries

well_known = **30**, rest = **180**

## 2. Complete-in-all-three-conditions subset (n=127)

well_known = **20**, rest = **107**

## 3. Zero-shot failures (wrong under `none`, Grader A) — n=64

well_known = **6**, rest = **58**

(This n=64 matches paper.md §6's stated "64 of 210 queries failed zero-shot" exactly — an
independent consistency check on the underlying cache, not just the fame split.)

## 4. Accuracy in `none` condition, by group, both graders

| Grader | well_known | rest |
|---|---|---|
| Grader A | 24/30 (80.0%) | 122/180 (67.8%) |
| Grader B | 23/30 (76.7%) | 124/180 (68.9%) |

## 5. Truncation (capped) rate in `none`, by group

well_known: 8/30 (26.7%) — rest: 56/180 (31.1%)

## 6. Complete-127 `none`-condition accuracy per group (is the 97.6–100% ceiling driven by well_known queries?)

| Grader | well_known | rest | Diff |
|---|---|---|---|
| Grader A | 20/20 (100.0%) | 106/107 (99.1%) | 0.9pt |
| Grader B | 20/20 (100.0%) | 107/107 (100.0%) | 0.0pt |

Both diffs are under the 15-point threshold set for this check, so no significance test was
computed (per instruction). **The complete-answers ceiling is not driven by well_known queries** —
it holds essentially uniformly across both groups (99.1–100% vs 100%), so the fame composition of
the 210-query pool is not doing the work of producing the near-ceiling ceiling on the complete
subset.

## Reconciliation (2026-08-18)

**Trigger.** In item 4, well_known `none`-condition correct (Grader A) is 24/30. In item 5,
well_known `none`-condition capped is 8/30, implying complete (non-capped) is 22/30. 24 correct
against 22 complete is arithmetically impossible if paper.md §6's claim holds — "the alignment is
exact under Grader A: the 146 answers that finished and the 146 answers scored correct are the same
set; every completed answer was right and every truncated one wrong" — because if two sets are
literally identical, every subgroup restriction of them must have equal size too.

**(a) Set comparison, full 210 `none`-condition records, Grader A.** Directly computed, zero API
calls:

- `{correct under none, Grader A}`: 146 queries
- `{finish_reason == "stop" under none}`: 146 queries
- **These are not the same set.** Symmetric difference = 4 queries:
  - Correct despite truncation (in the correct set, not the complete set): `apm_2015_af5968`,
    `imo_2015_bab8ff`
  - Wrong despite finishing (in the complete set, not the correct set): `irn_2025_912017`,
    `mem_2023_1807d4`
- The two counts still coincide at 146 each only because the swap is symmetric (2 in, 2 out on both
  sides) — a coincidence of totals, not evidence of set equality. 206 of 210 queries do align exactly
  (complete-and-correct or truncated-and-wrong, as claimed); 4 do not.
- Cross-check: `capped` (boolean field) and `finish_reason == "length"` are exactly co-extensive
  across all 210 `none` records (symmetric difference = empty set) — the mismatch is not a
  capped-vs-finish_reason field disagreement, it's a genuine correct-vs-complete disagreement.

**(b) Recount, well_known subgroup, same definitions.**

- well_known correct (Grader A): **24/30** — unchanged, independently reconfirmed
- well_known complete (`finish_reason == "stop"`): **22/30** — unchanged, independently reconfirmed
- Both of the 2 "correct despite truncated" queries (`apm_2015_af5968`, `imo_2015_bab8ff`) are
  well_known, which is the entire source of the 24-vs-22 gap. Neither "complete but wrong" query is
  well_known, so the rest group carries the mirror-image gap instead: rest correct 122/180, rest
  complete 124/180.

**Which number was wrong, and why.** None of the six numbers above (24, 22-implied, 8, 122,
124-implied, 56) was a computation error — all are independently reconfirmed from the raw cache and
are internally consistent with their own definitions. **What was wrong is the assumption, inherited
from paper.md §6's phrasing, that "correct" and "complete" are the same set** — an assumption this
task's original item 4/5 numbers never asserted directly but that the apparent 24-vs-22 conflict
exposed. Once that assumption is dropped, 24 correct and 22 complete are simply two different,
non-nested quantities for the same subgroup, and there is no error to resolve in the fame-split
numbers themselves.

**Paper impact — the alignment claim itself fails.** Quoting `results/paper.md` §6 verbatim: *"In
the no-context condition the alignment is exact under Grader A: the 146 answers that finished and
the 146 answers scored correct are the same set; every completed answer was right and every
truncated one wrong."* This is falsified by the 4 discordant query IDs in (a). **This is a paper-edit
candidate** — flagged here per the pre-commitment rule (this finding does falsify a sentence in the
paper text, so unlike the rest of this document it crosses the bar to matter for the paper). Per
this task's scope, `results/paper.md` was **not** edited; `results/FINAL_NUMBERS.md` was corrected
directly (it is not a paper file) and now carries both the corrected claim and a matching
Discrepancies-section flag for the integration pass.

## 7. Verdict (revised 2026-08-18)

**One paper sentence is affected — a correction candidate for the integration pass, not editable
here.** The complete-answers-ceiling question (item 6) that motivated this whole check came back
negative on its own terms (the ceiling is not fame-driven, 99.1–100% vs 100%), which would have kept
this entirely as meeting-prep material under the decision rule. But reconciling the arithmetic behind
items 4 and 5 (prompted by the 24-vs-22 well_known mismatch) surfaced a separate, genuine
falsification of §6's "same set" / "every completed answer was right and every truncated one wrong"
claim: see the Reconciliation section above for the 4 discordant query IDs. Per the decision rule as
literally stated, this now crosses the bar ("enters the paper... if it falsifies a sentence") — it is
flagged in `results/FINAL_NUMBERS.md`'s Discrepancies section for the integration pass. No paper file
was edited as part of this task.
