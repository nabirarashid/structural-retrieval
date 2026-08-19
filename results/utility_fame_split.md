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

## 7. Verdict

**No paper sentence affected.** Checked against `results/paper.md` §6 (the only section making
claims about the complete-answers-only ceiling and its cause) and §9 (the contamination-limitation
sentence, which is scoped to the retrieval/reranking domain, §4, not the downstream solver
experiment). The one finding that could in principle have mattered — whether the 97.6–100%
complete-127 ceiling is an artifact of well_known-query overrepresentation — is answered no (item 6:
both groups sit at 99.1–100%, a 0.0–0.9pt gap), which is consistent with, not contradictory to,
§6's existing claim that the ceiling reflects near-zero solver headroom rather than a group-specific
effect. The larger full-sample gap in raw `none`-condition accuracy (item 4: 80.0% vs 67.8%,
Grader A, 12.2pt) does not exceed the 15pt threshold set for this check and is not the subject of
any specific sentence in the paper to begin with (§6 does not decompose zero-shot accuracy by
query fame). Per the decision rule, this stays meeting-prep material only; no paper wording changes.
