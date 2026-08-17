# Paired bootstrap CIs for judge differences

Paper-review follow-up. The paper's "non-overlapping CIs" claim checks marginal CIs (Table 2/5) —
that doesn't settle the paired question, since two judges reranking the *same* top-10 lists for the
*same* queries are correlated, not independent. This computes the paired difference directly:
resample query indices (unit = query ID, with replacement), and for each resample recompute both
judges' share-of-gap-closed using that resample's own Hit@1/Hit@10 as the ceiling (judge-independent
within a resample, since both judges reranked the identical embedding ranking). 10,000 resamples,
seed=42. Script: `scripts/task_paired_judge_diff_cis.py`. Raw output: `results/paired_judge_diff_cis.json`.

**All 10 cells: query-ID sets matched exactly (no mismatches), zero degenerate resamples (gap=0)
anywhere, and every paired-difference CI excludes zero.** The marginal-CI-overlap noted in the
motivation (e.g. trajectory Qwen-emb GLM [50.9,87.5] vs. Gemini [27.3,59.3]) does not indicate a real
problem — the paired test, which is the methodologically correct one for this comparison, confirms
every judge-pair difference is real.

## Math domain (n=500), Gemini-j minus GLM-j

| Candidates | Tier | Diff (pt) | 95% CI | Zero excluded |
|---|---|---|---|---|
| Gemini-emb | Easy | +10.5 | [+5.2, +15.7] | Yes |
| Gemini-emb | Hard | +33.9 | [+27.8, +40.2] | Yes |
| Qwen-emb | Easy | +23.6 | [+18.3, +28.8] | Yes |
| Qwen-emb | Hard | +22.9 | [+12.2, +33.7] | Yes |

## Trajectory domain (n=118), GLM-j minus Gemini-j

| Candidates | Diff (pt) | 95% CI | Zero excluded |
|---|---|---|---|
| Qwen-emb (labembed) | +25.9 | [+13.3, +39.2] | Yes |
| Gemini-emb | +30.3 | [+15.3, +44.8] | Yes |
| MiniLM | +16.1 | [+4.7, +28.3] | Yes |

## Trajectory domain (n=118), GLM-j minus Haiku-j

Computed because the paper claims GLM diverges from *both* other judges in this domain, not just Gemini-j.

| Candidates | Diff (pt) | 95% CI | Zero excluded |
|---|---|---|---|
| Qwen-emb (labembed) | +22.2 | [+10.0, +35.1] | Yes |
| Gemini-emb | +31.8 | [+18.3, +45.5] | Yes |
| MiniLM | +26.8 | [+15.7, +38.8] | Yes |

## Reading notes

- **Degenerate resamples** (ceiling = Hit@10 − Hit@1 = 0 for that resample, making share-of-gap-closed
  undefined) never occurred in any of the 10 × 10,000 resamples — the math-domain hard-tier Hit@1 is
  0% exactly, but Hit@10 is well above zero (55.4%/21.0%), so the ceiling never collapses even though
  the numerator can.
- Math-domain point estimates (Gemini-j minus GLM-j) are all positive, consistent with Gemini-j
  outperforming GLM-j on terse math reranking throughout this project. Trajectory-domain point
  estimates (GLM-j minus the other two) are likewise all positive, consistent with GLM-j being the
  trajectory-domain outlier.
- This does not by itself validate every qualitative claim built on top of the marginal CIs (e.g.
  "which judge is the outlier per domain" reasoning) — it validates specifically that the CI-overlap
  observation in the motivation was not evidence against those claims.
