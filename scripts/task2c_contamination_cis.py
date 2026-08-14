"""Task 2C: bootstrap 95% CIs for the well_known vs rest contamination gaps
(terse prompt, both judges, both candidate sets) -- pooled well_known(n=57)
vs rest(n=443), hard tier only, matching the existing pooled comparison in
llm_reranker_full.md / llm_reranker_glm_judge.md. Reuses the exact-match
gold scoring fixed in Task 2B (endswith over-counted decoy siblings).
"""
import json

import numpy as np

WELL_KNOWN = {"imo", "usa", "apm"}


def bucket_is_well_known(qid: str) -> bool:
    return qid.split("_")[0] in WELL_KNOWN


def bootstrap_gap_ci(well_known_hits: list, rest_hits: list, n_boot: int = 10000) -> dict:
    wk = np.array(well_known_hits, dtype=bool)
    rest = np.array(rest_hits, dtype=bool)
    rng = np.random.default_rng(98765)
    wk_idx = rng.integers(0, len(wk), size=(n_boot, len(wk)))
    rest_idx = rng.integers(0, len(rest), size=(n_boot, len(rest)))
    wk_rates = wk[wk_idx].mean(axis=1)
    rest_rates = rest[rest_idx].mean(axis=1)
    gaps = (wk_rates - rest_rates) * 100
    lo, hi = np.percentile(gaps, [2.5, 97.5])
    return {"well_known_rate": float(wk.mean()), "rest_rate": float(rest.mean()),
            "gap_pts": float(wk.mean() * 100 - rest.mean() * 100), "ci_lo": float(lo), "ci_hi": float(hi)}


def main():
    print("=== TASK 2C: contamination gap CIs, terse prompt, hard tier ===\n")
    results = {}
    for judge_label, cache_prefix in [("gemini-3.1-flash-lite", "full"), ("glm-5.2-fp8", "full_glm")]:
        for embed_provider in ["gemini", "deepinfra"]:
            fname = f"llm_reranker_cache/{cache_prefix}_{embed_provider}_hard.jsonl"
            import os
            if not os.path.exists(fname):
                continue
            recs = [json.loads(l) for l in open(fname)]
            hits = {}
            for r in recs:
                qid = r["query_id"]
                hits[qid] = r["chosen_id"] == f"{qid}::eq::hard"
            wk_hits = [hits[qid] for qid in hits if bucket_is_well_known(qid)]
            rest_hits = [hits[qid] for qid in hits if not bucket_is_well_known(qid)]
            ci = bootstrap_gap_ci(wk_hits, rest_hits)
            key = f"{judge_label}/{embed_provider}-embed"
            results[key] = {**ci, "n_well_known": len(wk_hits), "n_rest": len(rest_hits)}
            print(f"{key}: well_known={ci['well_known_rate']*100:.1f}%(n={len(wk_hits)})  "
                  f"rest={ci['rest_rate']*100:.1f}%(n={len(rest_hits)})  "
                  f"gap={ci['gap_pts']:+.1f}pts [{ci['ci_lo']:+.1f},{ci['ci_hi']:+.1f}]")

    json.dump(results, open("results/task2c_contamination_cis.json", "w"), indent=2)
    print("\nSaved: results/task2c_contamination_cis.json")


if __name__ == "__main__":
    main()
