"""Paper-review follow-up: paired bootstrap CIs for judge DIFFERENCES,
replacing the "non-overlapping CIs" claim (which only checked marginal
overlap -- not the same thing as a paired difference test). Free: reads
only existing per-query cache files, makes zero API calls.

Method per cell: resample query indices (with replacement, unit = query
ID), and for each resample recompute BOTH judges' share-of-gap-closed
using THAT RESAMPLE's own Hit@1/Hit@10 as the ceiling (not a fixed
baseline) -- since both judges reranked the identical embedding top-10
for the identical queries, the ceiling is judge-independent within a
resample, so this is a legitimate paired comparison. Record the
difference (direction stated per cell group). A resample whose ceiling
(Hit@10 - Hit@1) is exactly zero makes share-of-gap-closed undefined;
these are counted and reported explicitly, not silently dropped, and
excluded only from the percentile CI itself (which needs finite values).

10,000 resamples, seed=42 (per instruction -- a fresh seed for this
specific analysis, distinct from other bootstrap passes in this project).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

N_BOOT = 10000
SEED = 42


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


def per_query_hits(records, gold_fn):
    """records: list of {"query_id", "top10"/"top10_ids", "chosen_id"}.
    Returns dict query_id -> (orig_top1_hit, orig_top10_hit, judge_hit)."""
    out = {}
    for r in records:
        qid = r["query_id"]
        top10 = r.get("top10_ids") or r.get("top10")
        gold = gold_fn(qid)  # always a set, single-element for math
        orig_top1 = top10[0] in gold
        orig_top10 = any(t in gold for t in top10)
        judge_hit = (r["chosen_id"] in gold) if r["chosen_id"] else False
        out[qid] = {"top10": top10, "orig_top1": orig_top1, "orig_top10": orig_top10, "judge_hit": judge_hit}
    return out


def paired_diff_ci(per_q_A, per_q_B, label_A, label_B, n_boot=N_BOOT, seed=SEED):
    """per_q_A/B: dict query_id -> per-query dict from per_query_hits(), SAME
    underlying embedding ranking (orig_top1/orig_top10 must match between A/B
    per query -- verified by caller). Returns the paired-diff CI result dict."""
    qids_A, qids_B = set(per_q_A), set(per_q_B)
    if qids_A != qids_B:
        missing_in_B = qids_A - qids_B
        missing_in_A = qids_B - qids_A
        return {
            "error": "query_id_mismatch",
            "n_A": len(qids_A), "n_B": len(qids_B),
            "missing_in_B_sample": list(missing_in_B)[:5],
            "missing_in_A_sample": list(missing_in_A)[:5],
        }

    qids = sorted(qids_A)
    n = len(qids)
    orig_top1 = np.array([per_q_A[q]["orig_top1"] for q in qids], dtype=bool)
    orig_top10 = np.array([per_q_A[q]["orig_top10"] for q in qids], dtype=bool)
    # sanity: A and B must agree on the underlying (judge-independent) ranking facts
    orig_top1_B = np.array([per_q_B[q]["orig_top1"] for q in qids], dtype=bool)
    orig_top10_B = np.array([per_q_B[q]["orig_top10"] for q in qids], dtype=bool)
    ranking_mismatch = int((orig_top1 != orig_top1_B).sum() + (orig_top10 != orig_top10_B).sum())

    hit_A = np.array([per_q_A[q]["judge_hit"] for q in qids], dtype=bool)
    hit_B = np.array([per_q_B[q]["judge_hit"] for q in qids], dtype=bool)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))

    boot_top1 = orig_top1[idx].mean(axis=1)
    boot_top10 = orig_top10[idx].mean(axis=1)
    boot_gap = boot_top10 - boot_top1
    boot_hitA = hit_A[idx].mean(axis=1)
    boot_hitB = hit_B[idx].mean(axis=1)

    degenerate_mask = boot_gap == 0.0
    n_degenerate = int(degenerate_mask.sum())

    with np.errstate(divide="ignore", invalid="ignore"):
        share_A = (boot_hitA - boot_top1) / boot_gap * 100
        share_B = (boot_hitB - boot_top1) / boot_gap * 100
        diff = share_A - share_B

    finite_mask = np.isfinite(diff) & ~degenerate_mask
    diff_finite = diff[finite_mask]
    n_finite = int(finite_mask.sum())

    point_orig_h1 = float(orig_top1.mean())
    point_orig_h10 = float(orig_top10.mean())
    point_gap = point_orig_h10 - point_orig_h1
    point_hitA = float(hit_A.mean())
    point_hitB = float(hit_B.mean())
    point_share_A = (point_hitA - point_orig_h1) / point_gap * 100 if point_gap != 0 else None
    point_share_B = (point_hitB - point_orig_h1) / point_gap * 100 if point_gap != 0 else None
    point_diff = (point_share_A - point_share_B) if (point_share_A is not None and point_share_B is not None) else None

    if n_finite == 0:
        lo = hi = None
    else:
        lo, hi = (float(x) for x in np.percentile(diff_finite, [2.5, 97.5]))

    return {
        "label": f"{label_A} minus {label_B}",
        "n_queries": n,
        "n_boot": n_boot,
        "seed": seed,
        "point_share_A": point_share_A,
        "point_share_B": point_share_B,
        "point_diff": point_diff,
        "ci_lo": lo,
        "ci_hi": hi,
        "zero_excluded": (lo is not None and hi is not None and (lo > 0 or hi < 0)),
        "n_degenerate_resamples": n_degenerate,
        "n_finite_resamples_used_for_ci": n_finite,
        "ranking_fact_mismatches_A_vs_B": ranking_mismatch,
    }


def main():
    results = {}

    # ---------------- MATH: Gemini-j minus GLM-j, 4 configs ----------------
    print("=== MATH: Gemini-j minus GLM-j, paired bootstrap diff CIs ===\n")
    for provider, tier in [("gemini", "easy"), ("gemini", "hard"), ("deepinfra", "easy"), ("deepinfra", "hard")]:
        gemini_recs = load_jsonl(f"llm_reranker_cache/full_{provider}_{tier}.jsonl")
        glm_recs = load_jsonl(f"llm_reranker_cache/full_glm_{provider}_{tier}.jsonl")
        gold_fn = lambda qid, tier=tier: {f"{qid}::eq::{tier}"}
        pq_gemini = per_query_hits(gemini_recs, gold_fn)
        pq_glm = per_query_hits(glm_recs, gold_fn)
        res = paired_diff_ci(pq_gemini, pq_glm, "Gemini-j", "GLM-j")
        key = f"math/{provider}-embed/{tier}"
        results[key] = res
        if "error" in res:
            print(f"{key}: ERROR {res['error']} -- n_A={res['n_A']} n_B={res['n_B']}")
            continue
        print(f"{key}: diff={res['point_diff']:.1f}pt [{res['ci_lo']:.1f},{res['ci_hi']:.1f}] "
              f"zero_excluded={res['zero_excluded']} degenerate={res['n_degenerate_resamples']}")

    # ---------------- TRAJECTORIES: GLM-j minus Gemini-j, 3 embedders ----------------
    print("\n=== TRAJECTORIES: GLM-j minus Gemini-j, paired bootstrap diff CIs ===\n")
    traj_recs = load_jsonl("trajectory_reranker_cache/step5_llm_reranker_cache_n118.jsonl")
    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    tiers = expanded["tiers"]
    traj_gold_fn = lambda qid: set(tiers[qid]["STRICT"])

    for ename in ["labembed-Qwen3-8B", "gemini-embedding-001", "MiniLM-L6-v2"]:
        gemini_recs = [r for r in traj_recs if r["embedder"] == ename and r["judge"] == "gemini-3.1-flash-lite"]
        glm_recs = [r for r in traj_recs if r["embedder"] == ename and r["judge"] == "glm-5.2-fp8"]
        pq_gemini = per_query_hits(gemini_recs, traj_gold_fn)
        pq_glm = per_query_hits(glm_recs, traj_gold_fn)
        res = paired_diff_ci(pq_glm, pq_gemini, "GLM-j", "Gemini-j")
        key = f"trajectories/{ename}/GLM_minus_Gemini"
        results[key] = res
        if "error" in res:
            print(f"{key}: ERROR {res['error']} -- n_A={res['n_A']} n_B={res['n_B']}")
            continue
        print(f"{key}: diff={res['point_diff']:.1f}pt [{res['ci_lo']:.1f},{res['ci_hi']:.1f}] "
              f"zero_excluded={res['zero_excluded']} degenerate={res['n_degenerate_resamples']}")

    # ---------------- TRAJECTORIES: GLM-j minus Haiku-j, 3 embedders ----------------
    print("\n=== TRAJECTORIES: GLM-j minus Haiku-j, paired bootstrap diff CIs ===\n")
    haiku_recs_all = load_jsonl("trajectory_reranker_cache/step5_llm_reranker_cache_haiku_n118.jsonl")

    for ename in ["labembed-Qwen3-8B", "gemini-embedding-001", "MiniLM-L6-v2"]:
        glm_recs = [r for r in traj_recs if r["embedder"] == ename and r["judge"] == "glm-5.2-fp8"]
        haiku_recs = [r for r in haiku_recs_all if r["embedder"] == ename]
        pq_glm = per_query_hits(glm_recs, traj_gold_fn)
        pq_haiku = per_query_hits(haiku_recs, traj_gold_fn)
        res = paired_diff_ci(pq_glm, pq_haiku, "GLM-j", "Haiku-j")
        key = f"trajectories/{ename}/GLM_minus_Haiku"
        results[key] = res
        if "error" in res:
            print(f"{key}: ERROR {res['error']} -- n_A={res['n_A']} n_B={res['n_B']}")
            continue
        print(f"{key}: diff={res['point_diff']:.1f}pt [{res['ci_lo']:.1f},{res['ci_hi']:.1f}] "
              f"zero_excluded={res['zero_excluded']} degenerate={res['n_degenerate_resamples']}")

    json.dump(results, open("results/paired_judge_diff_cis.json", "w"), indent=2)
    print("\nSaved: results/paired_judge_diff_cis.json")


if __name__ == "__main__":
    main()
