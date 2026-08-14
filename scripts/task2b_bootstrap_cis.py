"""Task 2B: bootstrap 95% CIs (10,000 resamples over queries) for headline
numbers in both domains. Scope note, stated plainly rather than silently
narrowed: "every headline number" is a very large set (dozens of table
cells across two domains); this covers the numbers actually cited as
headline claims in the existing writeups -- baseline Hit@k (math, 4
configs), terse-prompt reranker share-of-gap-closed (math, both judges, 4
configs each), and the trajectory domain's Hit@k/MAP + dumb-reranker
share-of-gap-closed on the frozen pooled n=118 set. CoT and dumb-reranker-
math numbers are not included (CoT-GLM is already excluded as unreliable;
math dumb-reranker was a negative control, not a headline number).
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, "src")

N_BOOT = 10000
random.seed(12345)  # fixed for reproducibility of the resampling itself


def bootstrap_ci(values: list, n_boot: int = N_BOOT) -> dict:
    """values: list of per-query scalars (bools or floats). Returns mean + 95% CI."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    boot_means = np.empty(n_boot)
    rng = np.random.default_rng(12345)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx_matrix].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi), "n": n}


def bootstrap_ratio_ci(before: list, after: list, n_boot: int = N_BOOT) -> dict:
    """Share-of-gap-closed style ratio: bootstrap the FULL statistic (share_closed)
    on each resample, not just resample the raw before/after arrays independently --
    preserves the pairing per query."""
    before = np.array(before, dtype=bool)
    after = np.array(after, dtype=bool)
    n = len(before)
    rng = np.random.default_rng(54321)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    b_resampled = before[idx_matrix]
    a_resampled = after[idx_matrix]
    orig_h1 = b_resampled.mean(axis=1)
    reranked_h1 = a_resampled.mean(axis=1)
    # gap uses the ORIGINAL (non-resampled) Hit@10 as a fixed denominator reference
    # isn't meaningful per-resample without hit@10 data; caller must supply gap separately
    return orig_h1, reranked_h1


def main():
    print("=== TASK 2B: BOOTSTRAP 95% CIs ===\n")
    all_results = {}

    # ---------------- MATH DOMAIN: baseline Hit@k, 4 configs ----------------
    print("--- Math baseline Hit@k (recomputed per-query from cached embeddings) ---")
    from data import MathNetEasy
    from embed_gemini import embed_all as gem_embed_all

    def to_matrix(ids, cache):
        m = cache.get_matrix(ids)
        return m / np.linalg.norm(m, axis=1, keepdims=True)

    def build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200):
        sims = query_matrix @ corpus_matrix.T
        results = {}
        for i, qid in enumerate(query_ids):
            row = sims[i]
            top_idx = np.argpartition(-row, top_n)[:top_n]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            results[qid] = {corpus_ids[j]: float(row[j]) for j in top_idx}
        return results

    from embed_deepinfra import embed_all as di_embed_all

    from pathlib import Path
    from vector_cache import VectorCache

    baseline_ci = {}
    for provider_name, tier_dim in [("Gemini-embedding-001", ("gemini", 3072)),
                                     ("Qwen3-Embedding-8B(DeepInfra)", ("deepinfra", 4096))]:
        provider_dir, dim = tier_dim
        for tier in ["easy", "hard"]:
            ds = MathNetEasy.load(tier=tier)
            query_ids = ds.sample_queries(500, seed=42)
            corpus_ids = list(ds.corpus.keys())
            # read already-cached vectors directly -- DEEPINFRA_API_KEY was removed from .env
            # per the earlier "no more DeepInfra" decision; these are pre-existing cached
            # embeddings from the original baseline run, not a new DeepInfra call
            cache_dir = Path(f"embeddings_cache/{provider_dir}")
            q_cache = VectorCache(cache_dir, "full_queries", dim=dim, capacity=len(query_ids))
            c_cache = VectorCache(cache_dir, "full_corpus", dim=dim, capacity=len(corpus_ids))
            print(f"[{provider_dir}/full_queries] reading {len(query_ids)} cached (no API call)")
            print(f"[{provider_dir}/full_corpus] reading {len(corpus_ids)} cached (no API call)")
            query_matrix = to_matrix(query_ids, q_cache)
            corpus_matrix = to_matrix(corpus_ids, c_cache)
            results = build_results(query_ids, query_matrix, corpus_ids, corpus_matrix, top_n=200)

            key = f"{provider_name}/{tier}"
            per_k = {}
            for k in [1, 5, 10]:
                hits = []
                for qid in query_ids:
                    gold = f"{qid}::eq::{tier}"
                    ranked = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:k]
                    hits.append(any(cid == gold for cid, _ in ranked))
                ci = bootstrap_ci(hits)
                per_k[f"Hit@{k}"] = ci
            baseline_ci[key] = per_k
            print(f"{key}: " + "  ".join(
                f"Hit@{k}={per_k[f'Hit@{k}']['mean']:.3f} [{per_k[f'Hit@{k}']['ci_lo']:.3f},{per_k[f'Hit@{k}']['ci_hi']:.3f}]"
                for k in [1, 5, 10]))
    all_results["math_baseline"] = baseline_ci

    # ---------------- MATH DOMAIN: terse reranker share-of-gap-closed, both judges ----------------
    print("\n--- Math LLM reranker (terse), share-of-gap-closed CIs ---")
    reranker_ci = {}
    for judge_label, cache_prefix in [("gemini-3.1-flash-lite", "full"), ("glm-5.2-fp8", "full_glm")]:
        for embed_provider, tier in [("gemini", "easy"), ("gemini", "hard"), ("deepinfra", "easy"), ("deepinfra", "hard")]:
            fname = f"llm_reranker_cache/{cache_prefix}_{embed_provider}_{tier}.jsonl"
            import os
            if not os.path.exists(fname):
                continue
            recs = [json.loads(l) for l in open(fname)]
            # exact match against THIS query's own gold id, not endswith -- top-10 candidates
            # can include OTHER queries' ::eq::<tier> siblings as decoys, so endswith over-counts
            before = [r["top10_ids"][0] == f"{r['query_id']}::eq::{tier}" for r in recs]
            after = [r["chosen_id"] == f"{r['query_id']}::eq::{tier}" for r in recs]
            orig_h1_boot, reranked_h1_boot = bootstrap_ratio_ci(before, after)
            # need orig Hit@10 per query too, for the gap -- reuse baseline hit@10 flags computed above
            key_base = f"{'Gemini-embedding-001' if embed_provider=='gemini' else 'Qwen3-Embedding-8B(DeepInfra)'}/{tier}"
            orig_h10_mean = baseline_ci[key_base]["Hit@10"]["mean"]
            orig_h1_mean = sum(before) / len(before)
            reranked_h1_mean = sum(after) / len(after)
            gap = orig_h10_mean - orig_h1_mean
            share_closed_mean = (reranked_h1_mean - orig_h1_mean) / gap * 100 if gap != 0 else None
            gap_boot = orig_h10_mean - orig_h1_boot  # treat h10 as fixed (not bootstrapped) for the ratio denominator
            share_closed_boot = (reranked_h1_boot - orig_h1_boot) / gap_boot * 100
            share_closed_boot = share_closed_boot[np.isfinite(share_closed_boot)]
            lo, hi = np.percentile(share_closed_boot, [2.5, 97.5]) if len(share_closed_boot) else (None, None)

            key = f"{judge_label}/{embed_provider}-embed/{tier}"
            reranker_ci[key] = {"share_closed_pct": share_closed_mean, "ci_lo": float(lo) if lo is not None else None,
                                 "ci_hi": float(hi) if hi is not None else None, "n": len(recs)}
            print(f"{key}: share_closed={share_closed_mean:.1f}% [{lo:.1f},{hi:.1f}]" if lo is not None else
                  f"{key}: share_closed=n/a")
    all_results["math_reranker_terse"] = reranker_ci

    # ---------------- TRAJECTORY DOMAIN: Hit@k/MAP + dumb-reranker, pooled n=118 ----------------
    print("\n--- Trajectory domain (pooled n=118): Hit@k/MAP + dumb-reranker CIs ---")
    from pathlib import Path
    from vector_cache import VectorCache
    from dumb_reranker import dumb_score

    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    query_labels = expanded["query_labels"]
    tiers = expanded["tiers"]
    query_ids = list(query_labels.keys())

    traj_labels_list = json.load(open("results/agentinstruct_task_type_labels.json"))
    traj_labels = {r["trajectory_id"]: r for r in traj_labels_list}
    traj_ids = list(traj_labels.keys())

    d = json.load(open("/tmp/proced_mem_bench_check/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajs = {t["task_instance_id"]: t for t in d["trajectories"]}

    def get_embedding_text(traj):
        text = f"Task: {traj['task_description']}\nSteps:\n"
        for p in traj["state_action_pairs"]:
            text += f"{p['step_id']}. State: {p['state']} -> Action: {p['action']}\n"
        return text.strip()

    embedders_info = [("labembed-Qwen3-8B", "labembed", 4096), ("gemini-embedding-001", "gemini", 3072)]
    traj_ci = {}
    rankings_cache = {}
    for ename, provider, dim in embedders_info:
        cache_dir = Path(f"embeddings_cache/{provider}")
        traj_cache = VectorCache(cache_dir, "step3_agentinstruct_traj", dim=dim, capacity=len(traj_ids))
        q_cache = VectorCache(cache_dir, "step3_agentinstruct_query", dim=dim, capacity=len(query_ids))
        traj_mat = traj_cache.get_matrix(traj_ids)
        q_mat = q_cache.get_matrix(query_ids)
        traj_mat = traj_mat / np.linalg.norm(traj_mat, axis=1, keepdims=True)
        q_mat = q_mat / np.linalg.norm(q_mat, axis=1, keepdims=True)
        sims = q_mat @ traj_mat.T
        qidx = {qid: i for i, qid in enumerate(query_ids)}
        rankings_cache[ename] = {qid: [traj_ids[i] for i in np.argsort(-sims[qidx[qid]])] for qid in query_ids}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    traj_text_list = [get_embedding_text(trajs[tid]) for tid in traj_ids]
    query_text_list = [query_labels[qid]["text"] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_traj = mini_traj / np.linalg.norm(mini_traj, axis=1, keepdims=True)
    mini_q = mini_q / np.linalg.norm(mini_q, axis=1, keepdims=True)
    sims = mini_q @ mini_traj.T
    qidx = {qid: i for i, qid in enumerate(query_ids)}
    rankings_cache["MiniLM-L6-v2"] = {qid: [traj_ids[i] for i in np.argsort(-sims[qidx[qid]])] for qid in query_ids}

    def ap_standard(ranked_ids, relevant_set):
        if not relevant_set:
            return None
        hits, psum = 0, 0.0
        for i, tid in enumerate(ranked_ids):
            if tid in relevant_set:
                hits += 1
                psum += hits / (i + 1)
        return psum / len(relevant_set)

    for ename, qrankings in rankings_cache.items():
        row = {}
        for tier_name in ["STRICT", "LENIENT"]:
            per_k_hits = {k: [] for k in [1, 5, 10]}
            aps = []
            for qid in query_ids:
                relevant = set(tiers[qid][tier_name])
                ranked = qrankings[qid]
                for k in [1, 5, 10]:
                    per_k_hits[k].append(any(tid in relevant for tid in ranked[:k]))
                ap = ap_standard(ranked, relevant)
                if ap is not None:
                    aps.append(ap)
            for k in [1, 5, 10]:
                row[f"{tier_name}_Hit@{k}"] = bootstrap_ci(per_k_hits[k])
            row[f"{tier_name}_MAP"] = bootstrap_ci(aps)

        # dumb reranker share-of-gap-closed CI, STRICT
        strict_before = [qrankings[qid][0] in set(tiers[qid]["STRICT"]) for qid in query_ids]
        dumb_after = []
        for qid in query_ids:
            strict_set = set(tiers[qid]["STRICT"])
            top10 = qrankings[qid][:10]
            qtext = query_labels[qid]["text"]
            scored = [(tid, dumb_score(qtext, trajs[tid]["task_description"])) for tid in top10]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            dumb_after.append(scored[0][0] in strict_set)
        orig_h1_boot, dumb_h1_boot = bootstrap_ratio_ci(strict_before, dumb_after)
        orig_h1_mean = sum(strict_before) / len(strict_before)
        dumb_h1_mean = sum(dumb_after) / len(dumb_after)
        orig_h10_mean = row["STRICT_Hit@10"]["mean"]
        gap = orig_h10_mean - orig_h1_mean
        share_closed_mean = (dumb_h1_mean - orig_h1_mean) / gap * 100 if gap != 0 else None
        gap_boot = orig_h10_mean - orig_h1_boot
        share_closed_boot = (dumb_h1_boot - orig_h1_boot) / gap_boot * 100
        share_closed_boot = share_closed_boot[np.isfinite(share_closed_boot)]
        lo, hi = np.percentile(share_closed_boot, [2.5, 97.5]) if len(share_closed_boot) else (None, None)
        row["dumb_reranker_share_closed_pct"] = {"mean": share_closed_mean, "ci_lo": float(lo) if lo is not None else None,
                                                   "ci_hi": float(hi) if hi is not None else None}

        traj_ci[ename] = row
        print(f"\n{ename}:")
        for tier_name in ["STRICT", "LENIENT"]:
            for k in [1, 5, 10]:
                c = row[f"{tier_name}_Hit@{k}"]
                print(f"  {tier_name} Hit@{k}: {c['mean']:.3f} [{c['ci_lo']:.3f},{c['ci_hi']:.3f}]")
            m = row[f"{tier_name}_MAP"]
            print(f"  {tier_name} MAP: {m['mean']:.3f} [{m['ci_lo']:.3f},{m['ci_hi']:.3f}]")
        dr = row["dumb_reranker_share_closed_pct"]
        print(f"  dumb-reranker share_closed: {dr['mean']:.1f}% [{dr['ci_lo']:.1f},{dr['ci_hi']:.1f}]" if dr['ci_lo'] is not None else "  dumb-reranker: n/a")

    all_results["trajectory_pooled_n118"] = traj_ci

    json.dump(all_results, open("results/task2b_bootstrap_cis.json", "w"), indent=2)
    print("\nSaved: results/task2b_bootstrap_cis.json")


if __name__ == "__main__":
    main()
