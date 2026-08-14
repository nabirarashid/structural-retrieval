"""Step 0: exact hypergeometric chance baseline for STRICT/LENIENT Hit@k.
Step 4: dumb (lexical-only) reranker control on each embedder's top-10,
same design and dumb_score as the MathNet project (reused verbatim, not
reimplemented) -- rerank by query_text vs. trajectory task_description
lexical similarity alone, no semantics.
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "src")
from dumb_reranker import dumb_score

REPO = "/tmp/proced_mem_bench_check"
N = 336
K_VALUES = [1, 5, 10]


def hypergeometric_hit_prob(N: int, K: int, k: int) -> float:
    """Exact P(at least one of K gold items appears in a random k-sample
    from N), via math.comb -- not the large-N approximation."""
    if K == 0:
        return 0.0
    if K >= N:
        return 1.0
    return 1.0 - (math.comb(N - K, k) / math.comb(N, k))


def main():
    tier_data = json.load(open("results/step3_tier_labels.json"))
    tiers = tier_data["tiers"]
    query_ids = list(tiers.keys())

    # --- STEP 0: chance baseline ---
    print("=== STEP 0: EXACT HYPERGEOMETRIC CHANCE BASELINE (N=336, averaged over 40 queries) ===")
    chance = {"STRICT": {}, "LENIENT": {}}
    for tier_name in ["STRICT", "LENIENT"]:
        for k in K_VALUES:
            probs = [hypergeometric_hit_prob(N, len(tiers[qid][tier_name]), k) for qid in query_ids]
            chance[tier_name][k] = sum(probs) / len(probs)
        print(f"  {tier_name:8s}  " + "  ".join(f"Hit@{k}={chance[tier_name][k]:.3f}" for k in K_VALUES))

    baseline = json.load(open("results/step3_baseline_results.json"))
    print("\n=== ACTUAL vs. CHANCE, per embedder ===")
    for ename, metrics in baseline["metrics"].items():
        print(f"\n{ename}:")
        for tier_name in ["STRICT", "LENIENT"]:
            for k in K_VALUES:
                actual = metrics[tier_name][f"Hit@{k}"]
                ch = chance[tier_name][k]
                delta = actual - ch
                flag = " <-- BELOW CHANCE" if delta < 0 else ""
                print(f"  {tier_name:8s} Hit@{k:2d}: actual={actual:.3f}  chance={ch:.3f}  delta={delta:+.3f}{flag}")

    json.dump(chance, open("results/step0_chance_baseline.json", "w"), indent=2)

    # --- STEP 4: dumb reranker control ---
    print("\n\n=== STEP 4: DUMB RERANKER CONTROL (lexical-only, on each embedder's top-10) ===")

    d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajectories = {t["task_instance_id"]: t for t in d["trajectories"]}
    traj_ids = list(trajectories.keys())
    qd = json.load(open(f"{REPO}/procedural_memory_benchmark/benchmark/data/query_bank.json"))
    queries = {q["query_id"]: q for q in qd["queries"]}

    def get_embedding_text(traj):
        text = f"Task: {traj['task_description']}\nSteps:\n"
        for pair in traj["state_action_pairs"]:
            text += f"{pair['step_id']}. State: {pair['state']} -> Action: {pair['action']}\n"
        return text.strip()

    traj_texts = {tid: get_embedding_text(trajectories[tid]) for tid in traj_ids}

    from vector_cache import VectorCache
    from pathlib import Path

    embedders_cache = {
        "labembed-Qwen3-8B": ("labembed", 4096),
        "gemini-embedding-001": ("gemini", 3072),
    }

    rankings = {}  # ename -> {qid: [ranked traj_ids top-10]}
    for ename, (provider, dim) in embedders_cache.items():
        cache_dir = Path(f"embeddings_cache/{provider}")
        traj_cache = VectorCache(cache_dir, "step3_agentinstruct_traj", dim=dim, capacity=len(traj_ids))
        q_cache = VectorCache(cache_dir, "step3_agentinstruct_query", dim=dim, capacity=len(query_ids))
        traj_mat = traj_cache.get_matrix(traj_ids)
        q_mat = q_cache.get_matrix(query_ids)
        traj_mat = traj_mat / np.linalg.norm(traj_mat, axis=1, keepdims=True)
        q_mat = q_mat / np.linalg.norm(q_mat, axis=1, keepdims=True)
        sims = q_mat @ traj_mat.T
        rankings[ename] = {}
        for qi, qid in enumerate(query_ids):
            order = np.argsort(-sims[qi])[:10]
            rankings[ename][qid] = [traj_ids[i] for i in order]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    traj_text_list = [traj_texts[tid] for tid in traj_ids]
    query_text_list = [queries[qid]["query_text"] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_traj = mini_traj / np.linalg.norm(mini_traj, axis=1, keepdims=True)
    mini_q = mini_q / np.linalg.norm(mini_q, axis=1, keepdims=True)
    sims = mini_q @ mini_traj.T
    rankings["MiniLM-L6-v2"] = {}
    for qi, qid in enumerate(query_ids):
        order = np.argsort(-sims[qi])[:10]
        rankings["MiniLM-L6-v2"][qid] = [traj_ids[i] for i in order]

    dumb_results = {}
    for ename, qrankings in rankings.items():
        orig_hit1, dumb_hit1 = [], []
        for qid in query_ids:
            strict_set = set(tiers[qid]["STRICT"])
            top10 = qrankings[qid]
            orig_hit1.append(top10[0] in strict_set)

            qtext = queries[qid]["query_text"]
            scored = [(tid, dumb_score(qtext, trajectories[tid]["task_description"])) for tid in top10]
            scored.sort(key=lambda kv: kv[1], reverse=True)
            dumb_top1 = scored[0][0]
            dumb_hit1.append(dumb_top1 in strict_set)

        orig_h1 = sum(orig_hit1) / len(orig_hit1)
        dumb_h1 = sum(dumb_hit1) / len(dumb_hit1)
        orig_h10 = baseline["metrics"][ename]["STRICT"]["Hit@10"]
        gap = orig_h10 - orig_h1
        share_closed = (dumb_h1 - orig_h1) / gap * 100 if gap != 0 else None

        dumb_results[ename] = {
            "orig_hit1": orig_h1, "dumb_hit1": dumb_h1,
            "orig_hit10": orig_h10, "recoverable_gap_pts": gap * 100,
            "share_of_gap_closed_pct": share_closed,
        }
        print(f"\n{ename}:")
        print(f"  STRICT Hit@1: orig={orig_h1:.3f}  dumb-reranked={dumb_h1:.3f}  "
              f"delta={dumb_h1-orig_h1:+.3f}")
        print(f"  recoverable gap (Hit@10-Hit@1): {gap*100:.1f}pts  "
              f"share closed: {share_closed:.1f}%" if share_closed is not None else "  gap=0")

    json.dump(dumb_results, open("results/step4_dumb_reranker_control.json", "w"), indent=2)
    print("\nSaved: results/step0_chance_baseline.json, results/step4_dumb_reranker_control.json")


if __name__ == "__main__":
    main()
