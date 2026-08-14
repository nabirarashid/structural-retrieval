"""Task 2A: tier-definition robustness for the below-chance finding.
Recompute actual-vs-chance STRICT Hit@k under three definitions:
  (i)   frozen: same task_type, different target object
  (ii)  harsher: same task_type, different object AND different goal_receptacle
        (type-2 examine items have no receptacle at all, so (ii) falls back
        to (i) for those -- the receptacle constraint isn't applicable)
  (iii) lenient: same task_type, any object (reference point, = existing LENIENT tier)
Reports on the pooled n=118 set (the frozen final query set).
"""
import json
import math

N = 336
K_VALUES = [1, 5, 10]


def hypergeometric_hit_prob(N, K, k):
    if K == 0:
        return 0.0
    if K >= N:
        return 1.0
    return 1.0 - (math.comb(N - K, k) / math.comb(N, k))


def hit_at_k(ranked_ids, relevant_set, k):
    return any(tid in relevant_set for tid in ranked_ids[:k])


def main():
    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    query_labels = expanded["query_labels"]
    query_ids = list(query_labels.keys())

    traj_labels_list = json.load(open("results/agentinstruct_task_type_labels.json"))
    traj_labels = {r["trajectory_id"]: r for r in traj_labels_list}
    traj_ids = list(traj_labels.keys())

    # rebuild STRICT-harsher tier (ii) directly from labels (goal_receptacle now available both sides)
    strict_harsher = {}
    for qid in query_ids:
        qlabel = query_labels[qid]
        q_recep = qlabel.get("goal_receptacle")
        bucket = set()
        for tid in traj_ids:
            tlabel = traj_labels[tid]
            same_type = qlabel["task_type"] == tlabel["task_type"]
            same_obj = tlabel["target_object"] in qlabel["target_objects"]
            if not same_type or same_obj:
                continue
            if qlabel["task_type"] == 2:  # examine -- no receptacle concept, falls back to (i)
                bucket.add(tid)
                continue
            t_recep = tlabel.get("goal_receptacle")
            if q_recep is None or t_recep is None or q_recep != t_recep:
                bucket.add(tid)
        strict_harsher[qid] = bucket

    definitions = {
        "(i) frozen": expanded["tiers"],  # use tiers[qid]['STRICT']
        "(ii) harsher (+diff receptacle)": {qid: {"STRICT": strict_harsher[qid]} for qid in query_ids},
        "(iii) lenient (reference)": expanded["tiers"],  # use tiers[qid]['LENIENT']
    }

    def get_set(defname, qid):
        if defname == "(i) frozen":
            return set(expanded["tiers"][qid]["STRICT"])
        if defname == "(ii) harsher (+diff receptacle)":
            return strict_harsher[qid]
        return set(expanded["tiers"][qid]["LENIENT"])

    print("=== TASK 2A: tier-definition robustness, pooled n=118 ===\n")
    chance_by_def = {}
    for defname in definitions:
        chance = {}
        for k in K_VALUES:
            probs = [hypergeometric_hit_prob(N, len(get_set(defname, qid)), k) for qid in query_ids]
            chance[k] = sum(probs) / len(probs)
        chance_by_def[defname] = chance
        sizes = [len(get_set(defname, qid)) for qid in query_ids]
        print(f"{defname}: mean gold size={sum(sizes)/len(sizes):.1f}  "
              f"chance Hit@1={chance[1]:.3f} Hit@5={chance[5]:.3f} Hit@10={chance[10]:.3f}")

    # actual retrieval rankings, reuse cached embeddings
    import sys
    sys.path.insert(0, "src")
    from pathlib import Path
    from vector_cache import VectorCache
    import numpy as np

    rankings = {}
    for ename, provider, dim in [("labembed-Qwen3-8B", "labembed", 4096),
                                  ("gemini-embedding-001", "gemini", 3072)]:
        cache_dir = Path(f"embeddings_cache/{provider}")
        traj_cache = VectorCache(cache_dir, "step3_agentinstruct_traj", dim=dim, capacity=len(traj_ids))
        q_cache = VectorCache(cache_dir, "step3_agentinstruct_query", dim=dim, capacity=len(query_ids))
        traj_mat = traj_cache.get_matrix(traj_ids)
        q_mat = q_cache.get_matrix(query_ids)
        traj_mat = traj_mat / np.linalg.norm(traj_mat, axis=1, keepdims=True)
        q_mat = q_mat / np.linalg.norm(q_mat, axis=1, keepdims=True)
        sims = q_mat @ traj_mat.T
        rankings[ename] = {query_ids[qi]: [traj_ids[i] for i in np.argsort(-sims[qi])] for qi in range(len(query_ids))}

    from sentence_transformers import SentenceTransformer
    d = json.load(open("/tmp/proced_mem_bench_check/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajs = {t["task_instance_id"]: t for t in d["trajectories"]}

    def get_embedding_text(traj):
        text = f"Task: {traj['task_description']}\nSteps:\n"
        for p in traj["state_action_pairs"]:
            text += f"{p['step_id']}. State: {p['state']} -> Action: {p['action']}\n"
        return text.strip()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    traj_text_list = [get_embedding_text(trajs[tid]) for tid in traj_ids]
    query_text_list = [query_labels[qid]["text"] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_traj = mini_traj / np.linalg.norm(mini_traj, axis=1, keepdims=True)
    mini_q = mini_q / np.linalg.norm(mini_q, axis=1, keepdims=True)
    sims = mini_q @ mini_traj.T
    rankings["MiniLM-L6-v2"] = {query_ids[qi]: [traj_ids[i] for i in np.argsort(-sims[qi])] for qi in range(len(query_ids))}

    results = {}
    for ename, qrankings in rankings.items():
        print(f"\n{ename}:")
        results[ename] = {}
        for defname in definitions:
            actual = {}
            for k in K_VALUES:
                hits = [hit_at_k(qrankings[qid], get_set(defname, qid), k) for qid in query_ids]
                actual[k] = sum(hits) / len(hits)
            results[ename][defname] = actual
            below = all(actual[k] < chance_by_def[defname][k] for k in K_VALUES)
            flag = " <-- BELOW CHANCE AT ALL k" if below else ""
            print(f"  {defname:32s} actual Hit@1={actual[1]:.3f}(chance {chance_by_def[defname][1]:.3f})  "
                  f"Hit@5={actual[5]:.3f}({chance_by_def[defname][5]:.3f})  "
                  f"Hit@10={actual[10]:.3f}({chance_by_def[defname][10]:.3f}){flag}")

    json.dump({"chance_by_definition": chance_by_def, "actual_by_embedder": results},
              open("results/task2a_tier_robustness.json", "w"), indent=2)
    print("\nSaved: results/task2a_tier_robustness.json")


if __name__ == "__main__":
    main()
