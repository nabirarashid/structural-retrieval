"""Mechanism ablation for the Step 4 dumb-reranker sign flip (lexical
reranking HELPED here, unlike MathNet where it hurt). Hypothesis: it helps
because procedure/task_type is lexicalized in a small closed set of action
verbs (put/clean/heat/cool/examine/...), unlike MathNet where the solving
technique is never named in the problem text. Test: rerank by verb-token
Jaccard only vs. noun/object-token Jaccard only, compare share-of-gap-closed
against the full blended dumb_score (15.4% / 31.2% / 18.8%).

Design choice, stated explicitly rather than hidden: "two" (the multi-object
signal) is a quantifier, not a verb -- it's grouped with the noun/object
bucket here, not excluded. This means the noun bucket isn't purely object
identity; it also carries the type-6 structural signal. Reported as-is.
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, "src")

REPO = "/tmp/proced_mem_bench_check"
TOKEN_RE = re.compile(r"\w+")

ACTION_VERBS = {"put", "place", "find", "clean", "wash", "heat", "warm", "microwave",
                "cool", "chill", "examine", "look", "throw", "move"}
STOPWORDS = {"a", "an", "the", "and", "it", "them", "some", "in", "on", "to", "at",
             "with", "under", "of", "away"}


def tokens(text: str) -> set:
    return set(TOKEN_RE.findall(text.lower()))


def verb_jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a) & ACTION_VERBS, tokens(b) & ACTION_VERBS
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def noun_jaccard(a: str, b: str) -> float:
    ta = tokens(a) - ACTION_VERBS - STOPWORDS
    tb = tokens(b) - ACTION_VERBS - STOPWORDS
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def get_embedding_text(traj):
    text = f"Task: {traj['task_description']}\nSteps:\n"
    for p in traj["state_action_pairs"]:
        text += f"{p['step_id']}. State: {p['state']} -> Action: {p['action']}\n"
    return text.strip()


def main():
    d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajectories = {t["task_instance_id"]: t for t in d["trajectories"]}
    traj_ids = list(trajectories.keys())
    qd = json.load(open(f"{REPO}/procedural_memory_benchmark/benchmark/data/query_bank.json"))
    queries = {q["query_id"]: q for q in qd["queries"]}
    query_ids = list(queries.keys())

    tier_data = json.load(open("results/step3_tier_labels.json"))
    tiers = tier_data["tiers"]
    baseline = json.load(open("results/step3_baseline_results.json"))
    dumb_full = json.load(open("results/step4_dumb_reranker_control.json"))

    from pathlib import Path
    from vector_cache import VectorCache

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
        rankings[ename] = {query_ids[qi]: [traj_ids[i] for i in np.argsort(-sims[qi])[:10]] for qi in range(len(query_ids))}

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    traj_texts = {tid: get_embedding_text(trajectories[tid]) for tid in traj_ids}
    traj_text_list = [traj_texts[tid] for tid in traj_ids]
    query_text_list = [queries[qid]["query_text"] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_traj = mini_traj / np.linalg.norm(mini_traj, axis=1, keepdims=True)
    mini_q = mini_q / np.linalg.norm(mini_q, axis=1, keepdims=True)
    sims = mini_q @ mini_traj.T
    rankings["MiniLM-L6-v2"] = {query_ids[qi]: [traj_ids[i] for i in np.argsort(-sims[qi])[:10]] for qi in range(len(query_ids))}

    print("=== MECHANISM ABLATION: verb-only vs. noun/object-only lexical reranking ===")
    results = {}
    for ename, qrankings in rankings.items():
        orig_h1 = baseline["metrics"][ename]["STRICT"]["Hit@1"]
        orig_h10 = baseline["metrics"][ename]["STRICT"]["Hit@10"]
        gap = orig_h10 - orig_h1
        full_share = dumb_full[ename]["share_of_gap_closed_pct"]

        row = {"orig_hit1": orig_h1, "full_mix_share_closed_pct": full_share}
        for label, score_fn in [("verb_only", verb_jaccard), ("noun_only", noun_jaccard)]:
            hits = []
            for qid in query_ids:
                strict_set = set(tiers[qid]["STRICT"])
                top10 = qrankings[qid]
                qtext = queries[qid]["query_text"]
                scored = [(tid, score_fn(qtext, trajectories[tid]["task_description"])) for tid in top10]
                scored.sort(key=lambda kv: kv[1], reverse=True)
                hits.append(scored[0][0] in strict_set)
            h1 = sum(hits) / len(hits)
            share = (h1 - orig_h1) / gap * 100 if gap != 0 else None
            row[f"{label}_hit1"] = h1
            row[f"{label}_share_closed_pct"] = share

        results[ename] = row
        print(f"\n{ename} (orig Hit@1={orig_h1:.3f}, full-mix share closed={full_share:.1f}%):")
        print(f"  verb-only  Hit@1={row['verb_only_hit1']:.3f}  share_closed={row['verb_only_share_closed_pct']:.1f}%")
        print(f"  noun-only  Hit@1={row['noun_only_hit1']:.3f}  share_closed={row['noun_only_share_closed_pct']:.1f}%")

    json.dump(results, open("results/mechanism_ablation_results.json", "w"), indent=2)
    print("\nSaved: results/mechanism_ablation_results.json")


if __name__ == "__main__":
    main()
