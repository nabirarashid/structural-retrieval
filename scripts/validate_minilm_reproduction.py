"""Validation gate: reproduce the paper's own state-aware all-MiniLM-L6-v2
pipeline on their released data, scored against their released LLM-judge
labels, and check we land near their published Table 2 MAP (0.7945 overall;
0.842 EASY / 0.746 MEDIUM / 0.791 HARD).

Faithfulness details, copied from their own source (not reinvented):
  - Embedding text format: AgentInstructTrajectory.get_embedding_text() --
    "Task: {task_description}\\nSteps:\\n{step_id}. State: {state} -> Action:
    {action}\\n..." (procedural_memory_benchmark/agentinstruct/corpus_loader.py)
  - Retrieval: top-k=10 by cosine similarity (benchmark_runner.py:
    `retrieve(query, k=max(self.k_values))`, k_values defaults to [1,3,5,10])
  - AP formula: their MetricsCalculator._calculate_average_precision is
    NON-STANDARD -- it normalizes by the count of relevant items FOUND
    within the top-10, not by the total relevant count for the query. This
    matters and is reproduced exactly, not replaced with textbook AP.

Deliberate deviation from their pipeline, per instruction: relevance comes
from the RELEASED, PRE-COMPUTED query_bank.json judgments (threshold
score>=6.0), not from a fresh GPT-5 call on whatever we retrieve. Their own
runner calls the LLM fresh on each run's top-10, which would need new GPT-5
spend to replicate exactly. Using the released judgments as a fixed qrels
set is standard IR practice and keeps this step free -- but it means a
trajectory outside the originally keyword-matched/judged candidate pool for
a query is scored as not-relevant even if a human might call it relevant.
If our MAP comes out well off Table 2, this is one thing to check (overlap
between our top-10 and the originally-judged pool).
"""
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

REPO = "/tmp/proced_mem_bench_check"

d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
trajectories = d["trajectories"]

qd = json.load(open(f"{REPO}/procedural_memory_benchmark/benchmark/data/query_bank.json"))
queries = qd["queries"]


def get_embedding_text(traj: dict) -> str:
    text = f"Task: {traj['task_description']}\n"
    text += "Steps:\n"
    for pair in traj["state_action_pairs"]:
        text += f"{pair['step_id']}. State: {pair['state']} -> Action: {pair['action']}\n"
    return text.strip()


print(f"[validate] loading all-MiniLM-L6-v2 and encoding {len(trajectories)} trajectories...", flush=True)
model = SentenceTransformer("all-MiniLM-L6-v2")

traj_ids = [t["task_instance_id"] for t in trajectories]
traj_texts = [get_embedding_text(t) for t in trajectories]
traj_embs = model.encode(traj_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
traj_embs = traj_embs / np.linalg.norm(traj_embs, axis=1, keepdims=True)

print(f"[validate] encoding {len(queries)} queries...", flush=True)
query_texts = [q["query_text"] for q in queries]
query_embs = model.encode(query_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)

K = 10
RELEVANCE_THRESHOLD = 6.0


def average_precision(retrieved_ids: list, relevance_judgments: dict) -> float:
    """Exact copy of their MetricsCalculator._calculate_average_precision:
    normalizes by relevant items FOUND in retrieved_ids, not by total
    relevant count for the query -- non-standard, reproduced intentionally."""
    relevant_positions = []
    for i, tid in enumerate(retrieved_ids):
        if relevance_judgments.get(tid, False):
            relevant_positions.append(i + 1)
    if not relevant_positions:
        return 0.0
    precision_sum = 0.0
    for pos in relevant_positions:
        relevant_up_to_pos = sum(1 for p in relevant_positions if p <= pos)
        precision_sum += relevant_up_to_pos / pos
    return precision_sum / len(relevant_positions)


results = []
for qi, q in enumerate(queries):
    rel_map = {r["trajectory_id"]: r["relevance_score"] for r in q["relevant_trajectories"]}
    relevance_judgments = {tid: (score >= RELEVANCE_THRESHOLD) for tid, score in rel_map.items()}

    sims = traj_embs @ query_embs[qi]
    order = np.argsort(-sims)[:K]
    retrieved_ids = [traj_ids[i] for i in order]

    ap = average_precision(retrieved_ids, relevance_judgments)
    n_relevant_in_pool = sum(1 for v in relevance_judgments.values() if v)
    n_relevant_retrieved = sum(1 for tid in retrieved_ids if relevance_judgments.get(tid, False))

    results.append({
        "query_id": q["query_id"], "tier": q["tier"], "ap": ap,
        "n_relevant_in_judged_pool": n_relevant_in_pool,
        "n_relevant_retrieved_in_top10": n_relevant_retrieved,
        "retrieved_ids": retrieved_ids,
    })

overall_map = sum(r["ap"] for r in results) / len(results)
by_tier = {}
for tier in ["EASY", "MEDIUM", "HARD"]:
    tier_results = [r for r in results if r["tier"] == tier]
    by_tier[tier] = sum(r["ap"] for r in tier_results) / len(tier_results) if tier_results else None

print("\n=== VALIDATION GATE: our reproduction vs. paper's Table 2 ===")
print(f"{'':20s} {'ours':>10s} {'paper':>10s} {'delta':>8s}")
print(f"{'Overall MAP':20s} {overall_map:10.4f} {0.7945:10.4f} {overall_map-0.7945:8.4f}")
for tier, paper_val in [("EASY", 0.842), ("MEDIUM", 0.746), ("HARD", 0.791)]:
    ours = by_tier[tier]
    print(f"{tier+' MAP':20s} {ours:10.4f} {paper_val:10.4f} {ours-paper_val:8.4f}")

zero_ap_queries = [r for r in results if r["ap"] == 0.0]
print(f"\nQueries with AP=0.0 (no judged-relevant trajectory in top-10): {len(zero_ap_queries)}/{len(results)}")
for r in zero_ap_queries[:10]:
    print(f"  {r['query_id']} ({r['tier']}): pool_relevant={r['n_relevant_in_judged_pool']}, retrieved_relevant=0")

json.dump({
    "overall_map": overall_map, "by_tier": by_tier,
    "paper_overall": 0.7945, "paper_by_tier": {"EASY": 0.842, "MEDIUM": 0.746, "HARD": 0.791},
    "per_query": [{k: v for k, v in r.items() if k != "retrieved_ids"} for r in results],
}, open("results/minilm_validation_gate.json", "w"), indent=2)
print("\nSaved full results to results/minilm_validation_gate.json")
