"""Step 3: modern embedder baseline for the procedural-memory domain.

Gold definition (frozen, mirrors the MathNet-Retrieve tier design):
  STRICT:     same task_type, DIFFERENT target_object  (cross-object structural transfer)
  SIBLING:    same task_type, SAME target_object type   (incl. same-description-group trajectories)
  NEAR-MISS:  DIFFERENT task_type, SAME target_object   (the deliberate surface-similarity decoy)
  LENIENT:    STRICT union SIBLING (any same-task_type trajectory)

Query labels: task_type remapped from released query_type per the established
6-way mapping, with ONE manual override -- medium_13 ("Chill a potato and
place it in the microwave") is released as query_type=heating but its
explicit verb is an unambiguous cooling signal ("chill"); relabeled to
task_type=5 (cooling). This was the only conflict found when cross-checking
all 40 queries against explicit verb signals (see prior turn's diagnostic).

target_object extracted from query_text via regex (one manual override for
hard_9's composite-object phrasing, "Place a cup with a pen in it on the
desk" -> target=cup), then mapped to the corpus's canonical ALFWorld object
names via an alias table. 'bottle' is deliberately ambiguous -- it aliases
to all three corpus bottle-type objects (glassbottle/soapbottle/spraybottle),
per the Step 2 diagnostic showing GPT-5 confirmed all three as relevant to a
generic "bottle" query.
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, "src")

REPO = "/tmp/proced_mem_bench_check"

TASK_TYPE_NAMES = {
    1: "pick_and_place_simple", 2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep", 4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep", 6: "pick_two_obj_and_place",
}
QUERY_TYPE_TO_TASK_TYPE = {
    "placement": 1, "examination": 2, "cleaning": 3,
    "heating": 4, "cooling": 5, "multi_object": 6,
}
TASK_TYPE_OVERRIDES = {"medium_13": 5}  # 'Chill...' mislabeled 'heating' in the release

OBJECT_ALIAS = {
    "soap bar": {"soapbar"}, "soap bars": {"soapbar"},
    "bottle": {"glassbottle", "soapbottle", "spraybottle"},
    "bottles": {"glassbottle", "soapbottle", "spraybottle"},
    "mug": {"mug"}, "keychain": {"keychain"}, "keychains": {"keychain"},
    "tomato": {"tomato"}, "potato": {"potato"}, "phone": {"cellphone"},
    "cup": {"cup"}, "remote control": {"remotecontrol"}, "remote controls": {"remotecontrol"},
    "pen": {"pen"}, "pens": {"pen"}, "alarm clock": {"alarmclock"},
    "cellphone": {"cellphone"}, "book": {"book"}, "egg": {"egg"},
    "lettuce": {"lettuce"}, "apple": {"apple"},
}

TARGET_OVERRIDES = {"hard_9": "cup"}
ART = r"(?:(?:an|a|some)\b\s*)?"
GOAL_PATTERNS = [
    re.compile(r"^(?:put|find|place) two (.+?) and put them in (?:the )?(.+?)\.?$", re.I),
    re.compile(r"^put two (.+?) in (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:examine|look at) {ART}(.+?) (?:with|under) the (?:desk ?lamp|desklamp).*$", re.I),
    re.compile(rf"^(?:clean|wash) {ART}(.+?) and (?:throw it away)\.?$", re.I),
    re.compile(rf"^(?:clean|wash) {ART}(.+?) and (?:put|place) it (?:in|on) (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:heat|warm|microwave) {ART}(.+?) and (?:put|place|throw) it (?:in|on) (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:cool|chill) {ART}(.+?) and (?:put|place) it (?:in|on) (?:the )?(.+?)\.?$", re.I),
    re.compile(rf"^(?:put|place|move) {ART}(.+?) (?:in|on|to) (?:the )?(.+?)\.?$", re.I),
]


def extract_target(text: str) -> str:
    for pat in GOAL_PATTERNS:
        m = pat.match(text)
        if m:
            return m.group(1).strip()
    return None


def build_query_labels(queries: list) -> dict:
    labels = {}
    for q in queries:
        qid = q["query_id"]
        task_type = TASK_TYPE_OVERRIDES.get(qid, QUERY_TYPE_TO_TASK_TYPE[q["query_type"]])
        raw_target = TARGET_OVERRIDES.get(qid, extract_target(q["query_text"]))
        obj_set = OBJECT_ALIAS.get(raw_target)
        if obj_set is None:
            raise ValueError(f"{qid}: no object alias for target {raw_target!r} (text={q['query_text']!r})")
        labels[qid] = {"task_type": task_type, "target_objects": obj_set, "tier": q["tier"], "text": q["query_text"]}
    return labels


def classify_tier(query_label: dict, traj_label: dict) -> str:
    same_type = query_label["task_type"] == traj_label["task_type"]
    same_obj = traj_label["target_object"] in query_label["target_objects"]
    if same_type and not same_obj:
        return "STRICT"
    if same_type and same_obj:
        return "SIBLING"
    if not same_type and same_obj:
        return "NEAR_MISS"
    return "OTHER"


def get_embedding_text(traj: dict) -> str:
    text = f"Task: {traj['task_description']}\n"
    text += "Steps:\n"
    for pair in traj["state_action_pairs"]:
        text += f"{pair['step_id']}. State: {pair['state']} -> Action: {pair['action']}\n"
    return text.strip()


def average_precision_standard(ranked_ids: list, relevant_set: set) -> float:
    """Standard AP: normalizes by TOTAL relevant count for the query, not
    just relevant-found -- unlike the non-standard formula used to
    reproduce the paper's Table 2 in Step 2. This is our own metric now."""
    if not relevant_set:
        return None
    hits, precision_sum = 0, 0.0
    for i, tid in enumerate(ranked_ids):
        if tid in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant_set)


def hit_at_k(ranked_ids: list, relevant_set: set, k: int) -> bool:
    return any(tid in relevant_set for tid in ranked_ids[:k])


def main():
    d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajectories = d["trajectories"]
    qd = json.load(open(f"{REPO}/procedural_memory_benchmark/benchmark/data/query_bank.json"))
    queries = qd["queries"]

    traj_label_list = json.load(open("results/agentinstruct_task_type_labels.json"))
    traj_labels = {r["trajectory_id"]: r for r in traj_label_list}
    traj_ids = [t["task_instance_id"] for t in trajectories]

    query_labels = build_query_labels(queries)

    # tier membership matrix: query_id -> {STRICT: set(traj_ids), SIBLING: ..., NEAR_MISS: ..., LENIENT: ...}
    tiers = {}
    for qid, qlabel in query_labels.items():
        buckets = {"STRICT": set(), "SIBLING": set(), "NEAR_MISS": set(), "OTHER": set()}
        for tid in traj_ids:
            tier = classify_tier(qlabel, traj_labels[tid])
            buckets[tier].add(tid)
        buckets["LENIENT"] = buckets["STRICT"] | buckets["SIBLING"]
        tiers[qid] = buckets

    print("=== TIER SIZES (mean across 40 queries) ===")
    for tier in ["STRICT", "SIBLING", "NEAR_MISS", "LENIENT"]:
        sizes = [len(tiers[qid][tier]) for qid in query_labels]
        print(f"  {tier:10s} mean={sum(sizes)/len(sizes):.1f}  min={min(sizes)}  max={max(sizes)}  "
              f"zero_count={sum(1 for s in sizes if s == 0)}/40")

    json.dump(
        {"query_labels": {qid: {**v, "target_objects": list(v["target_objects"])} for qid, v in query_labels.items()},
         "tiers": {qid: {t: sorted(s) for t, s in b.items()} for qid, b in tiers.items()}},
        open("results/step3_tier_labels.json", "w"), indent=2,
    )
    print("Saved tier labels to results/step3_tier_labels.json")

    # --- embeddings: labembed (free) + gemini-embedding-001 (~$0.02) + MiniLM (free, local) ---
    traj_texts = [get_embedding_text(t) for t in trajectories]
    query_texts = [q["query_text"] for q in queries]
    query_ids = [q["query_id"] for q in queries]

    embedders = {}

    from embed_labembed import embed_all as mc_embed_all
    print("\n[embed] labembed Qwen3-8B...", flush=True)
    traj_items = {tid: txt for tid, txt in zip(traj_ids, traj_texts)}
    q_items = {qid: txt for qid, txt in zip(query_ids, query_texts)}
    mc_traj_cache = mc_embed_all(traj_items, cache_name="step3_agentinstruct_traj")
    mc_q_cache = mc_embed_all(q_items, cache_name="step3_agentinstruct_query")
    embedders["labembed-Qwen3-8B"] = (mc_traj_cache.get_matrix(traj_ids), mc_q_cache.get_matrix(query_ids))

    from embed_gemini import embed_all as gem_embed_all
    print("\n[embed] gemini-embedding-001 (~$0.02)...", flush=True)
    gem_traj_cache = gem_embed_all(traj_items, task_type="RETRIEVAL_DOCUMENT", cache_name="step3_agentinstruct_traj")
    gem_q_cache = gem_embed_all(q_items, task_type="RETRIEVAL_QUERY", cache_name="step3_agentinstruct_query")
    embedders["gemini-embedding-001"] = (gem_traj_cache.get_matrix(traj_ids), gem_q_cache.get_matrix(query_ids))

    from sentence_transformers import SentenceTransformer
    print("\n[embed] all-MiniLM-L6-v2 (free, local)...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    mini_traj = model.encode(traj_texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    mini_q = model.encode(query_texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    embedders["MiniLM-L6-v2"] = (mini_traj, mini_q)

    # --- score Hit@1/5/10 + MAP under STRICT and LENIENT, per embedder ---
    K_VALUES = [1, 5, 10]
    all_results = {}
    failure_taxonomy = {}

    for ename, (traj_mat, q_mat) in embedders.items():
        traj_mat_n = traj_mat / np.linalg.norm(traj_mat, axis=1, keepdims=True)
        q_mat_n = q_mat / np.linalg.norm(q_mat, axis=1, keepdims=True)
        sims = q_mat_n @ traj_mat_n.T  # (40, 336)

        per_tier_metrics = {}
        strict_misses_winners = []
        for tier_name in ["STRICT", "LENIENT"]:
            hits = {k: [] for k in K_VALUES}
            aps = []
            for qi, qid in enumerate(query_ids):
                relevant = tiers[qid][tier_name]
                order = np.argsort(-sims[qi])
                ranked_ids = [traj_ids[i] for i in order]
                for k in K_VALUES:
                    hits[k].append(hit_at_k(ranked_ids, relevant, k))
                ap = average_precision_standard(ranked_ids, relevant)
                if ap is not None:
                    aps.append(ap)

                if tier_name == "STRICT":
                    top1 = ranked_ids[0]
                    if top1 not in relevant:
                        if top1 in tiers[qid]["SIBLING"]:
                            winner_class = "sibling"
                        elif top1 in tiers[qid]["NEAR_MISS"]:
                            winner_class = "near_miss"
                        else:
                            winner_class = "other"
                        strict_misses_winners.append({"query_id": qid, "winner": top1, "winner_class": winner_class})

            per_tier_metrics[tier_name] = {
                f"Hit@{k}": sum(hits[k]) / len(hits[k]) for k in K_VALUES
            }
            per_tier_metrics[tier_name]["MAP"] = sum(aps) / len(aps) if aps else None

        all_results[ename] = per_tier_metrics
        failure_taxonomy[ename] = strict_misses_winners

    print("\n=== STEP 3 RESULTS: Hit@k and MAP, STRICT vs LENIENT gold ===")
    for ename, metrics in all_results.items():
        print(f"\n{ename}:")
        for tier_name in ["STRICT", "LENIENT"]:
            m = metrics[tier_name]
            print(f"  {tier_name:8s}  Hit@1={m['Hit@1']:.3f}  Hit@5={m['Hit@5']:.3f}  "
                  f"Hit@10={m['Hit@10']:.3f}  MAP={m['MAP']:.3f}")

    print("\n=== FAILURE TAXONOMY at STRICT rank-1 (winner classification) ===")
    for ename, misses in failure_taxonomy.items():
        n = len(misses)
        counts = {"sibling": 0, "near_miss": 0, "other": 0}
        for m in misses:
            counts[m["winner_class"]] += 1
        print(f"\n{ename}: {n}/40 STRICT misses at rank 1")
        for cls, c in counts.items():
            pct = c / n * 100 if n else 0
            print(f"  {cls:10s} {c:3d}/{n} ({pct:.1f}%)")

    json.dump({"metrics": all_results, "failure_taxonomy": failure_taxonomy},
              open("results/step3_baseline_results.json", "w"), indent=2)
    print("\nSaved full results to results/step3_baseline_results.json")


if __name__ == "__main__":
    main()
