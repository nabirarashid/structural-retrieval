"""Task 1, steps 6-7: full trajectory pipeline on the expanded n=118 query
set (40 original + 78 new valid_unseen). Reports pooled, old-only, and
new-only breakdowns for every metric. Also slices the verb-only dumb
reranker by surface form (adjective: "put a hot X in Y" vs verb: "heat some
X and put it in Y") for task_types 3/4/5 -- the original 40 turn out to be
100% verb-form (zero adjective-form queries), so this comparison is only
meaningful because the new 78 introduce genuine phrasing diversity.
"""
import json
import re
import sys

import numpy as np

sys.path.insert(0, "src")
from dumb_reranker import dumb_score

REPO = "/tmp/proced_mem_bench_check"
N = 336
K_VALUES = [1, 5, 10]

ACTION_VERBS = {"put", "place", "find", "clean", "wash", "heat", "warm", "microwave",
                "cool", "chill", "examine", "look", "throw", "move"}
STOPWORDS = {"a", "an", "the", "and", "it", "them", "some", "in", "on", "to", "at",
             "with", "under", "of", "away"}
TOKEN_RE = re.compile(r"\w+")


def tokens(text):
    return set(TOKEN_RE.findall(text.lower()))


def verb_jaccard(a, b):
    ta, tb = tokens(a) & ACTION_VERBS, tokens(b) & ACTION_VERBS
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def surface_form(text, task_type):
    if task_type not in (3, 4, 5):
        return None
    if re.match(r"^(clean|wash|heat|warm|microwave|cool|chill)\b", text, re.I):
        return "verb"
    if re.match(r"^(put|place|move)\b", text, re.I):
        return "adjective"
    return "other"


def get_embedding_text(traj):
    text = f"Task: {traj['task_description']}\nSteps:\n"
    for p in traj["state_action_pairs"]:
        text += f"{p['step_id']}. State: {p['state']} -> Action: {p['action']}\n"
    return text.strip()


def hypergeometric_hit_prob(N, K, k):
    import math
    if K == 0:
        return 0.0
    if K >= N:
        return 1.0
    return 1.0 - (math.comb(N - K, k) / math.comb(N, k))


def average_precision_standard(ranked_ids, relevant_set):
    if not relevant_set:
        return None
    hits, precision_sum = 0, 0.0
    for i, tid in enumerate(ranked_ids):
        if tid in relevant_set:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant_set)


def hit_at_k(ranked_ids, relevant_set, k):
    return any(tid in relevant_set for tid in ranked_ids[:k])


def main():
    d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajectories = {t["task_instance_id"]: t for t in d["trajectories"]}
    traj_ids = list(trajectories.keys())

    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    query_labels = expanded["query_labels"]
    tiers = expanded["tiers"]
    query_ids = list(query_labels.keys())
    old_ids = [q for q in query_ids if query_labels[q]["provenance"] == "original"]
    new_ids = [q for q in query_ids if query_labels[q]["provenance"] != "original"]
    print(f"[rerun] n={len(query_ids)} total ({len(old_ids)} old, {len(new_ids)} new)")

    traj_texts = {tid: get_embedding_text(trajectories[tid]) for tid in traj_ids}
    query_texts = {qid: query_labels[qid]["text"] for qid in query_ids}

    # --- embeddings: labembed + gemini (~pennies, estimate then log) + MiniLM ---
    traj_items = {tid: traj_texts[tid] for tid in traj_ids}
    q_items = {qid: query_texts[qid] for qid in query_ids}

    total_qchars = sum(len(t) for t in query_texts.values())
    est_new_tokens = total_qchars // 4
    est_cost = est_new_tokens / 1e6 * 0.15
    print(f"[rerun] estimated gemini-embedding-001 cost for full query set (only NEW 78 will actually bill, "
          f"40 cached): ~${est_cost:.4f} upper bound", flush=True)

    from embed_labembed import embed_all as mc_embed_all
    print("[rerun] embedding labembed Qwen3-8B (free)...", flush=True)
    mc_traj_cache = mc_embed_all(traj_items, cache_name="step3_agentinstruct_traj")
    mc_q_cache = mc_embed_all(q_items, cache_name="step3_agentinstruct_query")  # reuses 40 cached, embeds 78 new

    from embed_gemini import embed_all as gem_embed_all
    print("[rerun] embedding gemini-embedding-001...", flush=True)
    gem_traj_cache = gem_embed_all(traj_items, task_type="RETRIEVAL_DOCUMENT", cache_name="step3_agentinstruct_traj")
    gem_q_cache = gem_embed_all(q_items, task_type="RETRIEVAL_QUERY", cache_name="step3_agentinstruct_query")

    from sentence_transformers import SentenceTransformer
    print("[rerun] embedding MiniLM-L6-v2 (free, local)...", flush=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    traj_text_list = [traj_texts[tid] for tid in traj_ids]
    query_text_list = [query_texts[qid] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)

    embedders = {
        "labembed-Qwen3-8B": (mc_traj_cache.get_matrix(traj_ids), mc_q_cache.get_matrix(query_ids)),
        "gemini-embedding-001": (gem_traj_cache.get_matrix(traj_ids), gem_q_cache.get_matrix(query_ids)),
        "MiniLM-L6-v2": (mini_traj, mini_q),
    }

    # record actual new-query embedding cost (gemini only bills for new items;
    # embed_gemini doesn't self-report, so log our own char-based estimate for the NEW portion only)
    from spend_tracker import record_call, spend_line
    new_qchars = sum(len(query_texts[qid]) for qid in new_ids)
    new_tokens_est = new_qchars // 4
    cost = record_call("gemini-embedding-001", new_tokens_est, 0,
                        note="task1 expanded query set, 78 new queries only (40 already cached)")
    print(f"[rerun] recorded actual new-query gemini cost: ${cost:.4f}")
    print(spend_line())

    subsets = {"pooled": query_ids, "old_40": old_ids, "new_78": new_ids}

    # --- compute full-similarity rankings ONCE per embedder, over ALL 118 queries ---
    rankings_by_embedder = {}
    sims_by_embedder = {}
    for ename, (traj_mat, q_mat) in embedders.items():
        traj_mat_n = traj_mat / np.linalg.norm(traj_mat, axis=1, keepdims=True)
        q_mat_n = q_mat / np.linalg.norm(q_mat, axis=1, keepdims=True)
        sims_by_embedder[ename] = q_mat_n @ traj_mat_n.T
        qidx = {qid: i for i, qid in enumerate(query_ids)}
        rankings = {}
        for qid in query_ids:
            order = np.argsort(-sims_by_embedder[ename][qidx[qid]])
            rankings[qid] = [traj_ids[i] for i in order[:10]]
        rankings_by_embedder[ename] = rankings

    all_out = {}
    for subset_name, subset_ids in subsets.items():
        print(f"\n{'='*20} SUBSET: {subset_name} (n={len(subset_ids)}) {'='*20}")
        subset_out = {"n": len(subset_ids)}

        # chance baseline
        chance = {"STRICT": {}, "LENIENT": {}}
        for tier_name in ["STRICT", "LENIENT"]:
            for k in K_VALUES:
                probs = [hypergeometric_hit_prob(N, len(tiers[qid][tier_name]), k) for qid in subset_ids]
                chance[tier_name][k] = sum(probs) / len(probs)
        subset_out["chance"] = chance
        print(f"chance STRICT: " + "  ".join(f"Hit@{k}={chance['STRICT'][k]:.3f}" for k in K_VALUES))
        print(f"chance LENIENT: " + "  ".join(f"Hit@{k}={chance['LENIENT'][k]:.3f}" for k in K_VALUES))

        embedder_metrics = {}
        for ename in embedders:
            sims = sims_by_embedder[ename]
            rankings = rankings_by_embedder[ename]
            qidx = {qid: i for i, qid in enumerate(query_ids)}

            per_tier = {}
            strict_misses_winners = []
            for tier_name in ["STRICT", "LENIENT"]:
                hits = {k: [] for k in K_VALUES}
                aps = []
                for qid in subset_ids:
                    relevant = set(tiers[qid][tier_name])
                    order = np.argsort(-sims[qidx[qid]])
                    ranked_ids = [traj_ids[i] for i in order]
                    if tier_name == "STRICT":
                        rankings[qid] = ranked_ids[:10]
                    for k in K_VALUES:
                        hits[k].append(hit_at_k(ranked_ids, relevant, k))
                    ap = average_precision_standard(ranked_ids, relevant)
                    if ap is not None:
                        aps.append(ap)
                    if tier_name == "STRICT":
                        top1 = ranked_ids[0]
                        if top1 not in relevant:
                            if top1 in set(tiers[qid]["SIBLING"]):
                                wc = "sibling"
                            elif top1 in set(tiers[qid]["NEAR_MISS"]):
                                wc = "near_miss"
                            else:
                                wc = "other"
                            strict_misses_winners.append({"query_id": qid, "winner_class": wc})
                per_tier[tier_name] = {f"Hit@{k}": sum(hits[k]) / len(hits[k]) for k in K_VALUES}
                per_tier[tier_name]["MAP"] = sum(aps) / len(aps) if aps else None

            embedder_metrics[ename] = per_tier
            rankings_by_embedder[ename] = rankings

            print(f"\n{ename}:")
            for tier_name in ["STRICT", "LENIENT"]:
                m = per_tier[tier_name]
                print(f"  {tier_name:8s}  Hit@1={m['Hit@1']:.3f}  Hit@5={m['Hit@5']:.3f}  "
                      f"Hit@10={m['Hit@10']:.3f}  MAP={m['MAP']:.3f}")
            n_miss = len(strict_misses_winners)
            counts = {"sibling": 0, "near_miss": 0, "other": 0}
            for m in strict_misses_winners:
                counts[m["winner_class"]] += 1
            print(f"  STRICT misses: {n_miss}/{len(subset_ids)}  " +
                  "  ".join(f"{k}={v}/{n_miss}({v/n_miss*100:.0f}%)" for k, v in counts.items() if n_miss))

            # dumb reranker: full mix + verb-only + noun-only, share of gap closed
            orig_h1 = per_tier["STRICT"]["Hit@1"]
            orig_h10 = per_tier["STRICT"]["Hit@10"]
            gap = orig_h10 - orig_h1
            dumb_row = {}
            for label, score_fn in [("full_mix", dumb_score), ("verb_only", verb_jaccard)]:
                hits1 = []
                for qid in subset_ids:
                    strict_set = set(tiers[qid]["STRICT"])
                    top10 = rankings[qid]
                    qtext = query_texts[qid]
                    scored = [(tid, score_fn(qtext, trajectories[tid]["task_description"])) for tid in top10]
                    scored.sort(key=lambda kv: kv[1], reverse=True)
                    hits1.append(scored[0][0] in strict_set)
                h1 = sum(hits1) / len(hits1)
                share = (h1 - orig_h1) / gap * 100 if gap != 0 else None
                dumb_row[label] = {"hit1": h1, "share_closed_pct": share}
            print(f"  dumb full-mix Hit@1={dumb_row['full_mix']['hit1']:.3f} "
                  f"share_closed={dumb_row['full_mix']['share_closed_pct']:.1f}%  |  "
                  f"verb-only Hit@1={dumb_row['verb_only']['hit1']:.3f} "
                  f"share_closed={dumb_row['verb_only']['share_closed_pct']:.1f}%")
            embedder_metrics[ename]["dumb_reranker"] = dumb_row

        subset_out["embedders"] = embedder_metrics
        all_out[subset_name] = subset_out

    # --- adjective-form vs verb-form slice, verb-only dumb reranker, STRICT ---
    print(f"\n{'='*20} SURFACE-FORM SLICE (types 3/4/5 only, verb-only dumb reranker) {'='*20}")
    form_groups = {"verb": [], "adjective": [], "other": []}
    for qid in query_ids:
        tt = query_labels[qid]["task_type"]
        form = surface_form(query_texts[qid], tt)
        if form:
            form_groups[form].append(qid)
    print(f"verb-form: {len(form_groups['verb'])} (old={sum(1 for q in form_groups['verb'] if q in old_ids)}, "
          f"new={sum(1 for q in form_groups['verb'] if q in new_ids)})")
    print(f"adjective-form: {len(form_groups['adjective'])} (old={sum(1 for q in form_groups['adjective'] if q in old_ids)}, "
          f"new={sum(1 for q in form_groups['adjective'] if q in new_ids)})")

    surface_results = {}
    for ename in embedders:
        rankings = rankings_by_embedder[ename]
        row = {}
        for form_name in ["verb", "adjective"]:
            ids = form_groups[form_name]
            if not ids:
                continue
            orig_hits = [rankings[qid][0] in set(tiers[qid]["STRICT"]) for qid in ids]
            orig_h1 = sum(orig_hits) / len(orig_hits)
            dumb_hits = []
            for qid in ids:
                strict_set = set(tiers[qid]["STRICT"])
                top10 = rankings[qid]
                qtext = query_texts[qid]
                scored = [(tid, verb_jaccard(qtext, trajectories[tid]["task_description"])) for tid in top10]
                scored.sort(key=lambda kv: kv[1], reverse=True)
                dumb_hits.append(scored[0][0] in strict_set)
            dumb_h1 = sum(dumb_hits) / len(dumb_hits)
            row[form_name] = {"n": len(ids), "orig_hit1": orig_h1, "verb_dumb_hit1": dumb_h1, "delta": dumb_h1 - orig_h1}
        surface_results[ename] = row
        print(f"\n{ename}:")
        for form_name, r in row.items():
            print(f"  {form_name:10s} n={r['n']:3d}  orig_Hit@1={r['orig_hit1']:.3f}  "
                  f"verb-only-dumb_Hit@1={r['verb_dumb_hit1']:.3f}  delta={r['delta']:+.3f}")

    json.dump({"subsets": all_out, "surface_form": surface_results,
               "form_group_ids": {k: v for k, v in form_groups.items()}},
              open("results/task1_expanded_full_results.json", "w"), indent=2)
    print("\nSaved: results/task1_expanded_full_results.json")


if __name__ == "__main__":
    main()
