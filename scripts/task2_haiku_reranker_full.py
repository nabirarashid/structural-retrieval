"""Task 2: Claude Haiku 4.5 as a third reranker judge -- full run, terse
prompt only, same protocol as the existing two judges (Gemini, GLM).
Math: 500 queries x 4 configs (2 embed providers x 2 tiers), reusing the
existing top10_ids from the frozen full_{provider}_{tier}.jsonl caches.
Trajectories: 118 queries x 3 embedders, same setup as
scripts/task_traj_reranker_n118.py.

10-call pilot (scripts/task2_haiku_pilot.py) already confirmed plain terse
output (stop_reason=end_turn, no capping) before this ran. Pre-estimate
from real probe calls: ~$5.46 total, under the $15 stop threshold.

Hard call cap: 2800 (real need is 2354; this is a backstop, not expected
to trigger). Truncation/finish_reason audited before any accuracy number
is reported, per standing rule.

Conflict-of-interest note for the record: Claude (Anthropic) authored
nothing being judged here -- the candidates under review are MathNet
problems/reformulations and AgentInstruct/ALFWorld trajectories, and the
other two judges are Gemini and GLM. Haiku is being scored purely as a
third independent reranker.
"""
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from llm_judge_haiku import call_judge, MODEL as HAIKU_MODEL
from llm_reranker import PROMPT_TEMPLATE as MATH_PROMPT_TEMPLATE
from data import MathNetEasy
from spend_tracker import record_call, spend_line
from step5_llm_reranker import PROMPT_TEMPLATE as TRAJ_PROMPT_TEMPLATE, get_embedding_text

JUDGE_MAX_TOKENS = 16
WORKERS = 8
HARD_CALL_CAP = 2800

_call_lock = threading.Lock()
_call_count = 0


def _guarded_call(prompt):
    global _call_count
    with _call_lock:
        _call_count += 1
        if _call_count > HARD_CALL_CAP:
            raise RuntimeError(f"HARD_CALL_CAP ({HARD_CALL_CAP}) exceeded -- stopping")
    j = call_judge(prompt, max_tokens=JUDGE_MAX_TOKENS)
    cost = record_call(HAIKU_MODEL, j["input_tokens"], j["output_tokens"], note="task2 haiku full")
    return j, cost


def parse_choice(text):
    import re
    m = re.search(r"\b([1-9]|10)\b", text)
    return int(m.group(1)) if m else None


def run_math():
    print("[task2-haiku] === MATH DOMAIN (500 x 4 configs) ===", flush=True)
    all_records = []
    for provider, tier in [("gemini", "easy"), ("gemini", "hard"), ("deepinfra", "easy"), ("deepinfra", "hard")]:
        ds = MathNetEasy.load(tier=tier)
        cache_path = f"llm_reranker_cache/full_haiku_{provider}_{tier}.jsonl"
        cache = {}
        if os.path.exists(cache_path):
            for line in open(cache_path):
                r = json.loads(line)
                cache[r["query_id"]] = r

        src_recs = [json.loads(l) for l in open(f"llm_reranker_cache/full_{provider}_{tier}.jsonl")]
        todo = [r for r in src_recs if r["query_id"] not in cache]
        for r in src_recs:
            if r["query_id"] in cache:
                all_records.append(cache[r["query_id"]])

        out_f = open(cache_path, "a")
        write_lock = threading.Lock()
        done = [0]

        def process(r):
            qid = r["query_id"]
            top10 = r["top10_ids"]
            candidates_text = "\n".join(f"[{i+1}] {ds.corpus[cid]}" for i, cid in enumerate(top10))
            prompt = MATH_PROMPT_TEMPLATE.format(anchor=ds.queries[qid], candidates=candidates_text)
            j, cost = _guarded_call(prompt)
            choice_idx = parse_choice(j["text"])
            chosen_id = top10[choice_idx - 1] if choice_idx else None
            capped = j["output_tokens"] >= JUDGE_MAX_TOKENS
            record = {"query_id": qid, "raw_response": j["text"], "choice_idx": choice_idx,
                      "chosen_id": chosen_id, "top10_ids": top10, "completion_tokens": j["output_tokens"],
                      "stop_reason": j["stop_reason"], "capped": capped, "provider": provider, "tier": tier}
            with write_lock:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                all_records.append(record)
                done[0] += 1
                if done[0] % 100 == 0:
                    print(f"[task2-haiku] math {provider}/{tier}: {done[0]}/{len(todo)} -- {spend_line()}", flush=True)
            return record

        if todo:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                list(ex.map(process, todo))
        out_f.close()
        print(f"[task2-haiku] math {provider}/{tier}: complete ({len(todo)} new, {len(cache)+len(todo)} total)", flush=True)
    return all_records


def build_traj_rankings(query_ids, query_labels):
    d = json.load(open("/tmp/proced_mem_bench_check/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajs = {t["task_instance_id"]: t for t in d["trajectories"]}
    traj_ids = list(trajs.keys())
    traj_texts = {tid: get_embedding_text(trajs[tid]) for tid in traj_ids}

    rankings = {}
    from vector_cache import VectorCache
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
    traj_text_list = [traj_texts[tid] for tid in traj_ids]
    query_text_list = [query_labels[qid]["text"] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_traj = mini_traj / np.linalg.norm(mini_traj, axis=1, keepdims=True)
    mini_q = mini_q / np.linalg.norm(mini_q, axis=1, keepdims=True)
    sims = mini_q @ mini_traj.T
    rankings["MiniLM-L6-v2"] = {query_ids[qi]: [traj_ids[i] for i in np.argsort(-sims[qi])[:10]] for qi in range(len(query_ids))}
    return rankings, traj_texts


def run_trajectories():
    print("\n[task2-haiku] === TRAJECTORY DOMAIN (118 x 3 embedders) ===", flush=True)
    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    query_labels = expanded["query_labels"]
    query_ids = list(query_labels.keys())

    rankings, traj_texts = build_traj_rankings(query_ids, query_labels)

    cache_path = "trajectory_reranker_cache/step5_llm_reranker_cache_haiku_n118.jsonl"
    cache = {}
    if os.path.exists(cache_path):
        for line in open(cache_path):
            r = json.loads(line)
            cache[(r["embedder"], r["query_id"])] = r

    all_records = []
    write_lock = threading.Lock()
    for ename, qrankings in rankings.items():
        todo = [qid for qid in query_ids if (ename, qid) not in cache]
        for qid in query_ids:
            if (ename, qid) in cache:
                all_records.append(cache[(ename, qid)])
        done = [0]

        def process(qid):
            top10 = qrankings[qid]
            candidates_text = "\n".join(f"[{i+1}] {traj_texts[tid]}" for i, tid in enumerate(top10))
            prompt = TRAJ_PROMPT_TEMPLATE.format(anchor=query_labels[qid]["text"], candidates=candidates_text)
            j, cost = _guarded_call(prompt)
            choice_idx = parse_choice(j["text"])
            chosen_id = top10[choice_idx - 1] if choice_idx else None
            capped = j["output_tokens"] >= JUDGE_MAX_TOKENS
            record = {"embedder": ename, "query_id": qid, "raw_response": j["text"], "choice_idx": choice_idx,
                      "chosen_id": chosen_id, "top10": top10, "completion_tokens": j["output_tokens"],
                      "stop_reason": j["stop_reason"], "capped": capped}
            with write_lock:
                with open(cache_path, "a") as out_f:
                    out_f.write(json.dumps(record) + "\n")
                cache[(ename, qid)] = record
                all_records.append(record)
                done[0] += 1
                if done[0] % 30 == 0:
                    print(f"[task2-haiku] traj {ename}: {done[0]}/{len(todo)} -- {spend_line()}", flush=True)
            return record

        if todo:
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                list(ex.map(process, todo))
        print(f"[task2-haiku] traj {ename}: complete ({len(todo)} new)", flush=True)
    return all_records, rankings


def bootstrap_ratio_ci(before, after, n_boot=10000, seed=54321):
    before = np.array(before, dtype=bool)
    after = np.array(after, dtype=bool)
    n = len(before)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    return before[idx].mean(axis=1), after[idx].mean(axis=1)


def bootstrap_prop_ci(values, n_boot=10000, seed=98765):
    arr = np.array(values, dtype=float)
    n = len(arr)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(arr.mean()), float(lo), float(hi)


def main():
    math_records = run_math()
    traj_records, traj_rankings = run_trajectories()

    # --- truncation / finish_reason audit FIRST, per standing rule ---
    print("\n=== TRUNCATION / FINISH_REASON AUDIT ===")
    results_summary = {"math": {}, "trajectories": {}}
    for label, records in [("math", math_records), ("trajectories", traj_records)]:
        n_capped = sum(1 for r in records if r["capped"])
        n_unparsed = sum(1 for r in records if r["choice_idx"] is None)
        non_end_turn = sum(1 for r in records if r.get("stop_reason") != "end_turn")
        print(f"{label}: n={len(records)} capped={n_capped}({n_capped/len(records)*100:.2f}%) "
              f"unparsed={n_unparsed}({n_unparsed/len(records)*100:.2f}%) "
              f"non-end_turn={non_end_turn}({non_end_turn/len(records)*100:.2f}%)")
        if n_capped / len(records) > 0.10:
            print(f"*** {label}: truncation exceeds 10% -- STOPPING before reporting accuracy ***")
            return

    print(f"\n{spend_line()}")

    # --- MATH: share of gap closed, 4 configs, with bootstrap CIs ---
    print("\n=== MATH: share of recoverable gap closed (Haiku, terse), with CIs ===")
    baseline = json.load(open("results/task2b_bootstrap_cis.json"))["math_baseline"]
    math_by_config = {}
    for r in math_records:
        math_by_config.setdefault((r["provider"], r["tier"]), {})[r["query_id"]] = r
    key_map = {"gemini": "Gemini-embedding-001", "deepinfra": "Qwen3-Embedding-8B(DeepInfra)"}
    for provider, tier in [("gemini", "easy"), ("gemini", "hard"), ("deepinfra", "easy"), ("deepinfra", "hard")]:
        recs_by_qid = math_by_config[(provider, tier)]
        src_recs = {r["query_id"]: r for r in [json.loads(l) for l in open(f"llm_reranker_cache/full_{provider}_{tier}.jsonl")]}
        qids = list(recs_by_qid.keys())
        before = [src_recs[q]["top10_ids"][0] == f"{q}::eq::{tier}" for q in qids]
        after = [recs_by_qid[q]["chosen_id"] == f"{q}::eq::{tier}" for q in qids]
        orig_h1 = sum(before) / len(before)
        reranked_h1 = sum(after) / len(after)
        orig_h10 = baseline[f"{key_map[provider]}/{tier}"]["Hit@10"]["mean"]
        gap = orig_h10 - orig_h1
        share_closed = (reranked_h1 - orig_h1) / gap * 100 if gap != 0 else None
        before_boot, after_boot = bootstrap_ratio_ci(before, after)
        gap_boot = orig_h10 - before_boot
        share_boot = (after_boot - before_boot) / gap_boot * 100
        share_boot = share_boot[np.isfinite(share_boot)]
        lo, hi = np.percentile(share_boot, [2.5, 97.5]) if len(share_boot) else (None, None)
        results_summary["math"][f"{provider}/{tier}"] = {
            "n": len(qids), "orig_h1": orig_h1, "reranked_h1": reranked_h1,
            "share_closed_pct": share_closed, "ci_lo": float(lo) if lo is not None else None,
            "ci_hi": float(hi) if hi is not None else None,
        }
        print(f"{provider}-embed/{tier}: Hit@1 {orig_h1:.3f}->{reranked_h1:.3f} "
              f"share_closed={share_closed:.1f}% [{lo:.1f},{hi:.1f}]")

    # --- MATH: contamination split, hard tier, both candidate sets, with CI ---
    print("\n=== MATH: contamination (well_known n=57 vs rest n=443), hard tier, Haiku, with CI ===")
    WELL_KNOWN = {"imo", "usa", "apm"}
    results_summary["math"]["contamination"] = {}
    for provider in ["gemini", "deepinfra"]:
        recs_by_qid = math_by_config[(provider, "hard")]
        qids = list(recs_by_qid.keys())
        hits = {q: recs_by_qid[q]["chosen_id"] == f"{q}::eq::hard" for q in qids}
        wk_hits = [hits[q] for q in qids if q.split("_")[0] in WELL_KNOWN]
        rest_hits = [hits[q] for q in qids if q.split("_")[0] not in WELL_KNOWN]
        wk_mean, wk_lo, wk_hi = bootstrap_prop_ci(wk_hits)
        rest_mean, rest_lo, rest_hi = bootstrap_prop_ci(rest_hits)
        # gap CI via joint resample, matching task2c methodology
        wk_arr = np.array(wk_hits, dtype=bool)
        rest_arr = np.array(rest_hits, dtype=bool)
        rng = np.random.default_rng(98765)
        wk_idx = rng.integers(0, len(wk_arr), size=(10000, len(wk_arr)))
        rest_idx = rng.integers(0, len(rest_arr), size=(10000, len(rest_arr)))
        gaps = (wk_arr[wk_idx].mean(axis=1) - rest_arr[rest_idx].mean(axis=1)) * 100
        glo, ghi = np.percentile(gaps, [2.5, 97.5])
        gap_pts = wk_mean * 100 - rest_mean * 100
        results_summary["math"]["contamination"][provider] = {
            "well_known_rate": wk_mean, "rest_rate": rest_mean, "gap_pts": gap_pts,
            "ci_lo": float(glo), "ci_hi": float(ghi), "n_well_known": len(wk_hits), "n_rest": len(rest_hits),
        }
        print(f"{provider}-embed: well_known={wk_mean*100:.1f}%(n={len(wk_hits)}) rest={rest_mean*100:.1f}%(n={len(rest_hits)}) "
              f"gap={gap_pts:+.1f}pts [{glo:+.1f},{ghi:+.1f}]")

    # --- TRAJECTORIES: share of gap closed, 3 embedders, with CIs ---
    print("\n=== TRAJECTORIES: share of recoverable gap closed (Haiku), with CIs ===")
    full_traj = json.load(open("results/task1_expanded_full_results.json"))
    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    tiers = expanded["tiers"]
    query_labels = expanded["query_labels"]
    query_ids = list(query_labels.keys())
    traj_by_embedder = {}
    for r in traj_records:
        traj_by_embedder.setdefault(r["embedder"], {})[r["query_id"]] = r

    results_summary["trajectories"]["reranker"] = {}
    results_summary["trajectories"]["taxonomy"] = {}
    for ename in traj_rankings:
        recs = traj_by_embedder[ename]
        before, after = [], []
        sib, nm, other, unparsed = 0, 0, 0, 0
        for qid in query_ids:
            r = recs[qid]
            strict_set = set(tiers[qid]["STRICT"])
            b = traj_rankings[ename][qid][0] in strict_set
            a = (r["chosen_id"] in strict_set) if r["chosen_id"] else False
            before.append(b)
            after.append(a)
            if not a:
                if r["chosen_id"] is None:
                    unparsed += 1
                elif r["chosen_id"] in set(tiers[qid]["SIBLING"]):
                    sib += 1
                elif r["chosen_id"] in set(tiers[qid]["NEAR_MISS"]):
                    nm += 1
                else:
                    other += 1
        orig_h1 = full_traj["subsets"]["pooled"]["embedders"][ename]["STRICT"]["Hit@1"]
        orig_h10 = full_traj["subsets"]["pooled"]["embedders"][ename]["STRICT"]["Hit@10"]
        reranked_h1 = sum(after) / len(after)
        gap = orig_h10 - orig_h1
        share_closed = (reranked_h1 - orig_h1) / gap * 100 if gap != 0 else None
        before_boot, after_boot = bootstrap_ratio_ci(before, after)
        gap_boot = orig_h10 - before_boot
        share_boot = (after_boot - before_boot) / gap_boot * 100
        share_boot = share_boot[np.isfinite(share_boot)]
        lo, hi = np.percentile(share_boot, [2.5, 97.5]) if len(share_boot) else (None, None)
        n_miss = len(query_ids) - sum(after)
        results_summary["trajectories"]["reranker"][ename] = {
            "n": len(query_ids), "orig_h1": orig_h1, "reranked_h1": reranked_h1,
            "share_closed_pct": share_closed, "ci_lo": float(lo) if lo is not None else None,
            "ci_hi": float(hi) if hi is not None else None,
        }
        results_summary["trajectories"]["taxonomy"][ename] = {
            "n_miss": n_miss, "sibling_pct": sib / n_miss * 100 if n_miss else None,
            "near_miss_pct": nm / n_miss * 100 if n_miss else None,
            "other_pct": other / n_miss * 100 if n_miss else None,
            "unparsed_pct": unparsed / n_miss * 100 if n_miss else None,
        }
        print(f"{ename}: Hit@1 {orig_h1:.3f}->{reranked_h1:.3f} share_closed={share_closed:.1f}% [{lo:.1f},{hi:.1f}] "
              f"| taxonomy(n_miss={n_miss}): SIBLING={sib/n_miss*100:.1f}% NEAR_MISS={nm/n_miss*100:.1f}% "
              f"OTHER={other/n_miss*100:.1f}% unparsed={unparsed/n_miss*100:.1f}%")

    json.dump(results_summary, open("results/task2_haiku_reranker_full.json", "w"), indent=2)
    print(f"\n{spend_line()}")
    print("Saved: results/task2_haiku_reranker_full.json")


if __name__ == "__main__":
    main()
