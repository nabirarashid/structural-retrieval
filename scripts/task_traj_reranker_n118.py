"""Final experiment: LLM reranker on the trajectory domain at the full
n=118 query set (40 original + 78 new), both judges, terse prompt, all
three embedders' top-10, STRICT scoring under the frozen tier definition
(i) (results/task1_expanded_tier_labels.json).

Reuses the exact prompt/judge-call code from scripts/step5_llm_reranker.py
(the original n=40 run) rather than duplicating it, and reuses that run's
240 cached judge calls for the 40 original queries directly -- confirmed
byte-identical query_ids, STRICT tiers, and top-10 rankings against the
new n=118 pipeline before trusting the reuse (see inline checks in the
session that produced this script). Only the 78 new queries x 3 embedders
x 2 judges = 468 calls are new spend.

Standing rule: truncation/finish_reason checked and reported BEFORE any
accuracy number is reported.
"""
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from spend_tracker import record_call, spend_line, PRICING
from step5_llm_reranker import (
    PROMPT_TEMPLATE, get_embedding_text, call_gemini, call_glm, parse_choice,
    GRADER_A_MODEL, JUDGE_MAX_TOKENS,
)

OLD_CACHE_PATH = "trajectory_reranker_cache/step5_llm_reranker_cache.jsonl"
NEW_CACHE_PATH = "trajectory_reranker_cache/step5_llm_reranker_cache_n118.jsonl"
REPO = "/tmp/proced_mem_bench_check"
WORKERS = 8


def build_rankings(query_ids, query_labels):
    d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajs = {t["task_instance_id"]: t for t in d["trajectories"]}
    traj_ids = list(trajs.keys())
    traj_texts = {tid: get_embedding_text(trajs[tid]) for tid in traj_ids}

    rankings = {}
    for ename, provider, dim in [("labembed-Qwen3-8B", "labembed", 4096),
                                  ("gemini-embedding-001", "gemini", 3072)]:
        from vector_cache import VectorCache
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


def main():
    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    query_labels = expanded["query_labels"]
    tiers = expanded["tiers"]
    query_ids = list(query_labels.keys())
    print(f"[traj-reranker-n118] n={len(query_ids)} queries", flush=True)

    rankings, traj_texts = build_rankings(query_ids, query_labels)

    # --- seed cache from the n=40 run, verify it, then extend ---
    cache = {}
    if os.path.exists(NEW_CACHE_PATH):
        with open(NEW_CACHE_PATH) as f:
            for line in f:
                r = json.loads(line)
                cache[(r["embedder"], r["judge"], r["query_id"])] = r
        print(f"[traj-reranker-n118] resuming from existing {NEW_CACHE_PATH}: {len(cache)} cached", flush=True)
    else:
        seed = [json.loads(l) for l in open(OLD_CACHE_PATH)]
        n_mismatch = 0
        with open(NEW_CACHE_PATH, "w") as out_f:
            for r in seed:
                key = (r["embedder"], r["judge"], r["query_id"])
                # sanity: seeded top10 must match the freshly rebuilt ranking for this query/embedder.
                # labembed/gemini are memmap-cached embeddings (byte-identical across runs);
                # MiniLM is recomputed fresh from the raw model each time and can shift very-close
                # cosine ties across sentence-transformers/torch minor versions -- discard and
                # re-query rather than trust a stale top10 the judge never actually saw.
                if rankings[r["embedder"]][r["query_id"]] != r["top10"]:
                    n_mismatch += 1
                    continue
                cache[key] = r
                out_f.write(json.dumps(r) + "\n")
        print(f"[traj-reranker-n118] seeded {NEW_CACHE_PATH} from {OLD_CACHE_PATH}: {len(cache)} reused, "
              f"{n_mismatch} discarded (top10 mismatch, will re-query)", flush=True)

    todo = []
    for ename in rankings:
        for judge_name in ["gemini-3.1-flash-lite", "glm-5.2-fp8"]:
            for qid in query_ids:
                if (ename, judge_name, qid) not in cache:
                    todo.append((ename, judge_name, qid))
    n_gemini_todo = sum(1 for e, j, q in todo if j == "gemini-3.1-flash-lite")
    n_glm_todo = sum(1 for e, j, q in todo if j == "glm-5.2-fp8")
    print(f"[traj-reranker-n118] todo: {len(todo)} calls ({n_gemini_todo} gemini paid, {n_glm_todo} glm free)", flush=True)

    if todo:
        # --- real pre-estimate: one live probe call, then extrapolate ---
        probe = next((e, j, q) for e, j, q in todo if j == "gemini-3.1-flash-lite")
        ename, judge_name, qid = probe
        top10 = rankings[ename][qid]
        candidates_text = "\n".join(f"[{i+1}] {traj_texts[tid]}" for i, tid in enumerate(top10))
        prompt = PROMPT_TEMPLATE.format(anchor=query_labels[qid]["text"], candidates=candidates_text)
        r = call_gemini(prompt)
        cost0 = record_call(GRADER_A_MODEL, r["prompt_tokens"], r["completion_tokens"], note=f"traj-n118 probe {ename}/{qid}")
        per_call_cost = r["prompt_tokens"] / 1e6 * PRICING[GRADER_A_MODEL]["in"] + r["completion_tokens"] / 1e6 * PRICING[GRADER_A_MODEL]["out"]
        est_total = per_call_cost * n_gemini_todo
        print(f"[traj-reranker-n118] pre-estimate: probe call cost=${cost0:.5f} "
              f"(prompt_tokens={r['prompt_tokens']}, completion_tokens={r['completion_tokens']}) "
              f"-> est. ${est_total:.3f} for remaining {n_gemini_todo-1} gemini calls "
              f"(GLM calls free, lab-hosted)", flush=True)

        choice_idx = parse_choice(r["text"])
        chosen_id = top10[choice_idx - 1] if choice_idx else None
        capped = r["completion_tokens"] >= JUDGE_MAX_TOKENS
        record = {"embedder": ename, "judge": judge_name, "query_id": qid, "raw_response": r["text"],
                  "choice_idx": choice_idx, "chosen_id": chosen_id, "completion_tokens": r["completion_tokens"],
                  "capped": capped, "top10": top10}
        cache[(ename, judge_name, qid)] = record
        todo.remove(probe)
        write_lock = threading.Lock()
        with open(NEW_CACHE_PATH, "a") as out_f:
            out_f.write(json.dumps(record) + "\n")

        done_counter = [1]

        def process(item):
            ename, judge_name, qid = item
            top10 = rankings[ename][qid]
            candidates_text = "\n".join(f"[{i+1}] {traj_texts[tid]}" for i, tid in enumerate(top10))
            prompt = PROMPT_TEMPLATE.format(anchor=query_labels[qid]["text"], candidates=candidates_text)
            call_fn = call_gemini if judge_name == "gemini-3.1-flash-lite" else call_glm
            r = call_fn(prompt)
            if judge_name == "gemini-3.1-flash-lite":
                record_call(GRADER_A_MODEL, r["prompt_tokens"], r["completion_tokens"], note=f"traj-n118 {ename}/{qid}")
            choice_idx = parse_choice(r["text"])
            chosen_id = top10[choice_idx - 1] if choice_idx else None
            capped = r["completion_tokens"] >= JUDGE_MAX_TOKENS
            record = {"embedder": ename, "judge": judge_name, "query_id": qid, "raw_response": r["text"],
                      "choice_idx": choice_idx, "chosen_id": chosen_id, "completion_tokens": r["completion_tokens"],
                      "capped": capped, "top10": top10}
            with write_lock:
                with open(NEW_CACHE_PATH, "a") as out_f:
                    out_f.write(json.dumps(record) + "\n")
                cache[(ename, judge_name, qid)] = record
                done_counter[0] += 1
                if done_counter[0] % 50 == 0:
                    print(f"[traj-reranker-n118] {done_counter[0]}/{len(todo)+1} done -- {spend_line()}", flush=True)
            return record

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(process, todo))
        print(f"[traj-reranker-n118] all {len(cache)} calls complete -- {spend_line()}", flush=True)

    # --- truncation / finish_reason check FIRST, per standing rule ---
    all_records = list(cache.values())
    n_capped = sum(1 for r in all_records if r["capped"])
    n_unparsed = sum(1 for r in all_records if r["choice_idx"] is None)
    print(f"\n=== TRUNCATION CHECK ({len(all_records)} calls total) ===")
    print(f"capped at {JUDGE_MAX_TOKENS}-token cap: {n_capped}/{len(all_records)} ({n_capped/len(all_records)*100:.1f}%)")
    print(f"unparsed (no digit found): {n_unparsed}/{len(all_records)} ({n_unparsed/len(all_records)*100:.1f}%)")
    n_problem = sum(1 for r in all_records if r["capped"] or r["choice_idx"] is None)
    by_judge = {}
    for r in all_records:
        if r["capped"] or r["choice_idx"] is None:
            by_judge[r["judge"]] = by_judge.get(r["judge"], 0) + 1
    print(f"combined (capped OR unparsed): {n_problem}/{len(all_records)} ({n_problem/len(all_records)*100:.1f}%), "
          f"by judge: {by_judge}")
    # Diagnosed (not silently patched): every capped/unparsed record inspected by hand. Two distinct
    # mechanisms, both isolated to glm-5.2-fp8 (zero Gemini issues) -- (1) GLM occasionally ignores
    # the terse "ONLY the number" instruction and starts an unsolicited JSON reasoning trace, which
    # then gets cut off mid-structure (7/8 capped cases); (2) GLM occasionally answers with a
    # digit outside the valid 1-10 range ("11", "12") or "None" -- a genuine model miscount, not a
    # parsing bug (parse_choice's \b([1-9]|10)\b correctly rejects these as unparsed). Both are
    # scored as misses (chosen_id=None), not excluded. Matches this project's established GLM-terse
    # profile (median ~1 completion token, occasional non-terse exceptions) -- not the catastrophic,
    # majority-truncated failure mode that got GLM-CoT excluded. Proceeding per the same <=10%
    # threshold used for the math-domain third-judge truncation gate (scripts/task3_deepseek_third_judge.py).
    if n_capped / len(all_records) > 0.10:
        print("STOPPING before reporting accuracy -- truncation rate exceeds 10%, needs further diagnosis.")
        return

    # --- accuracy + bootstrap CIs + failure taxonomy + provenance split ---
    full = json.load(open("results/task1_expanded_full_results.json"))
    old_ids = [q for q in query_ids if query_labels[q]["provenance"] == "original"]
    new_ids = [q for q in query_ids if query_labels[q]["provenance"] != "original"]

    def bootstrap_ratio_ci(before, after, n_boot=10000):
        before = np.array(before, dtype=bool)
        after = np.array(after, dtype=bool)
        n = len(before)
        rng = np.random.default_rng(54321)
        idx = rng.integers(0, n, size=(n_boot, n))
        return before[idx].mean(axis=1), after[idx].mean(axis=1)

    print("\n=== TRAJECTORY LLM RERANKER, n=118, STRICT scoring ===")
    summary = {}
    for ename in rankings:
        summary[ename] = {}
        for judge_name in ["gemini-3.1-flash-lite", "glm-5.2-fp8"]:
            for subset_name, subset_ids, orig_h1, orig_h10 in [
                ("pooled", query_ids, full["subsets"]["pooled"]["embedders"][ename]["STRICT"]["Hit@1"],
                 full["subsets"]["pooled"]["embedders"][ename]["STRICT"]["Hit@10"]),
                ("old_40", old_ids, full["subsets"]["old_40"]["embedders"][ename]["STRICT"]["Hit@1"],
                 full["subsets"]["old_40"]["embedders"][ename]["STRICT"]["Hit@10"]),
                ("new_78", new_ids, full["subsets"]["new_78"]["embedders"][ename]["STRICT"]["Hit@1"],
                 full["subsets"]["new_78"]["embedders"][ename]["STRICT"]["Hit@10"]),
            ]:
                before, after = [], []
                sib, nm, other, unparsed = 0, 0, 0, 0
                for qid in subset_ids:
                    r = cache[(ename, judge_name, qid)]
                    strict_set = set(tiers[qid]["STRICT"])
                    b = rankings[ename][qid][0] in strict_set
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
                orig_h1_actual = sum(before) / len(before)
                reranked_h1 = sum(after) / len(after)
                gap = orig_h10 - orig_h1
                share_closed = (reranked_h1 - orig_h1) / gap * 100 if gap != 0 else None

                ci_lo = ci_hi = None
                if subset_name == "pooled":
                    before_boot, after_boot = bootstrap_ratio_ci(before, after)
                    gap_boot = orig_h10 - before_boot
                    share_boot = (after_boot - before_boot) / gap_boot * 100
                    share_boot = share_boot[np.isfinite(share_boot)]
                    if len(share_boot):
                        ci_lo, ci_hi = (float(x) for x in np.percentile(share_boot, [2.5, 97.5]))

                n_miss = len(subset_ids) - sum(after)
                taxonomy = None
                if subset_name == "pooled":
                    taxonomy = {"n_miss": n_miss, "sibling_pct": sib / n_miss * 100 if n_miss else None,
                                "near_miss_pct": nm / n_miss * 100 if n_miss else None,
                                "other_pct": other / n_miss * 100 if n_miss else None,
                                "unparsed_pct": unparsed / n_miss * 100 if n_miss else None}

                key = f"{judge_name}/{subset_name}"
                summary[ename][key] = {
                    "n": len(subset_ids), "orig_h1_baseline": orig_h1, "orig_h1_actual_topcand": orig_h1_actual,
                    "reranked_h1": reranked_h1, "share_closed_pct": share_closed,
                    "ci_lo": ci_lo, "ci_hi": ci_hi, "taxonomy": taxonomy,
                }
                ci_str = f" [{ci_lo:.1f},{ci_hi:.1f}]" if ci_lo is not None else ""
                sc_str = f"{share_closed:.1f}%{ci_str}" if share_closed is not None else "n/a"
                print(f"{ename:24s} {judge_name:22s} {subset_name:7s} Hit@1={reranked_h1:.3f} "
                      f"(orig={orig_h1:.3f}) share_closed={sc_str}")

    json.dump(summary, open("results/task_traj_reranker_n118.json", "w"), indent=2)
    print(f"\n{spend_line()}")
    print(f"Saved: results/task_traj_reranker_n118.json, {NEW_CACHE_PATH}")


if __name__ == "__main__":
    main()
