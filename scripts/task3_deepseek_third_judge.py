"""Task 3: DeepSeek (native API) as a third judge, terse prompt, temp=0.
Runs on FINAL frozen sets only: math (500 x 4 configs) and trajectories
(expanded n=118 x 3 embedders). Truncation/finish_reason check happens
BEFORE any accuracy is reported, per standing rule -- DeepSeek was terse as
a grader (utility curve); this verifies it stays terse as a reranker judge
too, since a different task can surface different verbosity behavior (as
GLM's terse-vs-CoT split already showed).
"""
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "src")
from llm_solver_deepseek import call_solver as deepseek_call  # generic chat call, reused for judging
from spend_tracker import record_call, spend_line

JUDGE_MAX_TOKENS = 16
MODEL = "deepseek-v4-flash"


def parse_choice(text: str) -> int:
    m = re.search(r"\b([1-9]|10)\b", text)
    return int(m.group(1)) if m else None


_spend_lock = threading.Lock()


def call_judge(prompt: str) -> dict:
    r = deepseek_call(prompt, max_tokens=JUDGE_MAX_TOKENS)
    with _spend_lock:
        cost = record_call(MODEL, r["prompt_tokens"], r["completion_tokens"], note="task3 deepseek judge")
    return {"text": r["text"], "prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
            "finish_reason": r["finish_reason"], "cost": cost}


WORKERS = 8


def run_math():
    from llm_reranker import PROMPT_TEMPLATE
    from data import MathNetEasy

    print("[task3] === MATH DOMAIN ===", flush=True)
    all_records = []
    for provider, tier in [("gemini", "easy"), ("gemini", "hard"), ("deepinfra", "easy"), ("deepinfra", "hard")]:
        ds = MathNetEasy.load(tier=tier)
        cache_path = f"llm_reranker_cache/full_deepseek_{provider}_{tier}.jsonl"
        import os
        cache = {}
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                for line in f:
                    r = json.loads(line)
                    cache[r["query_id"]] = r

        src_recs = [json.loads(l) for l in open(f"llm_reranker_cache/full_{provider}_{tier}.jsonl")]
        todo = [r for r in src_recs if r["query_id"] not in cache]
        for r in src_recs:
            if r["query_id"] in cache:
                all_records.append(cache[r["query_id"]])

        out_f = open(cache_path, "a")
        write_lock = threading.Lock()
        done_counter = [0]

        def process(r):
            qid = r["query_id"]
            top10 = r["top10_ids"]
            candidates_text = "\n".join(f"[{i+1}] {ds.corpus[cid]}" for i, cid in enumerate(top10))
            prompt = PROMPT_TEMPLATE.format(anchor=ds.queries[qid], candidates=candidates_text)
            j = call_judge(prompt)
            choice_idx = parse_choice(j["text"])
            chosen_id = top10[choice_idx - 1] if choice_idx else None
            capped = j["completion_tokens"] >= JUDGE_MAX_TOKENS
            record = {"query_id": qid, "raw_response": j["text"], "choice_idx": choice_idx,
                      "chosen_id": chosen_id, "top10_ids": top10, "completion_tokens": j["completion_tokens"],
                      "capped": capped, "provider": provider, "tier": tier}
            with write_lock:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                all_records.append(record)
                done_counter[0] += 1
                if done_counter[0] % 100 == 0:
                    print(f"[task3] math {provider}/{tier}: {done_counter[0]}/{len(todo)} new done -- {spend_line()}", flush=True)
            return record

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(process, todo))
        out_f.close()
        print(f"[task3] math {provider}/{tier}: complete ({len(todo)} new, {len(cache)+len(todo)} total)", flush=True)
    return all_records


def run_trajectories():
    import numpy as np
    from pathlib import Path
    from vector_cache import VectorCache

    print("\n[task3] === TRAJECTORY DOMAIN ===", flush=True)
    expanded = json.load(open("results/task1_expanded_tier_labels.json"))
    query_labels = expanded["query_labels"]
    query_ids = list(query_labels.keys())

    d = json.load(open("/tmp/proced_mem_bench_check/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajs = {t["task_instance_id"]: t for t in d["trajectories"]}
    traj_ids = list(trajs.keys())

    def get_embedding_text(traj):
        text = f"Task: {traj['task_description']}\nSteps:\n"
        for p in traj["state_action_pairs"]:
            text += f"{p['step_id']}. State: {p['state']} -> Action: {p['action']}\n"
        return text.strip()

    traj_texts = {tid: get_embedding_text(trajs[tid]) for tid in traj_ids}

    PROMPT_TEMPLATE = re.search(
        r'PROMPT_TEMPLATE = """(.*?)"""', open("scripts/step5_llm_reranker.py").read(), re.S
    ).group(1)

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
    traj_text_list = [traj_texts[tid] for tid in traj_ids]
    query_text_list = [query_labels[qid]["text"] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_traj = mini_traj / np.linalg.norm(mini_traj, axis=1, keepdims=True)
    mini_q = mini_q / np.linalg.norm(mini_q, axis=1, keepdims=True)
    sims = mini_q @ mini_traj.T
    rankings["MiniLM-L6-v2"] = {query_ids[qi]: [traj_ids[i] for i in np.argsort(-sims[qi])[:10]] for qi in range(len(query_ids))}

    cache_path = "step5_llm_reranker_cache_deepseek.jsonl"
    import os
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                cache[(r["embedder"], r["query_id"])] = r
    out_f = open(cache_path, "a")

    all_records = []
    write_lock = threading.Lock()
    for ename, qrankings in rankings.items():
        todo = [qid for qid in query_ids if (ename, qid) not in cache]
        for qid in query_ids:
            if (ename, qid) in cache:
                all_records.append(cache[(ename, qid)])
        done_counter = [0]

        def process(qid):
            top10 = qrankings[qid]
            candidates_text = "\n".join(f"[{i+1}] {traj_texts[tid]}" for i, tid in enumerate(top10))
            prompt = PROMPT_TEMPLATE.format(anchor=query_labels[qid]["text"], candidates=candidates_text)
            j = call_judge(prompt)
            choice_idx = parse_choice(j["text"])
            chosen_id = top10[choice_idx - 1] if choice_idx else None
            capped = j["completion_tokens"] >= JUDGE_MAX_TOKENS
            record = {"embedder": ename, "query_id": qid, "raw_response": j["text"], "choice_idx": choice_idx,
                      "chosen_id": chosen_id, "top10": top10, "completion_tokens": j["completion_tokens"], "capped": capped}
            with write_lock:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                all_records.append(record)
                done_counter[0] += 1
                if done_counter[0] % 30 == 0:
                    print(f"[task3] traj {ename}: {done_counter[0]}/{len(todo)} new done -- {spend_line()}", flush=True)
            return record

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(process, todo))
        print(f"[task3] traj {ename}: complete ({len(todo)} new)", flush=True)
    out_f.close()
    return all_records


if __name__ == "__main__":
    math_records = run_math()
    traj_records = run_trajectories()

    print("\n=== TRUNCATION / FINISH_REASON CHECK (standing rule, before any accuracy) ===")
    for label, records in [("math", math_records), ("trajectory", traj_records)]:
        n_capped = sum(1 for r in records if r["capped"])
        n_unparsed = sum(1 for r in records if r["choice_idx"] is None)
        comp_tokens = [r["completion_tokens"] for r in records]
        mean_comp = sum(comp_tokens) / len(comp_tokens)
        print(f"{label}: n={len(records)}  capped={n_capped}({n_capped/len(records)*100:.1f}%)  "
              f"unparsed={n_unparsed}({n_unparsed/len(records)*100:.1f}%)  mean_completion_tokens={mean_comp:.1f}")
        if n_capped / len(records) > 0.10:
            print(f"*** WARNING: {label} truncation rate exceeds 10% -- STOPPING before reporting accuracy ***")

    print(f"\n{spend_line()}")
