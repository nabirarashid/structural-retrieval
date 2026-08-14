"""Step 5: LLM reranker on each embedder's top-10, STRICT scoring.
Judges: gemini-3.1-flash-lite (real cost) and glm-5.2-fp8 (free), terse
prompt, temperature 0 -- mirrors the MathNet reranker configs.

Standing rule (this project's fifth silent-truncation-bug prevention):
truncation checked and reported BEFORE any accuracy number is reported.
"""
import json
import re
import sys
import time

import numpy as np
import requests

sys.path.insert(0, "src")
from llm_reranker import _HttpError, _read_env_var
from spend_tracker import record_call, spend_line

REPO = "/tmp/proced_mem_bench_check"
GRADER_A_MODEL = "gemini-3.1-flash-lite"
GRADER_A_KEY = _read_env_var("GEMINI_API_KEY")
GLM_MODEL = "glm-5.2-fp8"
GLM_BASE_URL = _read_env_var("MANTIS_LLM_BASE_URL")
JUDGE_MAX_TOKENS = 16

PROMPT_TEMPLATE = """You are given an agent task instruction ("ANCHOR") and 10 candidate task trajectories, labeled 1 through 10. Exactly one candidate follows the same underlying PROCEDURE as the anchor -- the same transformation type (e.g. simple placement, clean-then-place, heat-then-place, cool-then-place, examine-with-light, two-object placement) -- even though it may involve a completely different object.

IGNORE which specific object is involved when deciding: matching object names, matching receptacles, or similar surface phrasing do NOT mean same procedure. A candidate can involve the exact same object as the anchor and still follow a different procedure, and a candidate can involve a completely different object and still follow the exact same procedure.

Focus only on: what sequence of actions / transformation type would you need to perform for each. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Respond with ONLY the candidate number (1-10). No explanation, no other text."""


def get_embedding_text(traj: dict) -> str:
    text = f"Task: {traj['task_description']}\nSteps:\n"
    for pair in traj["state_action_pairs"]:
        text += f"{pair['step_id']}. State: {pair['state']} -> Action: {pair['action']}\n"
    return text.strip()


def call_gemini(prompt: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GRADER_A_MODEL}:generateContent"
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{url}?key={GRADER_A_KEY}",
                json={"contents": [{"parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": 0.0, "maxOutputTokens": JUDGE_MAX_TOKENS}},
                timeout=60,
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data["candidates"][0].get("content", {})
                parts = content.get("parts")
                text = parts[0]["text"] if parts else ""
                usage = data.get("usageMetadata", {})
                finish_reason = data["candidates"][0].get("finishReason", "")
                return {"text": text, "prompt_tokens": usage.get("promptTokenCount", 0),
                        "completion_tokens": usage.get("candidatesTokenCount", 0), "finish_reason": finish_reason}
            raise _HttpError(resp.status_code, resp.text)
        except (_HttpError, requests.exceptions.RequestException) as e:
            status = getattr(e, "status", None)
            if attempt == 5 or (status is not None and status not in (429, 500, 502, 503, 504)):
                raise
            time.sleep(min(30, 2 ** attempt))


def call_glm(prompt: str) -> dict:
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{GLM_BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={"model": GLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.0, "max_tokens": JUDGE_MAX_TOKENS},
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or msg.get("reasoning_content") or ""
            usage = data.get("usage", {})
            return {"text": text, "completion_tokens": usage.get("completion_tokens", 0),
                    "finish_reason": choice.get("finish_reason", "")}
        except requests.exceptions.RequestException:
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


def parse_choice(text: str) -> int:
    m = re.search(r"\b([1-9]|10)\b", text)
    return int(m.group(1)) if m else None


def main():
    d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
    trajectories = {t["task_instance_id"]: t for t in d["trajectories"]}
    traj_ids = list(trajectories.keys())
    qd = json.load(open(f"{REPO}/procedural_memory_benchmark/benchmark/data/query_bank.json"))
    queries = {q["query_id"]: q for q in qd["queries"]}
    query_ids = list(queries.keys())

    tier_data = json.load(open("results/step3_tier_labels.json"))
    tiers = tier_data["tiers"]

    # rebuild top-10 rankings per embedder (same as step0_step4 script)
    from pathlib import Path
    from vector_cache import VectorCache

    traj_texts = {tid: get_embedding_text(trajectories[tid]) for tid in traj_ids}
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
    query_text_list = [queries[qid]["query_text"] for qid in query_ids]
    mini_traj = model.encode(traj_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_q = model.encode(query_text_list, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    mini_traj = mini_traj / np.linalg.norm(mini_traj, axis=1, keepdims=True)
    mini_q = mini_q / np.linalg.norm(mini_q, axis=1, keepdims=True)
    sims = mini_q @ mini_traj.T
    rankings["MiniLM-L6-v2"] = {query_ids[qi]: [traj_ids[i] for i in np.argsort(-sims[qi])[:10]] for qi in range(len(query_ids))}

    cache_path = "step5_llm_reranker_cache.jsonl"
    cache = {}
    import os
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                cache[(r["embedder"], r["judge"], r["query_id"])] = r
    out_f = open(cache_path, "a")

    results = {}
    for ename, qrankings in rankings.items():
        for judge_name, call_fn, is_free in [("gemini-3.1-flash-lite", call_gemini, False), ("glm-5.2-fp8", call_glm, True)]:
            for qid in query_ids:
                key = (ename, judge_name, qid)
                if key in cache:
                    continue
                top10 = qrankings[qid]
                candidates_text = "\n".join(f"[{i+1}] {traj_texts[tid]}" for i, tid in enumerate(top10))
                prompt = PROMPT_TEMPLATE.format(anchor=queries[qid]["query_text"], candidates=candidates_text)

                r = call_fn(prompt)
                if not is_free:
                    record_call(GRADER_A_MODEL, r["prompt_tokens"], r["completion_tokens"], note=f"step5 {ename}/{qid}")

                choice_idx = parse_choice(r["text"])
                chosen_id = top10[choice_idx - 1] if choice_idx else None
                completion_tokens = r.get("completion_tokens", 0)
                capped = completion_tokens >= JUDGE_MAX_TOKENS

                record = {"embedder": ename, "judge": judge_name, "query_id": qid,
                          "raw_response": r["text"], "choice_idx": choice_idx, "chosen_id": chosen_id,
                          "completion_tokens": completion_tokens, "capped": capped, "top10": top10}
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                cache[key] = record
        print(f"[step5] done: {ename}", flush=True)
    out_f.close()

    # --- truncation check FIRST, per standing rule ---
    all_records = list(cache.values())
    n_capped = sum(1 for r in all_records if r["capped"])
    n_unparsed = sum(1 for r in all_records if r["choice_idx"] is None)
    print(f"\n=== TRUNCATION CHECK ({len(all_records)} calls total) ===")
    print(f"capped at {JUDGE_MAX_TOKENS}-token cap: {n_capped}/{len(all_records)} ({n_capped/len(all_records)*100:.1f}%)")
    print(f"unparsed (no digit found): {n_unparsed}/{len(all_records)} ({n_unparsed/len(all_records)*100:.1f}%)")
    if n_capped > 0:
        print("STOPPING before reporting accuracy -- truncation detected, must diagnose first.")
        return

    # --- accuracy: share of recoverable gap closed, per embedder per judge ---
    baseline = json.load(open("results/step3_baseline_results.json"))
    print("\n=== STEP 5: LLM RERANKER, STRICT scoring ===")
    summary = {}
    for ename in rankings:
        summary[ename] = {}
        orig_h1 = baseline["metrics"][ename]["STRICT"]["Hit@1"]
        orig_h10 = baseline["metrics"][ename]["STRICT"]["Hit@10"]
        gap = orig_h10 - orig_h1
        print(f"\n{ename} (orig Hit@1={orig_h1:.3f}, Hit@10={orig_h10:.3f}, gap={gap*100:.1f}pts):")
        for judge_name in ["gemini-3.1-flash-lite", "glm-5.2-fp8"]:
            hits = []
            for qid in query_ids:
                r = cache[(ename, judge_name, qid)]
                strict_set = set(tiers[qid]["STRICT"])
                hits.append(r["chosen_id"] in strict_set if r["chosen_id"] else False)
            reranked_h1 = sum(hits) / len(hits)
            share_closed = (reranked_h1 - orig_h1) / gap * 100 if gap != 0 else None
            summary[ename][judge_name] = {"reranked_hit1": reranked_h1, "share_of_gap_closed_pct": share_closed}
            sc_str = f"{share_closed:.1f}%" if share_closed is not None else "n/a"
            print(f"  {judge_name:22s} Hit@1={reranked_h1:.3f}  delta={reranked_h1-orig_h1:+.3f}  share_closed={sc_str}")

    json.dump(summary, open("results/step5_llm_reranker_results.json", "w"), indent=2)
    print(f"\n{spend_line()}")
    print("Saved: results/step5_llm_reranker_results.json, step5_llm_reranker_cache.jsonl")


if __name__ == "__main__":
    main()
