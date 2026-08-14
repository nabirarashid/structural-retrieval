"""Targeted repair for Step 5's 3 GLM calls that hit the 16-token cap
(0/120 for gemini-3.1-flash-lite, 3/120 for glm-5.2-fp8 -- judge-specific,
not the earlier systematic budget-truncation pattern). GLM occasionally
wraps its answer in unprompted JSON or repeats itself; 16 tokens cuts that
off before or right at the answer. Re-running just these 3 at a 64-token
cap rather than trusting the truncated text. The 2 separate 'unparsed'
cases (GLM answered '11', an out-of-range pick) are a genuine model error,
not a truncation artifact -- raising the cap wouldn't fix an out-of-range
answer, so those are left as-is and scored as misses.
"""
import json
import sys

sys.path.insert(0, "src")
from step5_llm_reranker import call_glm, parse_choice, JUDGE_MAX_TOKENS  # noqa
import step5_llm_reranker as s5

REPO = "/tmp/proced_mem_bench_check"
NEW_CAP = 64

TO_REPAIR = [
    ("labembed-Qwen3-8B", "glm-5.2-fp8", "easy_13"),
    ("gemini-embedding-001", "glm-5.2-fp8", "easy_13"),
    ("gemini-embedding-001", "glm-5.2-fp8", "medium_13"),
]


def call_glm_with_cap(prompt: str, max_tokens: int) -> dict:
    import requests
    for attempt in range(6):
        try:
            resp = requests.post(
                f"{s5.GLM_BASE_URL}/chat/completions",
                headers={"Authorization": "Bearer dummy", "Content-Type": "application/json"},
                json={"model": s5.GLM_MODEL, "messages": [{"role": "user", "content": prompt}],
                      "temperature": 0.0, "max_tokens": max_tokens},
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
            import time
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))


d = json.load(open(f"{REPO}/procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"))
trajectories = {t["task_instance_id"]: t for t in d["trajectories"]}
qd = json.load(open(f"{REPO}/procedural_memory_benchmark/benchmark/data/query_bank.json"))
queries = {q["query_id"]: q for q in qd["queries"]}

cache_path = "step5_llm_reranker_cache.jsonl"
records = [json.loads(l) for l in open(cache_path)]
by_key = {(r["embedder"], r["judge"], r["query_id"]): (i, r) for i, r in enumerate(records)}

for ename, judge, qid in TO_REPAIR:
    idx, old_record = by_key[(ename, judge, qid)]
    top10 = old_record["top10"]
    candidates_text = "\n".join(f"[{i+1}] {s5.get_embedding_text(trajectories[tid])}" for i, tid in enumerate(top10))
    prompt = s5.PROMPT_TEMPLATE.format(anchor=queries[qid]["query_text"], candidates=candidates_text)

    r = call_glm_with_cap(prompt, NEW_CAP)
    choice_idx = parse_choice(r["text"])
    chosen_id = top10[choice_idx - 1] if choice_idx else None
    capped = r["completion_tokens"] >= NEW_CAP

    print(f"{ename}/{judge}/{qid}: old_raw={old_record['raw_response']!r}")
    print(f"  new (cap={NEW_CAP}): completion_tokens={r['completion_tokens']} capped={capped} "
          f"choice={choice_idx} raw={r['text']!r}")

    records[idx] = {**old_record, "raw_response": r["text"], "choice_idx": choice_idx, "chosen_id": chosen_id,
                     "completion_tokens": r["completion_tokens"], "capped": capped, "repaired_at_cap": NEW_CAP}

with open(cache_path, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")
print(f"\nRewrote {cache_path} with {len(TO_REPAIR)} repaired records.")
