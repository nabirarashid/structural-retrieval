"""Task 2: 10-call Haiku 4.5 pilot before any real run. Real math-domain
reranking prompts (terse template, same as the other two judges), temp 0.
Verifies plain terse output (stop_reason, no hidden reasoning) and reports
measured cost per call from real API usage. STOP if anything looks off --
do not proceed to the full run from this script.
"""
import json
import re
import sys

sys.path.insert(0, "src")
from llm_judge_haiku import call_judge, MODEL
from llm_reranker import PROMPT_TEMPLATE
from data import MathNetEasy
from spend_tracker import record_call, spend_line

JUDGE_MAX_TOKENS = 16


def parse_choice(text: str):
    m = re.search(r"\b([1-9]|10)\b", text)
    return int(m.group(1)) if m else None


def main():
    ds = MathNetEasy.load(tier="hard")
    recs = [json.loads(l) for l in open("llm_reranker_cache/full_gemini_hard.jsonl")][:10]

    print(f"=== HAIKU 4.5 PILOT: 10 real reranking calls, max_tokens={JUDGE_MAX_TOKENS}, temp=0 ===\n")
    total_cost = 0.0
    n_capped = 0
    n_unparsed = 0
    for i, r in enumerate(recs):
        qid = r["query_id"]
        top10 = r["top10_ids"]
        candidates_text = "\n".join(f"[{i+1}] {ds.corpus[cid]}" for i, cid in enumerate(top10))
        prompt = PROMPT_TEMPLATE.format(anchor=ds.queries[qid], candidates=candidates_text)
        j = call_judge(prompt, max_tokens=JUDGE_MAX_TOKENS)
        cost = record_call(MODEL, j["input_tokens"], j["output_tokens"], note=f"task2 haiku pilot {qid}")
        total_cost += cost
        choice = parse_choice(j["text"])
        capped = j["output_tokens"] >= JUDGE_MAX_TOKENS
        if capped:
            n_capped += 1
        if choice is None:
            n_unparsed += 1
        print(f"[{i+1}] qid={qid} in_tok={j['input_tokens']} out_tok={j['output_tokens']} "
              f"stop_reason={j['stop_reason']} capped={capped} parsed_choice={choice} "
              f"cost=${cost:.5f} raw={j['text']!r}")

    print(f"\n=== PILOT SUMMARY ===")
    print(f"capped: {n_capped}/10, unparsed: {n_unparsed}/10")
    print(f"total cost: ${total_cost:.5f}, mean cost/call: ${total_cost/10:.5f}")
    print(f"\n{spend_line()}")
    if n_capped > 0 or n_unparsed > 2:
        print("\n*** SOMETHING LOOKS OFF -- STOPPING, do not proceed to full run ***")
    else:
        print("\nPilot looks clean -- terse, no capping. Safe to proceed to pre-estimate + full run.")


if __name__ == "__main__":
    main()
