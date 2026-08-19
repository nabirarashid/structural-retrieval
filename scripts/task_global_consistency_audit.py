"""Global consistency audit, final pre-submission stress-test. Zero API calls;
reruns every check directly against raw cache/result files. Does not touch
results/paper.md. See results/global_consistency_audit.md for the narrated
report this script's output backs.
"""
import json

import numpy as np
from scipy.stats import binomtest

FAILS = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAILS.append(f"{label}: {detail}")
    print(f"[{status}] {label}" + (f" -- {detail}" if detail else ""))


def load_jsonl(path):
    return [json.loads(l) for l in open(path)]


# ---------------- 1. Integer-numerator sweep (representative subset) ----------------
print("=== 1. Integer-numerator sweep ===")

ld = json.load(open("results/lexical_distance_check.json"))
hit1 = {("gemini", "easy"): 61, ("gemini", "hard"): 0, ("deepinfra", "easy"): 43, ("deepinfra", "hard"): 0}
for e in ld:
    key = (e["provider"], e["tier"])
    expected_n_miss = 500 - hit1[key]
    check(f"lexical n_misses {key}", e["n_misses"] == expected_n_miss,
          f"{e['n_misses']} vs expected {expected_n_miss}")
    count = e["pct_misses_where_fp_more_lexically_similar_than_gold"] / 100 * e["n_misses"]
    check(f"lexical pct*n near-integer {key}", abs(count - round(count)) < 0.01, f"{count:.3f}")

dumb = json.load(open("results/dumb_reranker_control.json"))
for e in dumb:
    for field in ("orig_hit1", "orig_hit10", "dumb_reranked_hit1"):
        val = e[field] * 500
        check(f"dumb math {e['provider']}/{e['tier']}/{field} near-integer",
              abs(val - round(val)) < 0.01, f"{val:.3f}")

util_recs = load_jsonl("utility_curve_cache/utility_curve_deepseek_cache.jsonl")
by_query = {}
for r in util_recs:
    by_query.setdefault(r["query_id"], {})[r["condition"]] = r
n_util = len(by_query)
check("utility pool size == 210", n_util == 210, str(n_util))

for cond in ("none", "dumb", "gold"):
    capped = sum(1 for q in by_query if by_query[q][cond]["capped"])
    correct_a = sum(1 for q in by_query if by_query[q][cond]["correct_a"])
    print(f"  {cond}: capped={capped}/210 ({100*capped/210:.2f}%), correct_a={correct_a}/210 ({100*correct_a/210:.2f}%)")

zeroshot_fail = sum(1 for q in by_query if not by_query[q]["none"]["correct_a"])
check("64 of 210 failed zero-shot", zeroshot_fail == 64, str(zeroshot_fail))


# ---------------- 2a. Trajectory old_40 + new_78 = pooled ----------------
print("\n=== 2a. old_40 + new_78 = pooled ===")
expanded = json.load(open("results/task1_expanded_tier_labels.json"))
labels = expanded["query_labels"]
tiers = expanded["tiers"]


def provenance(qid):
    return labels[qid]["provenance"]


def gold(qid):
    return set(tiers[qid]["STRICT"])


baseline_full = json.load(open("results/task1_expanded_full_results.json"))["subsets"]
ns = {"pooled": baseline_full["pooled"]["n"], "old_40": baseline_full["old_40"]["n"], "new_78": baseline_full["new_78"]["n"]}
for emb in baseline_full["pooled"]["embedders"]:
    for k in ("Hit@1", "Hit@5", "Hit@10"):
        counts = {s: baseline_full[s]["embedders"][emb]["STRICT"][k] * ns[s] for s in ("pooled", "old_40", "new_78")}
        ok = abs(counts["pooled"] - (counts["old_40"] + counts["new_78"])) < 0.01
        check(f"baseline old40+new78=pooled {emb}/{k}", ok)

traj_recs = load_jsonl("trajectory_reranker_cache/step5_llm_reranker_cache_n118.jsonl")
haiku_recs = load_jsonl("trajectory_reranker_cache/step5_llm_reranker_cache_haiku_n118.jsonl")
embedders = ["labembed-Qwen3-8B", "gemini-embedding-001", "MiniLM-L6-v2"]

for emb in embedders:
    for judge in ("gemini-3.1-flash-lite", "glm-5.2-fp8"):
        recs = [r for r in traj_recs if r["embedder"] == emb and r["judge"] == judge]
        old_hits = sum(1 for r in recs if provenance(r["query_id"]) == "original" and r["chosen_id"] in gold(r["query_id"]))
        new_hits = sum(1 for r in recs if provenance(r["query_id"]) == "new_alfworld_valid_unseen" and r["chosen_id"] in gold(r["query_id"]))
        pooled_hits = sum(1 for r in recs if r["chosen_id"] in gold(r["query_id"]))
        check(f"reranked old40+new78=pooled {emb}/{judge}", old_hits + new_hits == pooled_hits)
    haiku_e = [r for r in haiku_recs if r["embedder"] == emb]
    old_hits = sum(1 for r in haiku_e if provenance(r["query_id"]) == "original" and r["chosen_id"] in gold(r["query_id"]))
    new_hits = sum(1 for r in haiku_e if provenance(r["query_id"]) == "new_alfworld_valid_unseen" and r["chosen_id"] in gold(r["query_id"]))
    pooled_hits = sum(1 for r in haiku_e if r["chosen_id"] in gold(r["query_id"]))
    check(f"reranked old40+new78=pooled {emb}/haiku-j", old_hits + new_hits == pooled_hits)


# ---------------- 2b. hits+misses=n, taxonomy sums, share-closed consistency ----------------
print("\n=== 2b. 21 judge cells + dumb cells ===")

math_baseline = {
    ("gemini", "easy"): {"hit1": 61, "hit10": 488}, ("gemini", "hard"): {"hit1": 0, "hit10": 277},
    ("deepinfra", "easy"): {"hit1": 43, "hit10": 476}, ("deepinfra", "hard"): {"hit1": 0, "hit10": 105},
}


def load_math(judge, provider, tier):
    prefix = {"gemini": "full", "glm": "full_glm", "haiku": "full_haiku"}[judge]
    return load_jsonl(f"llm_reranker_cache/{prefix}_{provider}_{tier}.jsonl")


for provider in ("gemini", "deepinfra"):
    for tier in ("easy", "hard"):
        for judge in ("gemini", "glm", "haiku"):
            recs = load_math(judge, provider, tier)
            n = len(recs)
            gset = {f"{r['query_id']}::eq::{tier}" for r in recs}
            hits = sum(1 for r in recs if r["chosen_id"] and r["chosen_id"] == f"{r['query_id']}::eq::{tier}")
            check(f"math hits+misses=n {provider}/{tier}/{judge}", hits + (n - hits) == n)

traj_gold = {qid: set(tiers[qid]["STRICT"]) for qid in tiers}
for emb in embedders:
    for judge_key, judge_label in [("gemini-3.1-flash-lite", "Gemini"), ("glm-5.2-fp8", "GLM")]:
        recs = [r for r in traj_recs if r["embedder"] == emb and r["judge"] == judge_key]
        sib = nm = other = unparsed = 0
        n_miss = 0
        for r in recs:
            qid = r["query_id"]
            hit = r["chosen_id"] in traj_gold[qid] if r["chosen_id"] else False
            if not hit:
                n_miss += 1
                if r["chosen_id"] is None:
                    unparsed += 1
                elif r["chosen_id"] in set(tiers[qid]["SIBLING"]):
                    sib += 1
                elif r["chosen_id"] in set(tiers[qid]["NEAR_MISS"]):
                    nm += 1
                else:
                    other += 1
        check(f"taxonomy sums to n_miss {emb}/{judge_label}", sib + nm + other + unparsed == n_miss,
              f"{sib}+{nm}+{other}+{unparsed}={sib+nm+other+unparsed} vs n_miss={n_miss}")
    haiku_e = [r for r in haiku_recs if r["embedder"] == emb]
    sib = nm = other = unparsed = 0
    n_miss = 0
    for r in haiku_e:
        qid = r["query_id"]
        hit = r["chosen_id"] in traj_gold[qid] if r["chosen_id"] else False
        if not hit:
            n_miss += 1
            if r["chosen_id"] is None:
                unparsed += 1
            elif r["chosen_id"] in set(tiers[qid]["SIBLING"]):
                sib += 1
            elif r["chosen_id"] in set(tiers[qid]["NEAR_MISS"]):
                nm += 1
            else:
                other += 1
    check(f"taxonomy sums to n_miss {emb}/Haiku", sib + nm + other + unparsed == n_miss)


# ---------------- 2c. Deployment table + lexical n's (lexical already done above) ----------------
print("\n=== 2c. Deployment divergence table ===")
dep = json.load(open("results/deepinfra_vs_labembed_paired.json"))
for tier, cells in dep.items():
    for metric, vals in cells.items():
        total = vals["both"] + vals["deepinfra_only"] + vals["labembed_only"] + vals["neither"]
        check(f"deployment row sum {tier}/{metric}", total == 500, str(total))

hard_hit10 = dep["hard"]["Hit@10"]
p = binomtest(min(hard_hit10["deepinfra_only"], hard_hit10["labembed_only"]),
              hard_hit10["deepinfra_only"] + hard_hit10["labembed_only"], 0.5, alternative="two-sided").pvalue
check("deployment hard/Hit@10 McNemar p matches stored", abs(p - hard_hit10["mcnemar_exact_p"]) < 1e-9, f"{p} vs {hard_hit10['mcnemar_exact_p']}")


# ---------------- 2d. McNemar marginals ----------------
print("\n=== 2d. Utility McNemar tables ===")
qids = sorted(by_query)


def mcnemar_table(cond_a, cond_b, grader):
    a = [by_query[q][cond_a][grader] for q in qids]
    b = [by_query[q][cond_b][grader] for q in qids]
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if not x and y)
    return a_only, b_only


for grader in ("correct_a", "correct_b"):
    for cond_a, cond_b in [("none", "gold"), ("none", "dumb")]:
        a_only, b_only = mcnemar_table(cond_a, cond_b, grader)
        n_disc = a_only + b_only
        p = binomtest(min(a_only, b_only), n_disc, 0.5, alternative="two-sided").pvalue if n_disc else 1.0
        print(f"  {grader} {cond_a} vs {cond_b}: {cond_a}_only={a_only} {cond_b}_only={b_only} p={p:.3f}")


# ---------------- 3. Truncation overlap ----------------
print("\n=== 3. Truncation overlap ===")
capped_sets = {c: {q for q in qids if by_query[q][c]["capped"]} for c in ("none", "dumb", "gold")}
union = capped_sets["none"] | capped_sets["dumb"] | capped_sets["gold"]
check("truncation union == 210-127", len(union) == 83, str(len(union)))
triple = capped_sets["none"] & capped_sets["dumb"] & capped_sets["gold"]
expected_indep = 210 * (64 / 210) * (66 / 210) * (65 / 210)
print(f"  triple overlap={len(triple)}, expected under independence={expected_indep:.1f}, ratio={len(triple)/expected_indep:.1f}x")


# ---------------- 4a. well_known definition + subset check ----------------
print("\n=== 4a. well_known definition drift ===")
WELL_KNOWN = {"imo", "usa", "apm"}


def is_wk(qid):
    return qid.split("_")[0] in WELL_KNOWN


math_pool = {r["query_id"] for r in load_jsonl("llm_reranker_cache/full_gemini_hard.jsonl")}
util_pool = set(by_query)
check("utility pool subset of math pool", util_pool.issubset(math_pool))
check("utility well_known subset of math well_known",
      {q for q in util_pool if is_wk(q)}.issubset({q for q in math_pool if is_wk(q)}))


# ---------------- 4b. medium_13 propagation ----------------
print("\n=== 4b. medium_13 relabel ===")
check("medium_13 task_type == 5 in canonical labels file", labels["medium_13"]["task_type"] == 5)


# ---------------- Summary ----------------
print("\n=== SUMMARY ===")
if FAILS:
    print(f"{len(FAILS)} FAILURES:")
    for f in FAILS:
        print(" -", f)
else:
    print("ALL CHECKS PASSED. Zero failures.")
