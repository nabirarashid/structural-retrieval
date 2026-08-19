"""Internal robustness check (pre-committed as paper-external): famous-vs-obscure
composition of the utility experiment. Uses the same well_known definition as
Task 2C (scripts/task2c_contamination_cis.py): query_id prefix in {imo, usa, apm}.
Zero API calls -- reads only utility_curve_cache/utility_curve_deepseek_cache.jsonl
and results/utility_curve_deepseek.json.
"""
import json

from scipy.stats import fisher_exact

WELL_KNOWN = {"imo", "usa", "apm"}
CACHE_PATH = "utility_curve_cache/utility_curve_deepseek_cache.jsonl"


def is_well_known(qid):
    return qid.split("_")[0] in WELL_KNOWN


def main():
    records = [json.loads(l) for l in open(CACHE_PATH)]
    by_query = {}
    for r in records:
        by_query.setdefault(r["query_id"], {})[r["condition"]] = r

    all_qids = sorted(by_query)
    n_total = len(all_qids)
    wk_all = [q for q in all_qids if is_well_known(q)]
    rest_all = [q for q in all_qids if not is_well_known(q)]
    print(f"1. All {n_total} utility queries: well_known={len(wk_all)}, rest={len(rest_all)}")

    complete_qids = [
        q for q in all_qids
        if len(by_query[q]) == 3
        and all(by_query[q][c]["finish_reason"] == "stop" for c in ("none", "dumb", "gold"))
    ]
    wk_complete = [q for q in complete_qids if is_well_known(q)]
    rest_complete = [q for q in complete_qids if not is_well_known(q)]
    print(f"2. Complete-in-all-three subset (n={len(complete_qids)}): "
          f"well_known={len(wk_complete)}, rest={len(rest_complete)}")

    zeroshot_fail_qids = [q for q in all_qids if not by_query[q]["none"]["correct_a"]]
    wk_fail = [q for q in zeroshot_fail_qids if is_well_known(q)]
    rest_fail = [q for q in zeroshot_fail_qids if not is_well_known(q)]
    print(f"3. Zero-shot failures under none/Grader A (n={len(zeroshot_fail_qids)}): "
          f"well_known={len(wk_fail)}, rest={len(rest_fail)}")

    print("\n4. Accuracy in none condition, by group:")
    for grader in ("correct_a", "correct_b"):
        wk_correct = sum(1 for q in wk_all if by_query[q]["none"][grader])
        rest_correct = sum(1 for q in rest_all if by_query[q]["none"][grader])
        print(f"   {grader}: well_known {wk_correct}/{len(wk_all)} "
              f"({100*wk_correct/len(wk_all):.1f}%), "
              f"rest {rest_correct}/{len(rest_all)} ({100*rest_correct/len(rest_all):.1f}%)")

    print("\n5. Truncation (capped) rate in none, by group:")
    wk_capped = sum(1 for q in wk_all if by_query[q]["none"]["capped"])
    rest_capped = sum(1 for q in rest_all if by_query[q]["none"]["capped"])
    print(f"   well_known {wk_capped}/{len(wk_all)} ({100*wk_capped/len(wk_all):.1f}%), "
          f"rest {rest_capped}/{len(rest_all)} ({100*rest_capped/len(rest_all):.1f}%)")

    print("\n6. Complete-127 none-condition accuracy per group:")
    for grader in ("correct_a", "correct_b"):
        wk_correct = sum(1 for q in wk_complete if by_query[q]["none"][grader])
        rest_correct = sum(1 for q in rest_complete if by_query[q]["none"][grader])
        wk_pct = 100 * wk_correct / len(wk_complete) if wk_complete else float("nan")
        rest_pct = 100 * rest_correct / len(rest_complete) if rest_complete else float("nan")
        print(f"   {grader}: well_known {wk_correct}/{len(wk_complete)} ({wk_pct:.1f}%), "
              f"rest {rest_correct}/{len(rest_complete)} ({rest_pct:.1f}%), "
              f"diff={abs(wk_pct - rest_pct):.1f}pt")
        if abs(wk_pct - rest_pct) > 15:
            table = [[wk_correct, len(wk_complete) - wk_correct],
                     [rest_correct, len(rest_complete) - rest_correct]]
            _, p = fisher_exact(table)
            print(f"      diff exceeds 15pt -> two-sided Fisher exact p={p:.4f}")
        else:
            print("      diff does not exceed 15pt -> no significance test computed")


if __name__ == "__main__":
    main()
