"""Utility-curve robustness check, complete-answers-only subset (paper Section 6).

Promotes what was previously an ad-hoc computation to a script. Restricts the
210-query utility-curve run to the n=127 queries whose solver answer finished
within budget (finish_reason == "stop") in all three conditions (none, dumb,
gold), then reports per-condition grader accuracy and exact McNemar tests
(none vs. gold, none vs. dumb, both graders) on that subset. Zero API calls --
reads only the existing cached solver/grader responses.
"""
import json
from itertools import combinations

from scipy.stats import binomtest

CACHE_PATH = "utility_curve_cache/utility_curve_deepseek_cache.jsonl"


def load_records():
    return [json.loads(l) for l in open(CACHE_PATH)]


def mcnemar_exact(a_labels, b_labels):
    """a_labels/b_labels: parallel lists of bool "correct" for the same queries
    under two conditions. Returns (n_disc_a_only, n_disc_b_only, p_value)."""
    b = sum(1 for x, y in zip(a_labels, b_labels) if x and not y)
    c = sum(1 for x, y in zip(a_labels, b_labels) if not x and y)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue
    return b, c, p


def main():
    records = load_records()
    by_query = {}
    for r in records:
        by_query.setdefault(r["query_id"], {})[r["condition"]] = r

    complete_qids = sorted(
        qid
        for qid, conds in by_query.items()
        if len(conds) == 3
        and all(conds[c]["finish_reason"] == "stop" for c in ("none", "dumb", "gold"))
    )
    n = len(complete_qids)
    print(f"Complete-answers-only subset: n={n} of {len(by_query)}\n")

    for cond in ("none", "dumb", "gold"):
        a_correct = sum(1 for q in complete_qids if by_query[q][cond]["correct_a"])
        b_correct = sum(1 for q in complete_qids if by_query[q][cond]["correct_b"])
        print(
            f"{cond}: correct_a {a_correct}/{n} ({100*a_correct/n:.1f}%), "
            f"correct_b {b_correct}/{n} ({100*b_correct/n:.1f}%)"
        )

    print()
    for grader in ("correct_a", "correct_b"):
        for cond in ("gold", "dumb"):
            none_labels = [bool(by_query[q]["none"][grader]) for q in complete_qids]
            cond_labels = [bool(by_query[q][cond][grader]) for q in complete_qids]
            b, c, p = mcnemar_exact(none_labels, cond_labels)
            print(
                f"McNemar none vs {cond} ({grader}): "
                f"discordant none-only={b} {cond}-only={c} (total {b + c}), p={p:.3f}"
            )


if __name__ == "__main__":
    main()
