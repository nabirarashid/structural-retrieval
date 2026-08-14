"""§2b: for each strict miss, compare lexical similarity anchor<->gold vs
anchor<->top-ranked false positive, directly (not implied via the reranker)."""
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, "src")

from data import MathNetEasy
from dumb_reranker import dumb_score

RESULTS_DIR = Path("results")
CONFIGS = [
    ("gemini", "easy", "baseline_gemini.json"),
    ("gemini", "hard", "baseline_gemini_hard.json"),
    ("deepinfra", "easy", "baseline_deepinfra.json"),
    ("deepinfra", "hard", "baseline_deepinfra_hard.json"),
]

ds = MathNetEasy.load(tier="easy")  # corpus/queries only; gold ids come from the saved files

report = []
for provider, tier, fname in CONFIGS:
    data = json.load(open(RESULTS_DIR / fname))
    misses = data["failure_details"]

    gold_sims, fp_sims, deltas = [], [], []
    for m in misses:
        qtext = ds.queries[m["query_id"]]
        gold_text = ds.corpus[m["gold_id"]]
        fp_text = ds.corpus[m["top1_id"]]
        g = dumb_score(qtext, gold_text)
        f = dumb_score(qtext, fp_text)
        gold_sims.append(g)
        fp_sims.append(f)
        deltas.append(f - g)  # positive = false positive is MORE lexically similar than gold

    n = len(misses)
    fp_closer_count = sum(1 for d in deltas if d > 0)
    result = {
        "provider": provider,
        "tier": tier,
        "n_misses": n,
        "anchor_to_gold": {
            "mean": statistics.mean(gold_sims), "median": statistics.median(gold_sims),
            "stdev": statistics.stdev(gold_sims) if n > 1 else 0.0,
        },
        "anchor_to_false_positive": {
            "mean": statistics.mean(fp_sims), "median": statistics.median(fp_sims),
            "stdev": statistics.stdev(fp_sims) if n > 1 else 0.0,
        },
        "delta_fp_minus_gold": {
            "mean": statistics.mean(deltas), "median": statistics.median(deltas),
            "stdev": statistics.stdev(deltas) if n > 1 else 0.0,
        },
        "pct_misses_where_fp_more_lexically_similar_than_gold": fp_closer_count / n * 100,
    }
    report.append(result)
    print(f"{provider}/{tier}: n={n}  "
          f"anchor-gold sim={result['anchor_to_gold']['mean']:.3f}  "
          f"anchor-FP sim={result['anchor_to_false_positive']['mean']:.3f}  "
          f"FP more similar in {result['pct_misses_where_fp_more_lexically_similar_than_gold']:.1f}% of misses")

with open(RESULTS_DIR / "lexical_distance_check.json", "w") as f:
    json.dump(report, f, indent=2)
print("\nSaved to results/lexical_distance_check.json")
