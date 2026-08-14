# DeepInfra vs. the lab deployment: same model, identical text — do the embeddings match?

## Part 1: raw embedding similarity (500 identical texts, no retrieval involved)

Both providers claim to serve the exact same model: `Qwen/Qwen3-Embedding-8B`, confirmed
by querying each server's `/models` endpoint directly (not just trusting the provider name) —
both report identical model id and 4096-dim output. 500 corpus items sampled (seed=42, same
sampling convention as the rest of this project) and embedded by both providers on identical
text, with no prompt/task_type wrapping on either side.

| Metric | Value |
|---|---|
| Cosine similarity, mean | **0.9947** |
| Cosine similarity, median | 0.9950 |
| Cosine similarity, std | 0.00136 |
| Cosine similarity, min | 0.9884 |
| Cosine similarity, max | 0.9972 |
| Pairs below cosine 0.999 | 100% |
| Pairs below cosine 0.99 | 1.0% |

Both providers return embeddings that are already unit-norm server-side (norm stats
indistinguishable between them), which rules out pooling/normalization as the cause — the
remaining, small-but-systematic gap points to a numerical precision / serving-stack difference
(quantization, attention kernel, batching/padding, or vLLM version). Full detail on this part in
the earlier version of this file / `results/deepinfra_vs_labembed.json`.

**The open question from that result was: does a ~0.5% average cosine drift actually move
retrieval rankings at 117k-corpus scale, or is it noise that washes out?** That's what Part 2
answers.

## Part 2: full retrieval baseline — does it change Hit@k?

Ran the identical 500-query, full-117,088-corpus baseline (same seed, same queries, same qrels,
same eval code) through the lab deployment and compared directly against the existing DeepInfra
baseline. Same embeddings that back every other result in this project — this is not a toy check.

### Headline Hit@k (STRICT)

| Tier | k | DeepInfra | the lab deployment | Δ (pts) |
|---|---|---|---|---|
| Easy | Hit@1 | 8.6% | 8.6% | 0.0 |
| Easy | Hit@5 | 86.8% | 87.8% | +1.0 |
| Easy | Hit@10 | 95.2% | 95.4% | +0.2 |
| Hard | Hit@1 | 0.0% | 0.0% | 0.0 |
| Hard | Hit@5 | 2.8% | 2.6% | −0.2 |
| Hard | Hit@10 | **21.0%** | **17.8%** | **−3.2** |

At a glance these deltas look small and mixed-direction — exactly what you'd expect from noise.
But an unpaired comparison undersells this: **all 500 queries are identical between the two runs**
(same sample, same qrels, only the embedding provider differs), so the right test is paired
(McNemar's exact test on discordant query-level outcomes), not an independent-samples comparison.

### Paired comparison (McNemar's exact test on the same 500 queries)

| Tier | k | both hit | DeepInfra-only | Lab-only | neither | McNemar exact p |
|---|---|---|---|---|---|---|
| Easy | Hit@1 | — | 1 | 1 | — | 1.000 |
| Easy | Hit@5 | — | 5 | 10 | — | 0.302 |
| Easy | Hit@10 | — | 2 | 3 | — | 1.000 |
| Hard | Hit@1 | 0 | 0 | 0 | 500 | n/a (no hits either side) |
| Hard | Hit@5 | — | 2 | 1 | — | 1.000 |
| Hard | Hit@10 | 88 | **17** | **1** | 394 | **0.00014** |

Five of six Hit@k/tier combinations show no significant difference — the discordant pairs are
small and roughly balanced in both directions, consistent with the ~0.5% cosine drift being
retrieval-irrelevant noise almost everywhere. **One combination is a clear exception: hard-tier
Hit@10.** DeepInfra recovers the gold target in its top-10 for 17 queries that the lab deployment
misses, versus only 1 query going the other way — a lopsided, statistically significant asymmetry
(p=0.00014, exact binomial test on the 18 discordant pairs) that has nothing to do with sample
noise.

## Interpretation

This is a real, if narrow, finding: **the ~0.5% average embedding drift between the two
deployments is retrieval-irrelevant almost everywhere, but becomes consequential specifically at
the hard tier's Hit@10 margin.** That's the one place in this benchmark where rankings are being
decided by very fine distinctions — hard tier's reformulations have stripped away the easy lexical
signal, so gold and near-miss decoys sit close together in embedding space, and a small,
systematic numerical perturbation has enough room to flip which one lands inside vs. just outside
the top 10. Everywhere else (top-1, top-5, and easy tier generally) the signal is strong enough
that a 0.5%-level perturbation doesn't have room to change the outcome.

Practically: **DeepInfra and the lab deployment are safe to treat as equivalent for top-1/top-5
comparisons and for easy-tier work, but are not interchangeable for hard-tier Hit@10-sensitive
analysis.** Worth flagging to whoever operates the lab deployment — this is a
measurable, reproducible retrieval-level effect, not just a cosmetic embedding-space difference,
and if that deployment is meant to be a drop-in equivalent to the public DeepInfra endpoint, this
is the concrete evidence that it currently isn't at the margin.

## Reproducibility

Part 1: `scripts/compare_deepinfra_labembed.py`, seed=42, n=500. Raw output:
`results/deepinfra_vs_labembed.json`.

Part 2: `scripts/run_baseline.py --provider labembed` / `scripts/run_hard_tier.py --provider
labembed` for the baseline numbers (`results/baseline_labembed.json`,
`results/baseline_labembed_hard.json`); paired McNemar analysis computed directly from the
same cached embeddings used in every other DeepInfra/the lab deployment result in this project (no new
API calls for DeepInfra, which was already fully cached). Raw paired output:
`results/deepinfra_vs_labembed_paired.json`.
