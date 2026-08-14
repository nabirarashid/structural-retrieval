# Direct lexical distance check (§2b): anchor↔gold vs anchor↔false-positive

The dumb reranker control implies this result indirectly (reranking by lexical similarity makes
things worse). This measures it directly: for every strict rank-1 miss, compare the anchor's
lexical similarity to the actual gold target against its lexical similarity to the false positive
that won rank 1. Same `dumb_score` (Jaccard token overlap + edit-distance similarity + length
ratio, equal-weighted, 0-1 scale) used throughout, zero math awareness.

## Results

| Provider | Tier | n (misses) | anchor↔gold (mean) | anchor↔FP (mean) | FP more similar in |
|---|---|---|---|---|---|
| Gemini-embedding-001 | Easy | 439 | 0.652 | 0.908 | **95.7%** of misses |
| Gemini-embedding-001 | Hard | 500 | 0.367 | 0.883 | **99.8%** of misses |
| Qwen3-Embedding-8B | Easy | 457 | 0.656 | 0.911 | **95.2%** of misses |
| Qwen3-Embedding-8B | Hard | 500 | 0.367 | 0.892 | **99.4%** of misses |

Full distributions (mean/median/stdev per side, plus the paired delta) saved in
`results/lexical_distance_check.json`.

## Interpretation

This is a direct, not inferred, confirmation of the pattern in the qualitative examples and the
dumb reranker control: when the embedding models miss at rank 1, the winning wrong answer is
almost never a coincidence of lexical similarity — it is *the* systematically more lexically
similar candidate in essentially every miss (95%+ at easy tier, 99%+ at hard tier, where the
larger gap between anchor↔gold and anchor↔FP similarity at hard tier reflects gold being more
heavily disguised there). The near-perfect consistency (not just "usually," but 95-99.8%) is
itself informative: this is not noise or an occasional confusable pair, it is the dominant,
structural failure mode of embedding-based retrieval on this benchmark.

## Reproducibility

Computed entirely from already-saved `failure_details` in the four baseline result files — no
new embeddings, no new API calls. Raw output: `results/lexical_distance_check.json`.
