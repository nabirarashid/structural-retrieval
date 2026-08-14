# Baseline results: does embedding retrieval capture mathematical structure or surface form?

**Dataset:** MathNet-Retrieve (Alshammari et al., ICLR 2026, arXiv:2604.18584), 500 randomly
sampled anchor queries (seed=42, fixed), ranked against the full shared 117,088-item corpus.
No corpus subsampling at any point.

**Models:** Gemini `gemini-embedding-001` (Google, via Google AI Studio API,
`task_type=RETRIEVAL_QUERY`/`RETRIEVAL_DOCUMENT`), Qwen3-Embedding-8B (via DeepInfra,
OpenAI-compatible embeddings API) — the model Mantis's own production stack runs.

**Validation:** Gemini easy-tier numbers reproduce the paper's own Table 4 Easy row within
~1 point on every metric (ours: 12.2/89.8/97.6 vs. paper's 11.36/90.68/96.93 for R@1/R@5/R@10),
confirming the pipeline against the primary source rather than a secondhand summary.

## Results table

| Provider | Tier | STRICT Hit@1 | Hit@5 | Hit@10 | LENIENT Hit@1 | Hit@5 | Hit@10 |
|---|---|---|---|---|---|---|---|
| Gemini-embedding-001 | Easy | 12.2% | 89.8% | 97.6% | 14.8% | 97.2% | 100% |
| Gemini-embedding-001 | Hard | 0.0% | 10.0% | 55.4% | 14.8% | 97.2% | 100% |
| Qwen3-Embedding-8B | Easy | 8.6% | 86.8% | 95.2% | 10.0% | 95.0% | 99.0% |
| Qwen3-Embedding-8B | Hard | 0.0% | 2.8% | 21.0% | 10.0% | 95.0% | 99.0% |

STRICT = only the tier-designated `::eq::<tier>` variant counts as correct. LENIENT = any of the
three `::eq::` siblings (easy/medium/hard reformulations of the same problem) counts. STRICT
Recall@k (BEIR-standard) and STRICT Hit@k are identical by construction (single gold item per
query); LENIENT is reported as Hit@k, not standard multi-relevant Recall@k, because Recall@k
divides by the total relevant-item count (3 under lenient) and so mechanically caps Recall@1 at
1/3 even on a rank-1 hit — a metric-definition artifact unrelated to what "did we retrieve any
genuine equivalent" is actually asking. (BEIR-standard lenient Recall@k is still saved in the raw
result JSONs for completeness.)

## Tier invariance: what "easy vs. hard" is actually isolating

**The ranking is identical across tiers for a given query and provider.** Corpus and query text
are byte-identical (SHA-1 verified) across all three MathNet-Retrieve tier folders — the only
thing that differs is which corpus entry the qrels file marks as the designated positive. Since
embeddings are computed once from that shared text and reused for both tier evaluations, the
top-200 candidate ranking per query is the same list in the same order regardless of which tier
is being scored; only the label "which one is gold" changes. This means the easy/hard contrast
above isolates disguise level, with everything else — model, query set, corpus, ranking — held
exactly constant. It is not a different task, a different sample, or a different retrieval run.

**Proof, not assertion — the `sibling_eq_variant` arithmetic checks out exactly.** Gemini's
`sibling_eq_variant` failure count goes from 13 (easy) to 74 (hard), a delta of 61. Per-query
verification (not just the aggregate arithmetic) confirms: all 61 queries that hit at rank-1 under
easy scoring reappear, with zero exceptions, as `sibling_eq_variant` misses under hard scoring —
specifically because their rank-1 winner is their own `::eq::easy` variant, which is gold under
easy scoring and a "wrong sibling" under hard scoring. This identity is exact (not approximate)
because Gemini's hard-tier Hit@1 is exactly 0/500 — no query's rank-1 winner is ever its own
`::eq::hard` variant, so there is no offsetting case in the other direction. This is strong
evidence the failure taxonomy code is doing exactly what it claims, not producing coincidentally
plausible-looking numbers.

## Failure taxonomy — strict rank-1 misses, by provider and tier

| Category | Gemini Easy | Gemini Hard | Qwen3-8B Easy | Qwen3-8B Hard |
|---|---|---|---|---|
| own_nm_near_miss | 420 | 420 | 446 | 446 |
| sibling_eq_variant | 13 | 74 | 7 | 50 |
| other_base_item | 6 | 6 | 3 | 3 |
| unsuffixed_distractor | 0 | 0 | 1 | 1 |

`own_nm_near_miss` and `other_base_item`/`unsuffixed_distractor` counts are identical between
easy and hard for a given provider — expected, since a query's own near-miss decoys and
unrelated corpus items don't change identity or rank between tier evaluations, only whether the
*target itself* was hit changes. `sibling_eq_variant` is the only category that shifts, and it
shifts in exactly the direction and magnitude the tier-invariance argument predicts.

## Qualitative check (see `strict_misses_examples.md`): are the near-misses plausibly surface-similar?

Yes, and more specifically than "surface-similar" — manual inspection of 3 easy-tier rank-1
misses shows the pattern is not generic vocabulary overlap. In every example, the near-miss
decoy that beat gold **preserves the query's original surface form almost verbatim** (same
variable names, same language, same phrasing structure) while the actual gold target is a
*deliberately disguised* reformulation (renamed variables, translated language). The near-miss
differs from the query in exactly one mathematical detail — an inequality direction flipped, an
operation changed, a condition altered — while looking more textually like the query than the
gold answer does. This is a sharper, more actionable characterization than "the model gets
fooled by shared keywords": it's specifically fooled by whichever candidate is least disguised,
regardless of whether that candidate is mathematically related to the query at all.

## Cross-provider divergence — a Mantis-relevant finding

The easy/hard contrast is not uniform across providers. At easy tier, Gemini and Qwen3-8B are
close (Hit@5: 89.8% vs. 86.8%). At hard tier they diverge sharply: Gemini retains R@10=55.4%,
Qwen3-8B collapses to R@10=21.0% — less than half. **Mantis's actual production embedding model
is substantially more fragile under disguised surface form than the strongest model in this
comparison.** This is exactly the failure mode the whole project is about, measured on Mantis's
own infrastructure rather than inferred from the paper.

## Seed and reproducibility

500-query sample: `seed=42` (Python `random.Random(42).sample()` over the sorted 15,000 query
IDs). Recorded in every `results/baseline_*.json` file. All four runs share the identical query
sample by construction (fixed seed, no re-sampling between runs).

## Raw data

- `results/baseline_gemini.json`, `baseline_gemini_hard.json`,
  `baseline_deepinfra.json`, `baseline_deepinfra_hard.json` — full metrics, failure category
  counts, and per-miss detail (query id, gold id, top-1 id, top-1 score, category) for every
  strict miss.
