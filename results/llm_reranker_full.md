# LLM reranker, full run: 500 queries × 4 configs

Same setup as the 50-query validation, same prompt, same pinned judge model
(`gemini-3.1-flash-lite`, temperature 0.0) — unchanged so this is a clean scale-up, not a new
condition. 0 unparsed responses across all 2,000 judge calls.

## Headline results

| Provider | Tier | orig Hit@1 | LLM-reranked Hit@1 | Recoverable gap (Hit@10 − Hit@1) | Share of gap closed |
|---|---|---|---|---|---|
| Gemini-embedding-001 | Easy | 12.2% | 29.8% | 85.4 pts | **20.6%** |
| Gemini-embedding-001 | Hard | 0.0% | 24.6% | 55.4 pts | **44.4%** |
| Qwen3-Embedding-8B (DeepInfra) | Easy | 8.6% | 39.4% | 86.6 pts | **35.6%** |
| Qwen3-Embedding-8B (DeepInfra) | Hard | 0.0% | 8.6% | 21.0 pts | **41.0%** |

All four configs show a positive, non-trivial share of the recoverable gap closed by an LLM judge
instructed to ignore surface form — consistent with the 50-query validation (26.8% / 44.4% / 44.7%
/ 45.5%), though the easy-tier numbers came down somewhat at full scale (Gemini 26.8%→20.6%,
DeepInfra 44.7%→35.6%) while hard-tier numbers held steady (44.4%→44.4%, 45.5%→41.0%). The
validation's small-N easy-tier estimates were noisier than the hard-tier ones, in hindsight — hard
tier had a larger recoverable gap to work with in both runs and its share-closed estimate proved
more stable going from n=50 to n=500.

## Contamination split — full counts

Bucketed by competition-ID prefix: well_known = {imo, usa, apm}, regional = {rus, blr, twn, mng,
hrv, bra}, other = everything else. At n=500 the buckets land at **well_known n=57, regional n=85,
other n=358** (57+85+358=500) — a large jump in resolution from the validation run's n=4/8/38.

### Gemini-embedding-001

| Bucket | n | orig Hit@1 | LLM-reranked Hit@1 |
|---|---|---|---|
| **Easy** — well_known | 57 | 7/57 (12.3%) | 16/57 (28.1%) |
| Easy — regional | 85 | 14/85 (16.5%) | 23/85 (27.1%) |
| Easy — other | 358 | 40/358 (11.2%) | 110/358 (30.7%) |
| **Hard** — well_known | 57 | 0/57 (0.0%) | 24/57 (42.1%) |
| Hard — regional | 85 | 0/85 (0.0%) | 19/85 (22.4%) |
| Hard — other | 358 | 0/358 (0.0%) | 80/358 (22.3%) |

### Qwen3-Embedding-8B (DeepInfra)

| Bucket | n | orig Hit@1 | LLM-reranked Hit@1 |
|---|---|---|---|
| **Easy** — well_known | 57 | 6/57 (10.5%) | 15/57 (26.3%) |
| Easy — regional | 85 | 12/85 (14.1%) | 31/85 (36.5%) |
| Easy — other | 358 | 25/358 (7.0%) | 151/358 (42.2%) |
| **Hard** — well_known | 57 | 0/57 (0.0%) | 9/57 (15.8%) |
| Hard — regional | 85 | 0/85 (0.0%) | 5/85 (5.9%) |
| Hard — other | 358 | 0/358 (0.0%) | 29/358 (8.1%) |

## Does the well_known > regional gap at hard tier survive at n=57?

**Yes for Gemini, marginal for DeepInfra.** Two-proportion z-test and Fisher's exact test,
well_known (n=57) vs regional (n=85), both at hard tier:

| Provider | well_known Hit@1 | regional Hit@1 | gap (pts) | z | two-prop-z p | Fisher exact p |
|---|---|---|---|---|---|---|
| Gemini | 42.1% (24/57) | 22.4% (19/85) | +19.8 | 2.51 | **0.012** | **0.016** |
| DeepInfra | 15.8% (9/57) | 5.9% (5/85) | +9.9 | 1.94 | 0.052 | 0.082 |

The validation run's n=4 well_known bucket at hard tier (75% and 50% for Gemini/DeepInfra
respectively) was too small to trust on its own — a single-digit bucket can swing by tens of
points from one query flipping. At full scale the signal holds up for Gemini (p=0.012–0.016,
comfortably below the conventional 0.05 threshold, on 142 combined observations) but is only
borderline for DeepInfra (p=0.05–0.08, i.e. right at or just past the usual significance
threshold depending on the test — not confidently distinguishable from noise on its own).

**Honest read:** one of the two providers shows a statistically credible well_known > regional gap
at hard tier; the other shows a gap of similar direction and comparable relative size but doesn't
clear significance at this sample size. This is consistent with — not strong independent
confirmation of — a recognition-vs-reasoning effect: the judge may be doing better on well-known
competitions partly because it recognizes them, not purely because it reasons about technique
better. But it's not a slam-dunk finding on this data alone; call it a directionally consistent,
partially-significant signal that would benefit from a larger regional bucket or a second judge
model to corroborate (see below).

Easy-tier buckets show no such consistent pattern (well_known and regional LLM-Hit@1 are within a
few points of each other for both providers, and "other" is sometimes the highest of the three) —
the contamination effect, to the extent it's real, is specific to the hard tier, where the
reformulation has stripped away more of the easy lexical signal and the judge has to lean more on
whatever else it knows about a problem.

## Update: pooled well_known vs (regional + other), n=443

At hard tier, regional (22.4% / 5.9%) and other (22.3% / 8.1%) are close enough for both providers
that pooling them into a single "not well-known" comparison group is justified and buys
substantially more power (n=443 vs n=85 for the pairwise regional-only comparison above).

| Provider | well_known Hit@1 (n=57) | pooled non-well-known Hit@1 (n=443) | gap (pts) | z | two-prop-z p | Fisher exact p |
|---|---|---|---|---|---|---|
| Gemini | 42.1% (24/57) | 22.4% (99/443) | +19.8 | 3.26 | **0.0011** | **0.0018** |
| DeepInfra | 15.8% (9/57) | 7.7% (34/443) | +8.1 | 2.06 | 0.0397 | 0.0733 |

Pooling changes the picture for Gemini from "significant" to "clearly significant" (p drops by
roughly an order of magnitude, since the comparison group triples in size while its rate barely
moves: 22.4%→22.4% for Gemini, essentially unchanged). For DeepInfra it's more mixed: the
normal-approximation z-test now crosses the conventional 0.05 threshold (p=0.040), but the exact
Fisher test — which doesn't rely on the normal approximation and is more trustworthy when one cell
count is small (9 and 34 are both fairly small counts relative to their totals) — still does not
(p=0.073). This z-vs-Fisher disagreement is itself informative: it means the DeepInfra result sits
right at the edge of what this sample size can resolve, and which side of "significant" it falls on
depends on which test you trust more, not on a clean signal in the data.

**Revised honest read:** with pooling, the well_known > regional/other gap at hard tier is now
solidly significant for Gemini (p<0.002 by either test) and remains borderline-to-not-quite for
DeepInfra depending on test choice. The direction and rough magnitude are consistent across both
providers and both grouping schemes, which is itself evidence the effect is real rather than a
one-provider artifact — but DeepInfra's own data alone still can't rule out chance at conventional
thresholds. A second judge model (see the pluggable-backend refactor) would be the cleanest way to
resolve whether DeepInfra's borderline result is a real, smaller effect or noise.

## Reproducibility

`scripts/run_llm_reranker_full.py`, seed=42, `gemini-3.1-flash-lite` @ temperature=0.0, identical
prompt to the validation run. Raw output: `results/llm_reranker_full.json`. Per-query judge
responses cached verbatim in `llm_reranker_cache/full_{provider}_{tier}.jsonl`.
