# Dumb reranker control — a negative result, required before any smart reranker

**Why this control comes first:** the `::eq::`/`::nm::` variants in MathNet-Retrieve are LLM-
generated reformulations of an original problem (Gemini-3-flash, per the paper's Appendix F).
If those reformulations carry a detectable "generation fingerprint" — shared phrasing, template
structure, characteristic length — a reranker could score well by detecting *that*, not by
reasoning over mathematical structure. Any smart (LLM/semantic) reranker result is uninterpretable
until this confound is ruled in or out.

**Method:** rerank each query's existing top-10 (from the embedding baseline, unchanged) using a
score built entirely from lexical features with zero math or semantic awareness:
- word-level Jaccard token overlap
- normalized edit-distance similarity (`rapidfuzz.fuzz.ratio`, Levenshtein-based)
- text length ratio

Equal-weighted average of the three, no learned weights, no domain knowledge. Reranking only
reorders the existing top-10; it can never do better than the embedding's own Recall@10 (the
"recoverable gap" ceiling), and reranking accuracy is reported as **share of that gap closed**,
not raw Hit@1 delta, per the plan's own framing.

## Results

| Provider | Tier | Orig Hit@1 | Dumb-reranked Hit@1 | Recoverable gap (Hit@10 − Hit@1) | Share of gap closed |
|---|---|---|---|---|---|
| Gemini-embedding-001 | Easy | 12.2% | 4.4% | 85.4pp | **−9.1%** |
| Gemini-embedding-001 | Hard | 0.0% | 0.2% | 55.4pp | +0.4% |
| Qwen3-Embedding-8B | Easy | 8.6% | 4.4% | 86.6pp | **−4.8%** |
| Qwen3-Embedding-8B | Hard | 0.0% | 0.0% | 21.0pp | 0.0% |

## Interpretation

**The dumb reranker never helps, and actively hurts at easy tier for both providers.** This is
the opposite of what the generation-fingerprint risk predicted: if `::nm::` decoys were winning
because of a detectable lexical artifact and the embeddings were *already partially resistant* to
it, a purely lexical reranker exploiting that same artifact more aggressively should have shown
gains. Instead it makes retrieval measurably worse.

This is explained directly by the qualitative pattern already documented in
`strict_misses_examples.md`: the `::nm::` near-miss decoys are consistently **more lexically
similar to the query than the deliberately-disguised gold target is** (same variable names, same
language, same phrasing — gold intentionally changes all of these). A reranker built purely on
lexical similarity therefore reinforces exactly the failure mode already present in the
embeddings, rather than correcting it.

**Conclusion for interpreting future results — scoped precisely.** This control rules out one
specific mechanism: a **lexical** generation fingerprint (shared tokens, small edit distance,
similar length) is not what would make a reranker succeed here — pure lexical matching is actively
counterproductive, not a shortcut. That is a narrower claim than "generation artifacts in
general." An LLM reranker operates on signals token overlap and edit distance cannot see at all —
stylistic register, sentence structure, characteristic phrasing "moves," or other regularities
introduced by the generation process (Gemini-3-flash) that survive paraphrasing without producing
shared tokens or short edit distance. This control says nothing about whether an LLM judge could
pick up on regularities like that. So: a positive result from the LLM reranker below is **not**
guaranteed to reflect mathematical reasoning just because this lexical control came back negative
— it rules out one confound, not the general concern. The LLM reranker's own result needs to be
interpreted on its own evidence (e.g., the contamination-origin split), not inferred from this
control.

## Reproducibility

Same 500-query sample (seed=42), same full 117,088-item corpus, same cached embeddings as the
baseline run — no new API calls were made for this control (pure rescoring of already-embedded
text). Raw output: `results/dumb_reranker_control.json`.
