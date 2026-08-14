# Do embedding models retrieve mathematical structure, or surface form?

**Status:** retrieval + reranking results only. A separate RAG "does retrieval help a solver"
utility-curve experiment is in progress and deliberately excluded from this document — its
pipeline has an open, unresolved bug (systematic solver-output truncation) and its numbers are
not yet trustworthy. Everything below is retrieval- and reranking-level only, and is not affected
by that bug.

## 1. Question

Text-embedding retrieval is usually validated on tasks where surface form and semantic content
point the same direction (a query about "quadratic equations" retrieves a document containing the
words "quadratic equation"). Mathematical problems break that alignment on purpose: two problems
can be *the same problem* — same underlying technique, same solution — dressed in completely
different variable names, languages, numbers, and framing, while two *different* problems can
share heavy surface vocabulary (same named theorem, similar phrasing) while requiring unrelated
solution techniques.

The question this project asks: **when surface form and mathematical structure are pulled apart,
does embedding-based retrieval track structure, or does it track surface form?** And as a
follow-up: can an LLM reranker, explicitly instructed to ignore surface form, recover what
embedding-only retrieval misses — and if so, is that recovery actually reasoning about
mathematical technique, or something else (e.g. recognizing famous competition problems from
training data)?

## 2. Data and method

**Dataset:** MathNet-Retrieve (Alshammari et al., ICLR 2026, arXiv:2604.18584), a BEIR-format
retrieval benchmark built from competition mathematics problems. Each original problem has:
- Two to three **`::eq::<tier>`** variants (easy/medium/hard) — LLM-generated (Gemini-3-flash,
  per the paper's Appendix F) reformulations that preserve the exact underlying problem and
  solution technique while changing surface form. Tier controls how aggressively disguised the
  reformulation is (easy = light rewording, hard = renamed variables, translated language,
  restated framing).
- Zero or more **`::nm::`** ("near-miss") variants — problems that look similar on the surface but
  require a *different* technique. These are the deliberate distractors.

500 anchor queries sampled once (`seed=42`, fixed for every run in this project) from the shared
15,000-query pool, ranked against the full, shared 117,088-item corpus — no subsampling of the
corpus at any point, for any run.

**Task:** for a given anchor query, does the retriever surface its correct `::eq::<tier>`
reformulation at rank 1 (STRICT) or within rank k (STRICT Hit@k)? A LENIENT variant additionally
credits any of the three `::eq::` siblings as correct. STRICT Hit@k and BEIR-standard Recall@k are
identical by construction here (exactly one gold item per query under STRICT); LENIENT is reported
as Hit@k rather than Recall@k because Recall@k mechanically caps at 1/3 on a rank-1 hit under
multi-relevant scoring — a metric artifact, not a real ceiling on retrieval quality.

**Embedding models:** `gemini-embedding-001` (Google) and `Qwen3-Embedding-8B` (served via
DeepInfra — this is the model Mantis's own production stack runs). A third deployment of the same
Qwen3-Embedding-8B model, served from an internal lab deployment, was used for a
same-model cross-deployment check (§3.7).

**Provider choice — stated explicitly, not left implicit.** Every headline result in §3.1–3.6
(baseline Hit@k, dumb-reranker/lexical controls, LLM reranker gains, CoT effects, contamination)
uses the **DeepInfra-hosted** deployment of Qwen3-Embedding-8B, not the lab deployment,
even though both were run and are known to diverge measurably at hard-tier Hit@10 (§3.7,
p=0.00014). This was a deliberate choice, not an oversight: (1) DeepInfra is what Mantis's own
production stack actually runs, so it is the more representative choice, not less; (2)
re-running the full reranker + contamination pipeline on the lab deployment candidates to homogenize
providers would cost real money (~1,500 new judge calls, an estimated $1.4–$1.6 in new Gemini
spend) for a change that — per the paired analysis in §3.7 — only measurably affects 18 of 500
hard-tier queries; the other 482 queries' top-10 sets are effectively unchanged between
deployments, so a full re-run would very likely reproduce the same headline numbers within noise.
**Every table below that reports Qwen3-Embedding-8B results is DeepInfra-hosted unless labeled
otherwise; the lab deployment is reported only in §3.7, as its own finding.**

**Validation gate, passed before any downstream analysis:** Gemini easy-tier numbers reproduce the
paper's own Table 4 Easy row within ~1 point on every metric (ours: 12.2/89.8/97.6 vs. paper's
11.36/90.68/96.93 for Strict R@1/R@5/R@10) — the pipeline was checked against the primary source,
not a secondhand summary, before being trusted for anything else.

**Reranking:** after initial embedding retrieval, an LLM judge is shown the anchor and its top-10
embedding-retrieved candidates and asked which candidate needs "the same underlying mathematical
technique or method," with an explicit instruction to ignore shared variable names, language,
story framing, and numbers. Two prompt styles (terse — answer only; CoT — think step by step,
state a final answer) × two judge models (`gemini-3.1-flash-lite`, `glm-5.2-fp8`) × two embedding
providers × two tiers = 8 judge/prompt/config combinations, all run on the same fixed 500-query
sample, `temperature=0.0`.

## 3. Results

### 3.1 Embedding retrieval baseline: surface form dominates

| Provider | Tier | STRICT Hit@1 | Hit@5 | Hit@10 | LENIENT Hit@1 | Hit@5 | Hit@10 |
|---|---|---|---|---|---|---|---|
| Gemini-embedding-001 | Easy | 12.2% | 89.8% | 97.6% | 14.8% | 97.2% | 100% |
| Gemini-embedding-001 | Hard | **0.0%** | 10.0% | 55.4% | 14.8% | 97.2% | 100% |
| Qwen3-Embedding-8B (DeepInfra) | Easy | 8.6% | 86.8% | 95.2% | 10.0% | 95.0% | 99.0% |
| Qwen3-Embedding-8B (DeepInfra) | Hard | **0.0%** | 2.8% | 21.0% | 10.0% | 95.0% | 99.0% |

Zero out of 500 queries hit at rank 1 for either provider at hard tier, despite both providers
doing reasonably well at easy tier and despite the *correct answer being in the corpus the whole
time* (LENIENT Hit@10 = 100%/99% — the models can find it, they just don't rank it first when it's
disguised).

**Tier invariance — what this contrast actually isolates.** Corpus and query text are
byte-identical (SHA-1 verified) across all three tier folders; embeddings are computed once and
reused. The top-200 ranking per query and provider is the *same list in the same order* regardless
of which tier is being scored — only the label of which corpus entry counts as "gold" changes.
This was verified exactly, not just assumed: Gemini's `sibling_eq_variant` miss count goes from 13
(easy) to 74 (hard), and per-query checking confirms all 61 of the newly-appearing hard-tier misses
are queries whose rank-1 winner is their own `::eq::easy` variant — gold under easy scoring, a
"wrong sibling" under hard scoring. So easy vs. hard isolates *disguise level*, with model, query
set, corpus, and ranking held exactly constant — not a different task or a different sample.

### 3.2 Failure taxonomy

| Category | Gemini Easy | Gemini Hard | Qwen3-8B (DeepInfra) Easy | Qwen3-8B (DeepInfra) Hard |
|---|---|---|---|---|
| own_nm_near_miss | 420 | 420 | 446 | 446 |
| sibling_eq_variant | 13 | 74 | 7 | 50 |
| other_base_item | 6 | 6 | 3 | 3 |
| unsuffixed_distractor | 0 | 0 | 1 | 1 |

The dominant failure mode at both tiers, for both providers, is `own_nm_near_miss`: the model
prefers the query's own deliberately-planted near-miss distractor over its actual disguised
sibling. This is the failure mode the rest of the analysis is built around.

**Qualitative confirmation** (3 verbatim examples, `results/strict_misses_examples.md`): in every
case, the near-miss decoy that wins rank 1 preserves the query's original surface form almost
verbatim (same variable names, same language, same phrasing) while differing in exactly one
mathematical detail (an inequality flipped, an operation substituted, a condition altered) — while
gold, the actual same-technique problem, has been deliberately reworded. E.g. query
`sau_2012_bb6608` (`√(x−1)+⋯+√(x−n) < x`) is beaten at rank 1 by an identical-text near-miss with
the inequality flipped to `>` (cosine 0.860), while its true easy-tier sibling — same technique,
renamed variables — ranks 4th (cosine 0.821).

### 3.3 Ruling out a lexical fingerprint confound

Since the `::eq::`/`::nm::` variants are themselves LLM-generated, a smart reranker could in
principle "succeed" by detecting a generation artifact (shared phrasing, template structure)
rather than reasoning about mathematics. Two controls, run *before* any LLM reranker, to rule this
in or out:

**Dumb (purely lexical) reranker control.** Rerank the existing top-10 using only Jaccard token
overlap + edit-distance similarity + length ratio — zero math or semantic awareness.

| Provider | Tier | Orig Hit@1 | Dumb-reranked Hit@1 | Share of recoverable gap closed |
|---|---|---|---|---|
| Gemini-embedding-001 | Easy | 12.2% | 4.4% | **−9.1%** |
| Gemini-embedding-001 | Hard | 0.0% | 0.2% | +0.4% |
| Qwen3-Embedding-8B (DeepInfra) | Easy | 8.6% | 4.4% | **−4.8%** |
| Qwen3-Embedding-8B (DeepInfra) | Hard | 0.0% | 0.0% | 0.0% |

The dumb reranker never helps and actively *hurts* at easy tier for both providers — the opposite
of what a lexical-fingerprint confound would predict. This rules out one specific mechanism: a
purely lexical generation fingerprint is not what would make a reranker succeed here.

**Direct lexical distance check.** For every strict rank-1 miss, compare the anchor's lexical
similarity to true gold vs. to the false positive that won rank 1 (same `dumb_score`, 0–1 scale):

| Provider | Tier | n (misses) | anchor↔gold | anchor↔false-positive | FP more similar in |
|---|---|---|---|---|---|
| Gemini-embedding-001 | Easy | 439 | 0.652 | 0.908 | **95.7%** |
| Gemini-embedding-001 | Hard | 500 | 0.367 | 0.883 | **99.8%** |
| Qwen3-Embedding-8B (DeepInfra) | Easy | 457 | 0.656 | 0.911 | **95.2%** |
| Qwen3-Embedding-8B (DeepInfra) | Hard | 500 | 0.367 | 0.892 | **99.4%** |

Confirms the failure mode directly rather than by inference: when embeddings miss at rank 1, the
winning wrong answer is almost never a coincidence — it is the systematically more lexically
similar candidate in 95–99.8% of misses.

**Scope of what these two controls establish:** a *lexical* generation fingerprint is ruled out as
the mechanism a reranker could exploit — pure lexical matching is actively counterproductive here,
not a shortcut. This is narrower than "generation artifacts in general": an LLM reranker can use
signals (stylistic register, characteristic phrasing) that token overlap and edit distance can't
see. A positive LLM-reranker result below is therefore not automatically evidence of mathematical
reasoning just because these lexical controls came back negative — that claim needs its own
evidence (§3.6).

### 3.4 LLM reranker: terse prompt, both judges

Same top-10 candidates, reranked by an LLM instructed to ignore surface form and pick the
candidate needing the same technique. Reported as **share of the recoverable gap closed**
(Hit@10 − Hit@1 defines the ceiling; reranking can only reorder the existing top-10, never retrieve
outside it), not raw Hit@1 delta.

| Config | `gemini-3.1-flash-lite` | `glm-5.2-fp8` | ratio (GLM/Gemini) |
|---|---|---|---|
| Gemini-embed / Easy | 20.6% | 10.1% | 0.49x |
| Gemini-embed / Hard | 44.4% | 10.5% | 0.24x |
| DeepInfra-embed / Easy | 35.6% | 12.0% | 0.34x |
| DeepInfra-embed / Hard | 41.0% | 18.1% | 0.44x |

All 8 configs (2 judges × 4 provider/tier combos) are positive. **Direction reproduces cleanly
across two independently-sourced judge models** — real corroboration this isn't a
`gemini-3.1-flash-lite`-specific artifact. **Magnitude does not reproduce**: GLM's gains are
consistently 2–4x smaller. (A caveat on GLM specifically: it answered this task in ~2 completion
tokens, near-instantly, versus 288 reasoning tokens on an unrelated riddle sanity-check in the same
session — so this isn't a case of GLM running out of budget; it just isn't engaging visible
deliberation on the terse-prompt version of this task. Whether that reflects genuinely weaker
judgment or an unengaged reasoning path is not established by this run alone — see §3.5.)

### 3.5 CoT prompt effects, and a correction

Both judges were re-run with an explicit chain-of-thought prompt (think step by step, state a
final answer). This surfaced a serious data-quality problem in the GLM condition, caught only
after the numbers were first reported — **flagged here explicitly, since it changes what can be
cited from this section.**

> **Correction: 63.3% of the GLM CoT run's 2,000 responses were truncated, not concluded.** GLM's
> CoT responses were capped at 6,144 tokens; per-response token counts (checked via the serving
> endpoint's own tokenizer, after the fact) showed 63.3% hit that cap mid-deliberation. The
> parser still extracted *a* candidate number from nearly all of them (only 1 of 1,266 truncated
> responses failed to parse), which is why the run initially looked clean ("0 unparsed"). **A
> response that parses successfully is not the same as a response that reached a real
> conclusion** — one cached example ends mid-sentence ("...Since the problem says 'exactly one,'
> and both 4 and 5") with the parser grabbing "5" as if it were a decided answer. A follow-up test
> with an explicit concise-reasoning instruction and a 16,384-token cap got truncation down to
> 10.0% (3/30) on a small sample — a real improvement, but a full rerun would cost ~8.7 hours of
> wall time for what is a secondary question, so **GLM CoT was not rerun and is excluded from this
> writeup's numbers entirely.** Gemini's CoT run is unaffected (max 978/4,096 tokens observed,
> never truncated), as is every terse-prompt result above (GLM terse median completion length was
> 1 token — genuinely terse, not truncated-long).

With that exclusion, the citable CoT comparison is Gemini terse vs. Gemini CoT only:

| Config | Gemini, terse | Gemini, CoT |
|---|---|---|
| Gemini-embed / Easy | 20.6% | **44.7%** |
| Gemini-embed / Hard | **44.4%** | 22.7% |
| DeepInfra-embed / Easy | 35.6% | **55.4%** |
| DeepInfra-embed / Hard | 41.0% | 26.7% |

CoT roughly *doubles* Gemini's easy-tier gains but *cuts hard-tier gains by a third to a half*.
Reasoning traces read as high-quality (correct, detailed derivations), but on hard tier the model
sometimes explicitly favors a candidate it itself calls "a direct restatement" of the anchor — a
plausible (unconfirmed — would need per-query auditing) mechanism is that giving the model room to
reason lets it drift toward the most recognizable/lexically-similar sibling rather than the
harder-to-recognize hard-tier variant, i.e. CoT may reintroduce exactly the surface-form bias the
prompt is trying to suppress. **Report the reranking-gains direction as solid across every judge
and prompt tested (8/8 combinations positive, including the compromised GLM-CoT data on direction
alone); never report a specific magnitude as "the" effect size without naming the exact judge and
prompt it came from — magnitude swings 2–4x by judge and up to 2x by prompt within the same
judge.**

### 3.6 Contamination: is the reranker recognizing famous competitions, not reasoning?

If an LLM judge does better on well-known competitions (IMO, USAMO, APMO) than obscure regional
ones, that's consistent with the judge partly recognizing the problem from training data rather
than purely reasoning about technique — a distinct explanation from "the reranker understands
mathematical structure." Bucketed by competition-ID prefix (well_known: imo/usa/apm, n=57;
everything else pooled, n=443), hard tier, all judge/prompt combinations where the underlying data
is trustworthy:

| Candidate set | Judge | Prompt | gap (pts) | two-prop-z p | Fisher exact p |
|---|---|---|---|---|---|
| Gemini-embed | gemini-3.1-flash-lite | terse | +19.8 | **0.0011** | **0.0018** |
| Gemini-embed | gemini-3.1-flash-lite | cot | +13.5 | **0.0038** | **0.0091** |
| Gemini-embed | glm-5.2-fp8 | terse | +1.4 | 0.676 | 0.560 |
| DeepInfra-embed | gemini-3.1-flash-lite | terse | +8.1 | 0.040 | 0.073 |
| DeepInfra-embed | gemini-3.1-flash-lite | cot | +9.5 | **0.0033** | **0.0087** |
| DeepInfra-embed | glm-5.2-fp8 | terse | +5.6 | 0.037 | 0.054 |

(GLM-CoT rows omitted per the correction in §3.5 — they are truncation artifacts, not a null
result, and are not folded into this verdict either direction.)

**Final framing:** contamination is found **robustly under the Gemini judge across both prompts**
(4/4 tested combinations significant, p 0.001–0.009). Restricted to the only uncompromised GLM
data — the terse-prompt rows — **GLM shows the same direction in both candidate sets**, reaching
borderline significance on the DeepInfra-embedding candidate set (+5.6pt, p=0.037–0.054) but not on
the Gemini-embedding candidate set (+1.4pt, p=0.56–0.68). This is a genuinely mixed result: the
effect is not purely a one-judge artifact (it shows up under both judges on one of two candidate
sets, at plausibly-different magnitude given the two judges have different training-data coverage
of competitions), but it is also not a uniformly-replicated universal effect. **Report as:
"contamination found robustly under Gemini across both prompts; GLM terse shows the same direction
at borderline significance; GLM CoT excluded due to 63.3% truncation."** Do not report "absent
under GLM" — that would conflate the broken CoT measurement with a genuine negative result, and
the terse-only GLM data that *is* trustworthy shows the same direction.

**What this does and doesn't establish (see also §5):** this is evidence *consistent with*
partial competition-recognition contributing to the reranker's gains, particularly under the
Gemini judge — it is not proof that the reranker is "just" pattern-matching famous problems rather
than reasoning about technique. The qualitative CoT traces in the earlier diagnostic (§3.5) show
GLM producing genuine, correct, technique-specific reasoning on individual queries (e.g. correctly
identifying an invariant-mod-9 argument, or setting up a calculus optimization) — so both
mechanisms (real reasoning, and recognition-assisted shortcuts on famous problems) plausibly
coexist, in proportions this data cannot cleanly separate.

### 3.7 Deployment divergence: is a same-model swap actually safe?

A practical question for Mantis: `Qwen3-Embedding-8B` is served both via DeepInfra (used
throughout this project) and via a second, internal lab deployment. Same model ID,
confirmed via each server's own `/models` endpoint.

**Raw embedding similarity** (500 identical texts, no retrieval involved): mean cosine similarity
0.9947, median 0.9950, both providers already unit-norm server-side — a small but systematic
numerical drift (quantization, kernel, or batching difference), not a normalization bug.

**Does that ~0.5% drift move actual retrieval rankings?** Full 500-query, 117,088-corpus baseline
run through both providers, compared with a *paired* test (McNemar's exact test on the same 500
queries — an unpaired comparison would understate this, since all 500 queries are identical
between runs):

| Tier | k | both hit | DeepInfra-only | Lab-only | McNemar exact p |
|---|---|---|---|---|---|
| Easy | Hit@1 | — | 1 | 1 | 1.000 |
| Easy | Hit@5 | — | 5 | 10 | 0.302 |
| Easy | Hit@10 | — | 2 | 3 | 1.000 |
| Hard | Hit@5 | — | 2 | 1 | 1.000 |
| Hard | Hit@10 | 88 | **17** | **1** | **0.00014** |

Five of six combinations show no significant difference — the ~0.5% drift is retrieval-irrelevant
noise almost everywhere. **One exception: hard-tier Hit@10**, where DeepInfra recovers gold in the
top-10 for 17 queries the lab deployment misses, versus only 1 query going the other way — a lopsided,
statistically significant asymmetry unrelated to sample noise. This is the one place in the
benchmark where rankings are decided by very fine margins (hard tier has stripped away the easy
lexical signal, so gold and near-miss decoys sit close together in embedding space), and a small
systematic perturbation has enough room to flip which one lands inside vs. outside the top 10.

**Practical takeaway:** the two deployments are safe to treat as equivalent for top-1/top-5
comparisons and easy-tier work, but are measurably, reproducibly *not* interchangeable for
hard-tier Hit@10-sensitive analysis — worth knowing before treating the lab deployment as a drop-in
substitute for the public DeepInfra endpoint in anything hard-tier-adjacent.

## 4. Synthesis: what's actually going on

Put together, sections 3.1–3.7 support a specific, narrow story, not a generic "embeddings are
bad at math" claim:

1. Embedding retrieval tracks **surface form**, not mathematical structure, when the two are
   deliberately pulled apart — 0% rank-1 accuracy at hard tier for both providers, despite the
   correct answer being retrievable in the top 10 nearly always.
2. This isn't a lexical-fingerprint artifact of how the dataset's disguised variants were
   generated — a purely lexical reranker makes things *worse*, not better (§3.3).
3. An LLM reranker explicitly told to ignore surface form recovers real signal — 20–44% of the
   recoverable gap at hard tier depending on judge (§3.4) — and this direction replicates across
   two independent judge models, which is the most solid cross-judge finding in this project.
4. Part, but likely not all, of that recovery is attributable to the judge recognizing famous
   competition problems rather than purely reasoning about technique — robust under one judge,
   borderline-replicated under a second, on one of two candidate sets (§3.6). Qualitative CoT
   traces show genuine technique-level reasoning is also happening on individual queries, so this
   is a "both, in some proportion" finding, not an either/or one.
5. Model choice and prompt choice both swing the *magnitude* of every effect above by 2–4x, while
   direction stays stable — a caution against citing any single number as "the" effect size for
   this task without naming the exact judge and prompt.

## 5. Limitations

- **500-query sample, one fixed seed.** All results derive from one `seed=42` sample of 500
  anchors from a 15,000-query pool. No cross-seed variance estimate exists for any headline number
  in this document.
- **Two judge models, not a representative sample of LLMs.** `gemini-3.1-flash-lite` and
  `glm-5.2-fp8` disagree on magnitude by 2–4x; a third or fourth judge could plausibly land
  anywhere in or outside that range.
- **The GLM CoT condition is missing, not just caveated** — it was never successfully measured at
  scale (§3.5). Any claim about GLM's CoT-prompt behavior in this document is bounded to a small
  (n≤50) pre-truncation-discovery sample and should not be treated as equivalent in reliability to
  the terse-prompt numbers.
- **Contamination attribution is correlational, not causal.** The well_known-vs-rest gap is
  consistent with competition recognition but this dataset and design cannot cleanly separate
  "recognizes the problem" from "the well-known-competition subset happens to also be easier to
  reason about for some other reason" (e.g. cleaner problem statements, more standard techniques).
- **Reranking is bounded by initial retrieval.** Every reranker number here can only reorder the
  embedding model's own top-10 — it cannot recover a correct answer the embedding step never
  retrieved into that top-10 in the first place. Share-of-gap-closed is the right lens for this
  reason, but it also means these results say nothing about retrieval quality beyond top-10.
- **`::nm::` near-miss decoys are also LLM-generated** — by the same model (Gemini-3-flash) that
  generated the `::eq::` positives, per the paper. The dumb-reranker and lexical-distance controls
  rule out a *lexical* generation fingerprint but cannot rule out subtler generation-process
  regularities in either direction.

## 6. What this document does not show

- **Not** that embedding retrieval is unusable for math retrieval in general — easy-tier numbers
  are strong for both providers (86–90% Hit@5); the failure is specific to *deliberately disguised*
  reformulations.
- **Not** a full statement on whether LLM reranking "solves" this problem — even the best config
  (Gemini CoT, DeepInfra-embed, easy tier) closes 55.4% of the recoverable gap, leaving the
  majority of the gap unclosed, and hard tier fares worse.
- **Not** evidence about whether retrieval quality at this level actually helps or hurts a
  downstream solver's accuracy on these problems — that is a separate, ongoing experiment
  (deliberately excluded here, per §0) with an unresolved data-quality bug of its own.
- **Not** a claim that GLM 5.2 is a weaker *reasoner* than Gemini on this task — the CoT diagnostic
  (§3.5, small-n, pre-truncation-discovery) showed GLM producing genuine correct reasoning when
  prompted for it; what's established is that GLM's *default terse-prompt* behavior under-engages
  relative to Gemini's, and that GLM's CoT behavior specifically could not be measured reliably at
  scale with this project's token budget.
- **Not** a judge-independent contamination finding — §3.6 is explicit that the strongest
  contamination result (Gemini-embed candidates) does not clearly survive switching judges.

## 7. Reproducibility

All raw results and per-query cached responses referenced above live under `results/` and
`llm_reranker_cache/` in this repository. Fixed `seed=42` throughout; every script referenced below
takes no undocumented parameters.

| Section | Script | Raw output |
|---|---|---|
| §3.1–3.2 | `scripts/run_baseline.py`, `run_hard_tier.py` | `baseline_{gemini,deepinfra}[_hard].json`, `.md` |
| §3.3 | dumb reranker + lexical distance scripts | `dumb_reranker_control.json`/`.md`, `lexical_distance_check.json`/`.md` |
| §3.4 | `scripts/run_llm_reranker_full.py`, `run_llm_reranker_full_glm.py` | `llm_reranker_full.json`/`.md`, `llm_reranker_full_glm.json`, `llm_reranker_glm_judge.md` |
| §3.5–3.6 | `scripts/run_llm_reranker_full_cot.py --judge {gemini,glm}` | `llm_reranker_full_cot_{gemini,glm}.json`, `llm_reranker_cot_full_comparison.md`, `glm_cot_diagnostic.md` |
| §3.7 | `scripts/compare_deepinfra_labembed.py`, `run_baseline.py --provider labembed` | `deepinfra_vs_labembed.json`/`.md`, `deepinfra_vs_labembed_paired.json` |

Per-query judge responses are cached verbatim in `llm_reranker_cache/{run}_{judge}_{provider}_{tier}.jsonl` for every reranking run.
