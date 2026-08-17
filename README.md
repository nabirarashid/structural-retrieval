# Retrieved but not ranked: surface-form bias in structural retrieval

Code, results, and reproduction materials for the paper of the same name (Nabira Rashid, 2026;
draft in `results/paper_draft_v3.md`). We study retrieval that requires matching *structure* while
ignoring *surface form* — same underlying technique/procedure, different wording — in two unrelated
domains under one shared protocol: competition mathematics (MathNet-Retrieve, 500 queries against a
117,088-item corpus) and embodied-agent trajectories (ALFWorld-derived, 118 queries against 336
trajectories). In both domains, embedding retrieval tracks literal surface content over structure
whenever the two are separable — total failure in mathematics (0% rank-1 under heavy disguise, with
the right answer sitting lower in the list nearly every time), sub-chance retrieval in trajectories
once gold requires generalizing past the query's literal object and container. A cheap lexical
reranker control flips sign between the two domains (hurts in mathematics, helps in trajectories) —
a one-bit diagnostic of whether a benchmark's surface variation is adversarial or incidental — and
an LLM reranker recovers substantial ground in both, though its magnitude, and even which of two
judge models is stronger, is not portable across domains. See the abstract in
`results/paper_draft_v3.md` for the full claim set.

## Setup

Requires Python 3.11+ (developed on 3.12) and, for the free/cached analyses below, nothing else.

```bash
git clone <this-repo-url>
cd embedding-benchmark
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
cp .env.example .env   # only needed if you plan to re-run generation, not to read results
```

**Use exactly the pinned versions in `requirements.txt`.** This project hit a real incident where a
partial dependency upgrade (`numpy` alone, then `transformers` alone) silently broke the MiniLM
embedder mid-project — see the 2026-08-14 entry in `results/JOURNEY_LOG.md`. If you need newer
versions, upgrade `torch` first and re-resolve everything else together, then re-run the checks in
"Clean-clone reproduction" below before trusting any new number.

**API access.** `.env.example` lists every credential this project's code reads, with a note on
what each one unlocks:

| Variable | Unlocks | Public? |
|---|---|---|
| `GEMINI_API_KEY` | `gemini-embedding-001` embeddings; `gemini-3.1-flash-lite` reranker judge; `gemini-3-flash-preview` RAG grader | Yes — [aistudio.google.com](https://aistudio.google.com/apikey) |
| `DEEPINFRA_API_KEY` | Qwen3-Embedding-8B (DeepInfra-hosted) embeddings; DeepInfra solver comparison | Yes — [deepinfra.com](https://deepinfra.com) |
| `DEEPSEEK_API_KEY` | `deepseek-v4-flash` as the utility-curve solver (native API, not DeepInfra's checkpoint) | Yes — [platform.deepseek.com](https://platform.deepseek.com) |
| `ANTHROPIC_API_KEY` | `claude-haiku-4-5` as the third reranker judge (both domains) — see `scripts/task2_haiku_reranker_full.py` | Yes — [console.anthropic.com](https://console.anthropic.com) |
| `MANTIS_EMBEDDINGS_BASE_URL` | The second Qwen3-Embedding-8B deployment used in the §4 "deployment divergence" comparison | **No — private MIT CSAIL lab host, campus-network only** |
| `MANTIS_LLM_BASE_URL` | `glm-5.2-fp8` as the second reranker judge / RAG grader B | **No — private MIT CSAIL lab host, campus-network only** |

The two lab-hosted results (deployment divergence, and every `glm-5.2-fp8` number) are **not
independently re-runnable** outside the lab that produced them — there is no public substitute
endpoint. Every other result in the paper uses only publicly-obtainable API access.

## What you can and can't reproduce without credentials

**Without any credentials, a clean clone gives you every final number in this project** — all of
`results/*.json` and `results/*.md` are committed, final, and readable/citable with nothing but a
text editor. `results/FINAL_NUMBERS.md` is the flat digest of all of them; start there.

**No script in `scripts/` runs end-to-end on a clean clone with zero setup — but the setup is now
one command per domain, not a recovery project.** Verified directly (`git clone` into a scratch
directory, fresh venv, no `.env`): the first thing every script needs is the raw source data under
`data/` (math) or the trajectory corpus at a local scratch path (trajectory domain) — both
gitignored correctly (they're the upstream datasets, not ours to vendor). Run the three fetch
scripts (`scripts/fetch_mathnet_retrieve.py`, `scripts/fetch_procedural_memory_benchmark.py`,
`scripts/fetch_alfworld_valid_unseen.py` — see "Data" below) and this step is done; none need
credentials, since all three sources are public.

Once the data is fetched, the next blocker is credentials: every script recomputes its numbers from
raw embeddings (`embeddings_cache/`) and/or per-query judge/grader responses (`*.jsonl` caches), both
gitignored (too large to vendor, and the `*.jsonl` caches embed the source datasets' own query/
problem text in their cached prompts, which isn't ours to redistribute either). `VectorCache`
(`src/vector_cache.py`) never auto-fetches a missing embedding — every embedding has to be computed
once via a live API call before any downstream script can read it back. The one partial exception is
`all-MiniLM-L6-v2`, which embeds locally with no API key — but every script that uses it also builds
rankings for the other two embedders in the same pass, so it still needs at least `GEMINI_API_KEY`
(or `DEEPINFRA_API_KEY`) to complete a run.

Practically: to regenerate any specific number from scratch, first run the fetch scripts (see
"Data"), then get `GEMINI_API_KEY` (cheapest path — it covers embeddings, one reranker judge, and
one grader) and expect the relevant script to spend a small amount rebuilding the caches it needs
before it produces new output. Total historical spend for every result in this repo was $17.24
(`results/SPEND.json`); no single script comes close to that.

## Reproduction: paper section → script → raw output

Mirrors `results/paper_draft_v3.md`'s own section numbers. **Run the data-fetch scripts first** (see
"Data" below) — every row after this one assumes `data/` and the trajectory scratch path already
exist.

| Paper §, Table/Fig | Script(s) | Raw output |
|---|---|---|
| Data setup (run before anything else) | `scripts/fetch_mathnet_retrieve.py`, `scripts/fetch_procedural_memory_benchmark.py`, `scripts/fetch_alfworld_valid_unseen.py` | `data/{easy,medium,hard}/...`, `/tmp/proced_mem_bench_check/...`, `/tmp/alfworld_test.jsonl` — each SHA256/count-verified against what every result below was computed on |
| §3 Math validation gate | `scripts/run_baseline.py` | `results/baseline_gemini.json` |
| §3 Trajectory validation gate | `scripts/validate_minilm_reproduction.py` | `results/minilm_validation_gate.json` |
| §4 Table 1, baseline | `scripts/run_baseline.py`, `scripts/run_hard_tier.py` | `results/baseline_{gemini,deepinfra}[_hard].json` |
| §4 failure structure | `scripts/inspect_misses.py` | `results/strict_misses_examples.md`, `strict_misses_examples_raw.txt` |
| §4 lexical control | `scripts/run_dumb_reranker_control.py`, `scripts/lexical_distance_check.py` | `results/dumb_reranker_control.json`/`.md`, `lexical_distance_check.json`/`.md` |
| §4 Table 2, LLM reranking (terse) | `scripts/run_llm_reranker_full.py`, `scripts/run_llm_reranker_full_glm.py` | `results/llm_reranker_full.json`/`.md`, `llm_reranker_full_glm.json`, `llm_reranker_glm_judge.md` |
| §4 CoT + GLM-CoT exclusion | `scripts/run_llm_reranker_full_cot.py` | `results/llm_reranker_full_cot_{gemini,glm}.json`, `llm_reranker_cot_full_comparison.md`, `glm_cot_diagnostic.md` |
| §4 Table 3, contamination | `scripts/task2c_contamination_cis.py` | `results/task2c_contamination_cis.json` |
| §4 deployment divergence | `scripts/compare_deepinfra_labembed.py`, `scripts/run_baseline.py --provider labembed` | `results/deepinfra_vs_labembed.json`/`.md`, `deepinfra_vs_labembed_paired.json` |
| §4/§5 bootstrap CIs (both domains) | `scripts/task2b_bootstrap_cis.py` | `results/task2b_bootstrap_cis.json` |
| §5 query expansion | `scripts/task1_expand_queries.py`, `scripts/task1_filter_and_sample.py` | `results/task1_expanded_tier_labels.json` |
| §5 Table 4, baseline + tier-definition robustness | `scripts/task1_full_rerun.py`, `scripts/task2a_tier_robustness.py` | `results/task1_expanded_full_results.json`, `task2a_tier_robustness.json` |
| §5 lexical control + verb/noun ablation | `scripts/task1_full_rerun.py`, `scripts/mechanism_ablation.py` | `results/task1_expanded_full_results.json`, `mechanism_ablation_results.json` |
| §5 Table 5, LLM reranking + judge reversal | `scripts/task_traj_reranker_n118.py` | `results/task_traj_reranker_n118.json` |
| §5 label verification | `scripts/classify_agentinstruct_task_types.py` | `results/agentinstruct_task_type_labels.json`, `*_review_sample.json` |
| §6 downstream utility curve | `scripts/run_utility_curve_deepseek.py` | `results/utility_curve_deepseek.json`, `utility_curve_cache/utility_curve_deepseek_cache.jsonl` |
| §6 superseded 6-condition pilot | `scripts/run_rag_pilot.py` | `results/rag_pilot.md` (correction banner), `_discarded_glm_solver_pilots/` |
| §6 abandoned GLM solver run | `scripts/run_utility_curve_glm.py` | `utility_curve_cache/utility_curve_glm_cache.jsonl` (partial, unused) |
| §8 integrity incidents | no single script — see `results/JOURNEY_LOG.md` for the dated diagnosis of each, and `results/FINAL_NUMBERS.md` §4 for the one-line summary of all eight |
| Third judge, attempt 1 (DeepSeek) — abandoned | `scripts/task3_deepseek_third_judge.py` | none — infeasibility established by direct probing (unbounded reasoning), not a full run |
| Third judge, attempt 2 (Claude Haiku 4.5) — succeeded, both domains | `scripts/task2_haiku_pilot.py` (10-call pilot), `scripts/task2_haiku_reranker_full.py` | `results/task2_haiku_reranker_full.json` — Haiku is the outlier in the math domain (tier-inverted vs. both other judges), closely corroborates Gemini in the trajectory domain (where GLM is the outlier instead). **Not yet reflected in `paper_draft_v3.md` §8/§9, which still describe the third judge as abandoned — see the flag in `results/FINAL_NUMBERS.md`'s Discrepancies section.** |

**The numbers digest.** `results/FINAL_NUMBERS.md` is the single authoritative, flat list of every
number the paper cites, with its source file and a 95% CI where one was computed — read it before
citing any number from a section-specific `.md` file, since it's the most recently reconciled
source and states explicitly where it corrects an earlier document. `results/RESULTS_SUMMARY.md`
and `results/RETRIEVAL_WRITEUP.md` are the narrative writeups the digest was checked against.

## Data

This repository does not vendor **either** source dataset, and the two are excluded by two different
mechanisms — worth stating plainly rather than leaving to infer:

- **MathNet-Retrieve** (math domain): raw data lives in `data/` *inside* this repo's working tree,
  and is excluded via `.gitignore`. A reference clone of the MathNet paper's own repository
  (`MathNet/`, images/README only) is gitignored the same way.
- **Procedural-memory / ALFWorld trajectories** (trajectory domain): the raw corpus is never inside
  this repo's directory tree in the first place — every trajectory script reads it from a local
  scratch path (`/tmp/proced_mem_bench_check/...` in the scripts as committed), entirely outside
  `embedding-benchmark/`. It was never a candidate for tracking, not just excluded after the fact.

No dataset files from either source are committed anywhere in this repo (verified directly: a
full-history `git log` and a working-tree scan of every file, not just tracked ones, turn up none).

**Fetch scripts (resolved 2026-08-17).** The gap noted by the clean-clone test — no script existed
to rebuild either domain's data from a fresh clone — is closed. Three scripts, each downloading
from the real upstream source and verifying every file by SHA256 (and, where content rather than
byte-identity is what matters downstream, by exact count/text match) against what this project's
results were actually computed on — **failing loudly, not silently, on any mismatch**:

```bash
python3 scripts/fetch_mathnet_retrieve.py           # -> data/{easy,medium,hard}/...
python3 scripts/fetch_procedural_memory_benchmark.py # -> /tmp/proced_mem_bench_check/...
python3 scripts/fetch_alfworld_valid_unseen.py       # -> /tmp/alfworld_test.jsonl
```

None require credentials — all three sources are public.

- **Math**: `data/{easy,medium,hard}/{corpus.jsonl,queries.jsonl,qrels/test.tsv}` comes from
  [`ShadenA/MathNet-Retrieve`](https://huggingface.co/datasets/ShadenA/MathNet-Retrieve) — a
  **separate** Hugging Face dataset repo from the raw problem corpus (`ShadenA/MathNet`), which
  only has the underlying problems, not the equivalence/near-miss retrieval pairing. Verified
  byte-exact (SHA256) against the data every math-domain result in this repo was computed on, all 9
  files.
- **Trajectories**: the AgentInstruct/ALFWorld corpus comes from
  [`github.com/qpiai/Proced_mem_bench`](https://github.com/qpiai/Proced_mem_bench) (Ishant and
  Krishnan, 2025, arXiv:2511.21730), pinned to the exact commit this project's results were computed
  against, verified by SHA256 + trajectory/query count on both files every trajectory-domain script
  reads. The 78 new `valid_unseen` queries (§5 of the paper) come from
  [`hkust-nlp/agentboard`](https://huggingface.co/datasets/hkust-nlp/agentboard) on Hugging Face —
  verified by SHA256, episode/unique-goal count, and a full text cross-check against every one of
  the 78 stored query texts this project used (78/78 confirmed present).
- **Why this matters beyond the original gap**: the trajectory corpus's local scratch copy actually
  *disappeared* mid-project — a routine `/tmp` cleanup between sessions cleared it, `.git/config`
  included, with no fetch script yet to recover it (see the 2026-08-17 entry in `JOURNEY_LOG.md`).
  These scripts are that fix, not just a clean-clone nicety — re-run them any time `/tmp` gets swept.
  Confirmed after writing them: re-running the n=118 trajectory reranker end-to-end against
  freshly-fetched data reproduces `results/task_traj_reranker_n118.json` byte-for-byte.

Both datasets remain governed by their own licenses, not this repository's — see
`results/paper_draft_v3.md`'s References for full citations:

- MathNet-Retrieve: Alshammari et al., *MathNet*, ICLR 2026 (arXiv:2604.18584)
- Procedural-memory benchmark / ALFWorld trajectories: Ishant and Krishnan (2025) (arXiv:2511.21730), building on Shridhar et al., *ALFWorld*, ICLR 2021 (arXiv:2010.03768)

## License

Code and this project's own generated results/documentation: MIT (`LICENSE`). Not the datasets
above — see "Data".
