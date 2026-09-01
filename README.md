# Retrieved but not ranked: surface-form bias in structural retrieval

**Paper PDF:** [`paper/paper.pdf`](paper/paper.pdf)
arXiv version pending.

## Abstract

Embedding retrieval is typically validated on tasks where surface form and semantic content align. We study the case where they are deliberately separated, retrieving items that share underlying structure while differing in surface form, in two unrelated domains under one protocol: competition mathematics (MathNet-Retrieve; 500 queries against a 117,088-item corpus) and embodied-agent trajectories (ALFWorld-derived; 118 queries against 336 trajectories). In mathematics the failure is total and precisely located: strict Hit@1 at the heaviest disguise tier is 0.0% for both production embedders (bootstrap 95% CI [0.0, 0.0]) while the correct item sits in the top 10 nearly always, and in 95.2 to 99.8% of misses the winning candidate is more lexically similar to the query than the correct answer. In trajectories, where surface variation is incidental rather than adversarial, the same models sit at or near hypergeometric chance when gold requires a different target object, and fall below chance for all three embedders once gold requires a different object and receptacle, indicating that retrieval anchors on literal tokens rather than task structure. A lexical reranker control hurts in mathematics yet helps in trajectories (closing 26 to 36% of the recoverable gap, CIs excluding zero); which sign it takes turns out to depend on how the benchmark's surface variation was constructed, adversarial or incidental, so the control doubles as a cheap diagnostic. An LLM reranker recovers 5 to 63% of the gap in mathematics and 43 to 76% in trajectories, with direction replicating across three independently trained judges (all twenty-one judge-by-configuration cells positive) while nothing about magnitude transfers: effect sizes, tier profiles, and even which judge is the outlier all change with domain, with paired bootstrap differences between judges excluding zero in every configuration. Reranking gains in mathematics concentrate on well-known competitions (+19.8 points, CI [+6.7, +33.2] in one of six judge-by-candidate cells), so part, though not all, of the recovery reflects memorization. Finally, in a paired downstream experiment (210 queries, two graders at 96 to 99% agreement), oracle retrieval was statistically indistinguishable from adversarially bad retrieval (McNemar p = 0.678), and a complete-answers-only analysis shows why: the solver's 69.5% zero-shot accuracy is largely a truncation proxy, with 97 to 100% accuracy on answers that finish within budget, leaving retrieval almost no headroom to act on.

## Findings at a glance

- Embedding retrieval tracks literal surface content over structure in both domains: 0% strict Hit@1 under heavy disguise in math (with the answer sitting in the top 10), and below-chance retrieval in trajectories once gold excludes the query's literal tokens.
- The lexical-control sign flip: the same cheap reranker hurts in math and helps in trajectories, so its sign diagnoses whether a benchmark's surface variation is adversarial or incidental, a free diagnostic any evaluation should report.
- LLM reranking recovers real gap in both domains, but only the direction is portable: magnitudes, tier profiles, and even which judge is the outlier all change with domain and query style.

This repository contains the code, results, and reproduction materials behind the paper (Nabira
Rashid, 2026; full text: `results/paper.md`). Two deliberately unlike domains, competition
mathematics and embodied-agent trajectories, are evaluated under one shared protocol, so what
follows is meant to generalize past a single benchmark's quirks.

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
embedder mid-project: see the 2026-08-14 entry in `results/JOURNEY_LOG.md`. If you need newer
versions, upgrade `torch` first and re-resolve everything else together, then re-run the checks in
"Clean-clone reproduction" below before trusting any new number.

**API access.** `.env.example` lists every credential this project's code reads, with a note on
what each one unlocks:

| Variable | Unlocks | Public? |
|---|---|---|
| `GEMINI_API_KEY` | `gemini-embedding-001` embeddings; `gemini-3.1-flash-lite` reranker judge; `gemini-3-flash-preview` RAG grader | Yes: [aistudio.google.com](https://aistudio.google.com/apikey) |
| `DEEPINFRA_API_KEY` | Qwen3-Embedding-8B (DeepInfra-hosted) embeddings; DeepInfra solver comparison | Yes: [deepinfra.com](https://deepinfra.com) |
| `DEEPSEEK_API_KEY` | `deepseek-v4-flash` as the utility-curve solver (native API, not DeepInfra's checkpoint) | Yes: [platform.deepseek.com](https://platform.deepseek.com) |
| `ANTHROPIC_API_KEY` | `claude-haiku-4-5` as the third reranker judge (both domains): see `scripts/task2_haiku_reranker_full.py` | Yes: [console.anthropic.com](https://console.anthropic.com) |
| `MANTIS_EMBEDDINGS_BASE_URL` | The second Qwen3-Embedding-8B deployment used in the §4 "deployment divergence" comparison | **No: private lab host, not publicly reachable** |
| `MANTIS_LLM_BASE_URL` | `glm-5.2-fp8` as the second reranker judge / RAG grader B | **No: private lab host, not publicly reachable** |

The two lab-hosted results (deployment divergence, and every `glm-5.2-fp8` number) are **not
independently re-runnable** outside the lab that produced them: there is no public substitute
endpoint. Every other result in the paper uses only publicly-obtainable API access.

## What you can and can't reproduce without credentials

**Without any credentials, a clean clone gives you every final number in this project**: all of
`results/*.json` and `results/*.md` are committed, final, and readable/citable with nothing but a
text editor. `results/FINAL_NUMBERS.md` is the flat digest of all of them; start there.

**No script in `scripts/` runs end-to-end on a clean clone with zero setup, but the setup is now
one command per domain, not a recovery project.** Verified directly (`git clone` into a scratch
directory, fresh venv, no `.env`): the first thing every script needs is the raw source data under
`data/` (math) or the trajectory corpus at a local scratch path (trajectory domain), both
gitignored correctly (they're the upstream datasets, not ours to vendor). Run the three fetch
scripts (`scripts/fetch_mathnet_retrieve.py`, `scripts/fetch_procedural_memory_benchmark.py`,
`scripts/fetch_alfworld_valid_unseen.py`, see "Data" below) and this step is done; none need
credentials, since all three sources are public.

Once the data is fetched, the next blocker is credentials: every script recomputes its numbers from
raw embeddings (`embeddings_cache/`) and/or per-query judge/grader responses (`*.jsonl` caches), both
gitignored (too large to vendor, and the `*.jsonl` caches embed the source datasets' own query/
problem text in their cached prompts, which isn't ours to redistribute either). `VectorCache`
(`src/vector_cache.py`) never auto-fetches a missing embedding: every embedding has to be computed
once via a live API call before any downstream script can read it back. The one partial exception is
`all-MiniLM-L6-v2`, which embeds locally with no API key, but every script that uses it also builds
rankings for the other two embedders in the same pass, so it still needs at least `GEMINI_API_KEY`
(or `DEEPINFRA_API_KEY`) to complete a run.

Practically: to regenerate any specific number from scratch, first run the fetch scripts (see
"Data"), then get `GEMINI_API_KEY` (cheapest path: it covers embeddings, one reranker judge, and
one grader) and expect the relevant script to spend a small amount rebuilding the caches it needs
before it produces new output. Total historical cost of every result in this repo was $17.24 across
4,338 API calls (`results/SPEND.json`). Every publicly re-runnable result needs only public API keys
and costs well under that in total; two result families (all `glm-5.2-fp8` numbers and the
deployment-divergence comparison) used private lab endpoints and are not independently re-runnable,
as marked above.

## Reproduction: paper section → script → raw output

Mirrors `results/paper.md`'s own section numbers. **Run the data-fetch scripts first** (see "Data"
below): every row after this one assumes `data/` and the trajectory scratch path already exist.

| Paper §, Table/Fig | Script(s) | Raw output |
|---|---|---|
| Data setup (run before anything else) | `scripts/fetch_mathnet_retrieve.py`, `scripts/fetch_procedural_memory_benchmark.py`, `scripts/fetch_alfworld_valid_unseen.py` | `data/{easy,medium,hard}/...`, `/tmp/proced_mem_bench_check/...`, `/tmp/alfworld_test.jsonl` (each SHA256/count-verified against what every result below was computed on) |
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
| §4/§5 paired judge-difference CIs | `scripts/task_paired_judge_diff_cis.py` | `results/paired_judge_diff_cis.json` / `.md` |
| §5 query expansion | `scripts/task1_expand_queries.py`, `scripts/task1_filter_and_sample.py` | `results/task1_expanded_tier_labels.json` |
| §5 Table 4, baseline + tier-definition robustness | `scripts/task1_full_rerun.py`, `scripts/task2a_tier_robustness.py` | `results/task1_expanded_full_results.json`, `task2a_tier_robustness.json` |
| §5 lexical control + verb/noun ablation | `scripts/task1_full_rerun.py`, `scripts/mechanism_ablation.py` | `results/task1_expanded_full_results.json`, `mechanism_ablation_results.json` |
| §5 Table 5, LLM reranking + judge reversal | `scripts/task_traj_reranker_n118.py` | `results/task_traj_reranker_n118.json` |
| §5 label verification | `scripts/classify_agentinstruct_task_types.py` | `results/agentinstruct_task_type_labels.json`, `*_review_sample.json` |
| §6 downstream utility curve | `scripts/run_utility_curve_deepseek.py` | `results/utility_curve_deepseek.json`, `utility_curve_cache/utility_curve_deepseek_cache.jsonl` |
| §6 complete-answers-only robustness check | `scripts/task_complete_answers_robustness.py` | console output only; numbers recorded in `results/FINAL_NUMBERS.md` §3 (no separate JSON artifact) |
| §6 superseded 6-condition pilot | `scripts/run_rag_pilot.py` | `results/rag_pilot.md` (correction banner), `_discarded_glm_solver_pilots/` |
| §6 abandoned GLM solver run | `scripts/run_utility_curve_glm.py` | `utility_curve_cache/utility_curve_glm_cache.jsonl` (partial, unused) |
| §8 integrity incidents | no single script: see `results/JOURNEY_LOG.md` for the dated diagnosis of each, and `results/FINAL_NUMBERS.md` §4 for the one-line summary of all eight |
| Third judge, attempt 1 (DeepSeek, abandoned) | `scripts/task3_deepseek_third_judge.py` | none (infeasibility established by direct probing, unbounded reasoning, not a full run) |
| Third judge, attempt 2 (Claude Haiku 4.5, succeeded, both domains) | `scripts/task2_haiku_pilot.py` (10-call pilot), `scripts/task2_haiku_reranker_full.py` | `results/task2_haiku_reranker_full.json`: Haiku is the outlier in the math domain (tier-inverted vs. both other judges), closely corroborates Gemini in the trajectory domain (where GLM is the outlier instead). Reflected in paper Sections 4, 5, 8, and 9. |

**The numbers digest.** `results/FINAL_NUMBERS.md` is the single authoritative, flat list of every
number the paper cites, with its source file and a 95% CI where one was computed: read it before
citing any number from a section-specific `.md` file, since it's the most recently reconciled
source and states explicitly where it corrects an earlier document. `results/RESULTS_SUMMARY.md`
and `results/RETRIEVAL_WRITEUP.md` are the narrative writeups the digest was checked against.

## Evaluation integrity

This project surfaced eight documented data-integrity incidents during development, most involving
silent truncation that parsed as valid output and would have shipped as a clean number without a
dedicated audit. Standing practice since: finish-reason audits after every generation run,
per-condition truncation reporting wherever a budget cap exists, and recomputation verified against
frozen aggregates before any number is trusted. Full incident-by-incident detail is in
`results/paper.md` Section 8, with the one-line summary of all eight in `results/FINAL_NUMBERS.md`
Section 4. These practices, not their absence, are why the numbers in this repository can be
trusted.

## Data

This repository does not vendor **either** source dataset, and the two are excluded by two different
mechanisms, worth stating plainly rather than leaving to infer:

- **MathNet-Retrieve** (math domain): raw data lives in `data/` *inside* this repo's working tree,
  and is excluded via `.gitignore`. A reference clone of the MathNet paper's own repository
  (`MathNet/`, images/README only) is gitignored the same way.
- **Procedural-memory / ALFWorld trajectories** (trajectory domain): the raw corpus is never inside
  this repo's directory tree in the first place; every trajectory script reads it from a local
  scratch path (`/tmp/proced_mem_bench_check/...` in the scripts as committed), entirely outside
  `embedding-benchmark/`. It was never a candidate for tracking, not just excluded after the fact.

No dataset files from either source are committed anywhere in this repo (verified directly: a
full-history `git log` and a working-tree scan of every file, not just tracked ones, turn up none).

**Fetch scripts (resolved 2026-08-17).** The gap noted by the clean-clone test, no script existed
to rebuild either domain's data from a fresh clone, is closed. Three scripts, each downloading
from the real upstream source and verifying every file by SHA256 (and, where content rather than
byte-identity is what matters downstream, by exact count/text match) against what this project's
results were actually computed on, **failing loudly, not silently, on any mismatch**:

```bash
python3 scripts/fetch_mathnet_retrieve.py           # -> data/{easy,medium,hard}/...
python3 scripts/fetch_procedural_memory_benchmark.py # -> /tmp/proced_mem_bench_check/...
python3 scripts/fetch_alfworld_valid_unseen.py       # -> /tmp/alfworld_test.jsonl
```

None require credentials: all three sources are public.

- **Math**: `data/{easy,medium,hard}/{corpus.jsonl,queries.jsonl,qrels/test.tsv}` comes from
  [`ShadenA/MathNet-Retrieve`](https://huggingface.co/datasets/ShadenA/MathNet-Retrieve), a
  **separate** Hugging Face dataset repo from the raw problem corpus (`ShadenA/MathNet`), which
  only has the underlying problems, not the equivalence/near-miss retrieval pairing. Verified
  byte-exact (SHA256) against the data every math-domain result in this repo was computed on, all 9
  files.
- **Trajectories**: the AgentInstruct/ALFWorld corpus comes from
  [`github.com/qpiai/Proced_mem_bench`](https://github.com/qpiai/Proced_mem_bench) (Kohar and
  Krishnan, 2025, arXiv:2511.21730), pinned to the exact commit this project's results were computed
  against, verified by SHA256 + trajectory/query count on both files every trajectory-domain script
  reads. The 78 new `valid_unseen` queries (§5 of the paper) come from
  [`hkust-nlp/agentboard`](https://huggingface.co/datasets/hkust-nlp/agentboard) on Hugging Face,
  verified by SHA256, episode/unique-goal count, and a full text cross-check against every one of
  the 78 stored query texts this project used (78/78 confirmed present).
- **Why this matters beyond the original gap**: the trajectory corpus's local scratch copy actually
  *disappeared* mid-project, a routine `/tmp` cleanup between sessions cleared it, `.git/config`
  included, with no fetch script yet to recover it (see the 2026-08-17 entry in `JOURNEY_LOG.md`).
  These scripts are that fix, not just a clean-clone nicety: re-run them any time `/tmp` gets swept.
  Confirmed after writing them: re-running the n=118 trajectory reranker end-to-end against
  freshly-fetched data reproduces `results/task_traj_reranker_n118.json` byte-for-byte.

Both datasets remain governed by their own licenses, not this repository's; see `results/paper.md`'s
References for full citations:

- MathNet-Retrieve: Alshammari et al., *MathNet*, ICLR 2026 (arXiv:2604.18584)
- Procedural-memory benchmark / ALFWorld trajectories: Kohar and Krishnan (2025) (arXiv:2511.21730), building on Shridhar et al., *ALFWorld*, ICLR 2021 (arXiv:2010.03768)

## License

Code and this project's own generated results/documentation: MIT (`LICENSE`). Not the datasets
above, see "Data".

## Citing

```bibtex
@article{rashid2026retrieved,
  author        = {Rashid, Nabira},
  title         = {Retrieved but not ranked: surface-form bias in structural retrieval, from mathematics to agent trajectories},
  year          = {2026},
  eprint        = {TODO: arXiv ID (assign at submission)},
  archivePrefix = {arXiv}
}
```

## Acknowledgments

Thanks to Anas Maarouf for access to the lab-hosted embedding and language-model endpoints used in parts of this study and for feedback on the draft, to Pranava Kumar for early discussions of the team's evaluation priorities that helped shape the direction, and to the Agents and Reasoning team at the MIT CSAIL Kellis Lab for mentorship and compute infrastructure.
