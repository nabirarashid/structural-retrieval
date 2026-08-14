# Was GLM's weak reranking performance a prompt artifact, not weak judgment?

> ## ⚠️ CORRECTION (added after this doc was written) — the full-run verdict below was premature
>
> This doc's "materially better, confirmed" verdict (§2) was based on a 50-query diagnostic with a
> 4,096-token cap and no truncation check at the time. The subsequent full 500×4 GLM CoT run (6,144
> cap) turned out to have **63.3% of responses truncated mid-deliberation, not concluded** — see
> the correction banner in `results/llm_reranker_cot_full_comparison.md` for the full detail and a
> concrete truncated-response example. This diagnostic's own 50-query sample almost certainly had
> the same problem at a smaller scale (its `finish_reasons` set included `'length'`, meaning some
> of *these* responses were cut off too) — it just wasn't checked for at the time.
>
> **What still holds:** the core direction — CoT prompting engages real, substantive reasoning in
> GLM that the terse prompt suppressed — is still correct and visible in the qualitative examples
> below (the calculus and mod-9-invariant excerpts are genuine, complete, correct reasoning, not
> artifacts). **What doesn't hold:** the specific magnitude ("4%→10% Hit@1, 2.5x improvement") and
> the "confirmed" framing — those numbers weren't rigorously truncation-checked and shouldn't be
> cited as measured effect sizes.
>
> **Decided 2026-08-11: do not rerun GLM CoT.** A concise-reasoning-prompt test at 16,384 tokens
> (`scripts/test_glm_concise_cot.py`) got truncation down to 10.0% (3/30) — a real improvement, but
> a full rerun is still ~8.7 hours of wall time for what is a secondary question (cross-judge
> corroboration), not worth the cost. GLM CoT is excluded from the paper entirely; **GLM terse is
> used as the cross-judge data point instead** — it's the one GLM run that's genuinely clean
> (median 1 completion token, not truncated-long). Report framing: "contamination found robustly
> under Gemini across both prompts; GLM terse shows the same direction at borderline significance;
> GLM CoT excluded due to 63.3% truncation." See `results/llm_reranker_cot_full_comparison.md` for
> the full by-candidate-set numbers behind that framing.

## 1. Exact prompt sent to both judges (terse, production version)

```
You are given a math competition problem ("ANCHOR") and 10 candidate problems, labeled 1 through 10. Exactly one candidate relies on the same underlying mathematical technique or method as the anchor -- the same core idea you would actually use to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared language, shared story framing (e.g. both about chessboards, or both in the same language), or shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to the anchor and still use a completely different technique, and a candidate can look nothing like the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Respond with ONLY the candidate number (1-10). No explanation, no other text.
```

Byte-identical for both judges (source: `src/llm_reranker.py::PROMPT_TEMPLATE`).

### Raw responses, same 5 queries, both judges (hard tier, gemini-embed candidates)

| Query | gemini-3.1-flash-lite | glm-5.2-fp8 |
|---|---|---|
| sau_2012_bb6608 | `4` | `<arg_value>5` |
| bra_2019_7792eb | `4` | `2` |
| arg_2022_6b9544 | `5` | `<arg_value>3` |
| tur_2008_22f4a1 | `6` | `<arg_value>3` |
| fra_2014_eb9ec4 | `4` | `5` |

Gemini's raw output is a clean bare digit every time. GLM's is either a bare digit or a bare digit
prefixed with a stray `<arg_value>` token — looks like this deployment's chat template routes a
terse "just the number" instruction partway into a tool-call-argument code path rather than a
normal free-text answer, which would plausibly short-circuit whatever reasoning process the model
would otherwise run before committing to tool-call arguments.

## 2. Diagnostic: does GLM engage deliberation with an explicit chain-of-thought prompt?

50-query test (hard tier, gemini-embedding candidates, same fixed subsample used throughout this
project's validation runs). Two prompt variants sent to GLM only, both freshly called (not read
from the production cache) so completion-token accounting is apples-to-apples:

- **terse**: the production prompt above, unchanged.
- **cot**: same task description, plus an explicit "think step by step... state the core
  technique... for each candidate briefly note whether it matches... conclude with `FINAL ANSWER: <n>`
  on its own line" instruction. Same task, no relaxation of "ignore surface form."

| | terse | cot |
|---|---|---|
| Hit@1 | 2/50 (4.0%) | 5/50 (10.0%) |
| avg completion tokens | 84.1 | 3094.1 |
| avg reasoning tokens | 84.4 | 3085.6 |
| unparsed | 0 | 2 |
| wall time (50 calls) | 156s | 965s |

(orig Hit@1 for this 50-query subsample: 0/50; orig Hit@10: 27/50 — matches population baseline.)

Share of recoverable gap closed: terse 7.4%, cot 18.5% — **roughly 2.5x**. GLM's raw response
under CoT is genuine, substantive mathematical reasoning, not padding:

> *(sau_2012_bb6608)* "The core technique needed to solve the ANCHOR is to analyze the function
> f(x) = Σ√(x−k) − x for x≥n... by finding the maximum of f(x) using calculus (taking the
> derivative f'(x) = Σ 1/(2√(x−k)) − 1 and setting it to 0)..."

> *(bra_2019_7792eb)* "The core technique to solve this problem relies on **invariants modulo 9**.
> When we permute the digits of a number, its sum of digits remains exactly the same. Therefore,
> its value modulo 9 is invariant under digit permutation..."

Both are correct, relevant identifications of the actual solving technique — a categorically
different quality of output than a bare `<arg_value>3`.

**Verdict: materially better, confirmed.** This was a prompt artifact, not (necessarily) weak
judgment — the terse "no explanation" instruction was suppressing GLM's reasoning capability on
this task specifically. Per your instruction, proceeding to rerun all four configs with the CoT
prompt for **both** judges (changing it for GLM only would invalidate the comparison), currently
running in the background. GLM's CoT calls run ~15-20s each even with 6-way concurrency, so this
will take on the order of 1.5-2 hours; Gemini's CoT run (5-way concurrency, generally faster per
call) should finish sooner. Full results to follow once both complete.

## Framing update: contamination is expected to be judge-dependent

Worth stating plainly, and folding into how the earlier (terse-prompt) contamination results
should be read: **a "recognizes well-known competitions" effect is a property of a specific
model's training data, not a property of the task.** Gemini-3.1-flash-lite and GLM 5.2 were
trained on different corpora with different coverage of IMO/USAMO-style problems vs. regional
olympiads — so *some* judge-to-judge variation in the strength of a well_known-recognition effect
is the expected outcome, not evidence against the effect being real. Two judges showing the exact
same contamination magnitude would actually be a little surprising; two judges showing the same
*direction* on the same evidence, with different magnitude, is closer to what you'd predict from
two models with different training data both having *some* memorization advantage on famous
competitions.

Read against that expectation, the terse-prompt result holds up better than the initial "does it
replicate?" framing suggested: the well_known > rest gap on the **DeepInfra-embedding candidate
set** appeared under both judges, at similar (if judge-specific) magnitude — Gemini judge +8.1pt
(p=0.04–0.07), GLM judge +5.6pt (p=0.04–0.05). That's the finding that matters: genuine
cross-model corroboration of a real effect, not two independent coin flips that happened to agree.
The much larger effect on the Gemini-embedding candidate set that vanished under the GLM judge
(+19.8pt → +1.4pt) is better read as "the effect's *size* is judge-specific, consistent with
different memorized competition coverage" rather than "the effect isn't real." Whether this framing
changes once the CoT-prompt reruns land (below) is still open — noting it now since it applies to
the terse-prompt numbers regardless of what the reruns show.
