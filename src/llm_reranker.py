"""LLM-judge reranker: given an anchor and its top-10 embedding candidates,
ask the model which candidate relies on the same underlying mathematical
technique, explicitly instructed to ignore surface form.

Judge backend is pluggable (see JudgeBackend / GeminiJudgeBackend /
OpenAICompatJudgeBackend below) so a second judge model -- e.g. a future
lab-hosted GLM/Qwen model -- can be swapped in without touching the prompt,
parsing, caching, or reranking logic. Two judges agreeing on a result is what
would show the finding isn't an artifact of one specific model.
"""
from __future__ import annotations

import json
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"  # repo root
CACHE_DIR = Path(__file__).resolve().parent.parent / "llm_reranker_cache"
MAX_RETRIES = 6

PROMPT_TEMPLATE = """You are given a math competition problem ("ANCHOR") and 10 candidate \
problems, labeled 1 through 10. Exactly one candidate relies on the same underlying \
mathematical technique or method as the anchor -- the same core idea you would actually use \
to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared \
language, shared story framing (e.g. both about chessboards, or both in the same language), or \
shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to \
the anchor and still use a completely different technique, and a candidate can look nothing like \
the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve \
each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Respond with ONLY the candidate number (1-10). No explanation, no other text."""

# CoT variant: the terse prompt above ("ONLY the number, no explanation") turned
# out to suppress deliberation in glm-5.2-fp8 specifically -- ~2 completion
# tokens/call and occasional `<arg_value>N` template artifacts, vs 288 reasoning
# tokens on an unrelated riddle in the same session. A 50-query diagnostic
# (scripts/test_glm_cot_prompt.py) confirmed this prompt materially improves
# GLM's Hit@1 (4%->10%) with real, substantive reasoning. Used for both judges
# once confirmed, so the comparison stays fair -- changing the prompt for only
# one judge would invalidate it.
COT_PROMPT_TEMPLATE = """You are given a math competition problem ("ANCHOR") and 10 candidate \
problems, labeled 1 through 10. Exactly one candidate relies on the same underlying \
mathematical technique or method as the anchor -- the same core idea you would actually use \
to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared \
language, shared story framing (e.g. both about chessboards, or both in the same language), or \
shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to \
the anchor and still use a completely different technique, and a candidate can look nothing like \
the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve \
each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Think step by step. First state the core technique needed to solve the ANCHOR. Then, for each \
candidate in turn, briefly note the technique it actually needs and whether it matches the \
anchor's. After going through all 10, conclude.

End your response with your final answer on its own line, in exactly this format:
FINAL ANSWER: <candidate number>"""

# Concise-CoT variant: the plain COT_PROMPT_TEMPLATE above let glm-5.2-fp8
# deliberate at unbounded length -- 63.3% of full-run responses hit a 6,144-token
# cap without ever reaching a conclusion (see results/llm_reranker_cot_full_comparison.md
# correction). This variant keeps the same task and "ignore surface form"
# instruction, but explicitly asks for concise reasoning per candidate, to test
# whether that alone gets truncation down to a usable rate before spending a
# full run on a larger token budget.
COT_CONCISE_PROMPT_TEMPLATE = """You are given a math competition problem ("ANCHOR") and 10 candidate \
problems, labeled 1 through 10. Exactly one candidate relies on the same underlying \
mathematical technique or method as the anchor -- the same core idea you would actually use \
to solve it -- even though it may look completely different on the surface.

IGNORE surface-level similarity when deciding: shared variable names, shared wording, shared \
language, shared story framing (e.g. both about chessboards, or both in the same language), or \
shared numbers do NOT mean same technique. A candidate can look nearly identical in phrasing to \
the anchor and still use a completely different technique, and a candidate can look nothing like \
the anchor on the surface and still use the exact same technique.

Focus only on: what mathematical concept, theorem, or method would you actually use to solve \
each problem. Which candidate needs the same one as the anchor?

ANCHOR:
{anchor}

CANDIDATES:
{candidates}

Think step by step, but be CONCISE -- this is a triage judgment, not a full solution write-up. \
State the core technique needed for the ANCHOR in one sentence. Then for each candidate, in one \
short phrase each, name its technique and say match or no match -- do not re-derive or fully \
solve any candidate. If you are torn between two candidates, pick the better one directly rather \
than re-litigating the comparison at length.

End your response with your final answer on its own line, in exactly this format:
FINAL ANSWER: <candidate number>"""


def _read_env_var(name: str) -> str:
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"{name} not found in {ENV_PATH}")


class _HttpError(Exception):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:300]}")
        self.status = status
        self.text = text


class JudgeBackend(ABC):
    """A judge backend just needs to turn a prompt into raw text. Retry/backoff
    is shared (see _call_with_retry), so a new backend only implements the one
    HTTP call for its API shape."""

    name: str

    @abstractmethod
    def call(self, prompt: str) -> str:
        ...


class GeminiJudgeBackend(JudgeBackend):
    """Google AI Studio, native Gemini REST API (generateContent).

    Gemini 3 models (confirmed with gemini-3-flash-preview) are thinking
    models by default: they can spend their whole max_tokens budget on hidden
    thought tokens before writing any visible text (finishReason=MAX_TOKENS,
    empty content, no crash but no answer either), and even when they do write
    visible text they don't reliably respect a terse "answer only" instruction
    -- confirmed via direct API test, a grading prompt got discursive
    open-ended reasoning that ran past a 2048-token budget without ever
    stating a final score. disable_thinking=True sets thinkingConfig.
    thinkingBudget=0, which fixed this in testing (clean single-token replies).
    Leave it False for prompts that intentionally want visible reasoning
    (e.g. COT_PROMPT_TEMPLATE) -- this flag is for terse-answer use cases
    (grading, the original terse reranker prompt) where reasoning isn't wanted
    and burns budget/time for nothing."""

    def __init__(self, model: str, max_tokens: int = 16, disable_thinking: bool = False):
        self.model = model
        self.name = f"gemini:{model}"
        self.max_tokens = max_tokens
        self.disable_thinking = disable_thinking
        self._api_key = _read_env_var("GEMINI_API_KEY")

    def call(self, prompt: str) -> str:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        generation_config = {"temperature": 0.0, "maxOutputTokens": self.max_tokens}
        if self.disable_thinking:
            generation_config["thinkingConfig"] = {"thinkingBudget": 0}
        resp = requests.post(
            f"{url}?key={self._api_key}",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": generation_config,
            },
            timeout=120,
        )
        if resp.status_code == 200:
            content = resp.json()["candidates"][0].get("content", {})
            parts = content.get("parts")
            # Gemini 3 thinking models can burn the whole max_tokens budget on
            # hidden thought tokens (thoughtsTokenCount) before writing any
            # visible text -- finishReason=MAX_TOKENS with an empty {} content
            # and no "parts" key at all, rather than an error. Treat as empty
            # output (caller's parser will just find nothing) instead of
            # crashing; the fix that actually matters is sizing max_tokens
            # generously enough that this doesn't happen in practice.
            if not parts:
                return ""
            return parts[0]["text"]
        raise _HttpError(resp.status_code, resp.text)


class OpenAICompatJudgeBackend(JudgeBackend):
    """Any OpenAI-compatible /chat/completions endpoint -- DeepInfra, a
    lab-hosted vLLM/sglang server (e.g. GLM 5.2 / Qwen-122b), etc.
    api_key_env_var may be unset/dummy for internal servers that don't check it.

    Reasoning models (confirmed with GLM 5.2 on an internal lab sglang
    deployment) route their answer through a separate `reasoning_content`
    field and leave `content` empty even when explicitly told to answer with
    only a number -- this is a server/template behavior, not something a
    prompt instruction can override. max_tokens defaults higher than the
    16 that's plenty for a terse non-reasoning model, since a reasoning model
    spends real budget on chain-of-thought before it states an answer."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env_var: str | None = None,
        max_tokens: int = 16,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = f"openai-compat:{model}"
        self.max_tokens = max_tokens
        self._api_key = _read_env_var(api_key_env_var) if api_key_env_var else "dummy"

    def call(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": self.max_tokens,
            },
            timeout=300,
        )
        if resp.status_code == 200:
            message = resp.json()["choices"][0]["message"]
            content = message.get("content") or ""
            if not content.strip():
                content = message.get("reasoning_content") or ""
            return content
        raise _HttpError(resp.status_code, resp.text)


def _call_with_retry(backend: JudgeBackend, prompt: str, cache_name: str, log) -> str:
    attempt = 0
    while True:
        try:
            return backend.call(prompt)
        except _HttpError as e:
            if e.status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                wait = min(60, 2**attempt)
                log(f"[llm/{cache_name}] {e.status}, retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                attempt += 1
                continue
            raise
        except requests.exceptions.RequestException as e:
            # Connection-level failure (read timeout, connection reset, DNS
            # hiccup) -- no HTTP status at all, distinct from _HttpError
            # above, but just as transient and worth retrying the same way.
            if attempt < MAX_RETRIES:
                wait = min(60, 2**attempt)
                log(f"[llm/{cache_name}] {type(e).__name__}, retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                time.sleep(wait)
                attempt += 1
                continue
            raise


def _build_prompt(anchor_text: str, candidate_texts: list[str], template: str = PROMPT_TEMPLATE) -> str:
    candidates_block = "\n\n".join(f"{i+1}. {t}" for i, t in enumerate(candidate_texts))
    return template.format(anchor=anchor_text, candidates=candidates_block)


def _parse_choice(response_text: str, n: int) -> int | None:
    """Last in-range integer wins, not the first. A terse non-reasoning
    response is just the one number either way. A reasoning trace mentions
    many numbers along the way (step counts, candidate numbers referenced
    while thinking) and states its actual conclusion last."""
    matches = re.findall(r"\d+", response_text)
    for raw in reversed(matches):
        val = int(raw)
        if 1 <= val <= n:
            return val
    return None


def _parse_choice_cot(response_text: str, n: int) -> int | None:
    """For COT_PROMPT_TEMPLATE: prefer the explicit 'FINAL ANSWER: <n>' marker
    (last occurrence, in case it's echoed more than once), falling back to the
    last-in-range-integer heuristic if the model didn't use the marker."""
    markers = re.findall(r"final answer:?\s*(\d+)", response_text, re.IGNORECASE)
    for raw in reversed(markers):
        val = int(raw)
        if 1 <= val <= n:
            return val
    return _parse_choice(response_text, n)


def rerank_top10_llm(
    query_ids: list[str],
    results: dict[str, dict[str, float]],
    query_texts: dict[str, str],
    corpus_texts: dict[str, str],
    backend: JudgeBackend,
    cache_name: str,
    prompt_template: str = PROMPT_TEMPLATE,
    parse_fn=_parse_choice,
    max_workers: int = 1,
    log=print,
) -> dict[str, str | None]:
    """Returns query_id -> chosen top-1 corpus id (or None if the model's
    response couldn't be parsed after retries). Caches raw responses to disk,
    keyed by query_id, so a rerun costs nothing. cache_name should encode the
    judge backend AND prompt variant (e.g. include backend.name) if you'll run
    multiple judges/prompts over the same query set, so their caches don't
    collide. max_workers > 1 issues judge calls concurrently (threaded, I/O
    bound) -- worth it for a reasoning-model judge where each call can take
    ~15-20s; a single-threaded sglang/vLLM server generally batches concurrent
    requests fine rather than queuing them serially."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                d = json.loads(line)
                cache[d["query_id"]] = d

    choices: dict[str, str | None] = {}
    todo = [qid for qid in query_ids if qid not in cache]
    log(f"[llm/{cache_name}] backend={backend.name}  {len(query_ids) - len(todo)} cached, {len(todo)} to judge")

    out = open(cache_path, "a")
    write_lock = threading.Lock()
    progress = {"n": 0}

    def process_one(qid: str) -> None:
        top10 = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:10]
        top10_ids = [cid for cid, _ in top10]
        candidate_texts = [corpus_texts[cid] for cid in top10_ids]
        prompt = _build_prompt(query_texts[qid], candidate_texts, prompt_template)

        raw = _call_with_retry(backend, prompt, cache_name, log)
        choice_idx = parse_fn(raw, len(top10_ids))
        chosen_id = top10_ids[choice_idx - 1] if choice_idx else None
        record = {
            "query_id": qid, "raw_response": raw, "choice_idx": choice_idx,
            "chosen_id": chosen_id, "top10_ids": top10_ids, "backend": backend.name,
        }
        with write_lock:
            out.write(json.dumps(record) + "\n")
            out.flush()
            cache[qid] = record
            progress["n"] += 1
            if progress["n"] % 25 == 0 or progress["n"] == len(todo):
                log(f"[llm/{cache_name}] {progress['n']}/{len(todo)} judged")

    try:
        if max_workers > 1 and todo:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                list(ex.map(process_one, todo))
        else:
            for qid in todo:
                process_one(qid)
    finally:
        out.close()

    for qid in query_ids:
        choices[qid] = cache[qid]["chosen_id"]
    return choices
