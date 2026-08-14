"""Dumb (lexical-only, zero math awareness) reranker: token overlap, edit
distance, length ratio. No semantics, no embeddings, no LaTeX/math parsing.

Purpose: control for the possibility that ::nm:: decoys are single-token
perturbations of the query text, in which case a purely lexical method could
score well by detecting generation artifacts rather than mathematical
structure. Must run before interpreting any smart (LLM/semantic) reranker.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def dumb_score(query_text: str, candidate_text: str) -> float:
    """Average of three purely lexical signals, each normalized to [0, 1]."""
    q_tok, c_tok = _tokens(query_text), _tokens(candidate_text)
    union = q_tok | c_tok
    jaccard = len(q_tok & c_tok) / len(union) if union else 0.0

    edit_sim = fuzz.ratio(query_text, candidate_text) / 100.0  # rapidfuzz, C-backed

    len_q, len_c = len(query_text), len(candidate_text)
    length_ratio = min(len_q, len_c) / max(len_q, len_c) if max(len_q, len_c) else 0.0

    return (jaccard + edit_sim + length_ratio) / 3.0


def rerank_top10(
    query_ids: list[str],
    results: dict[str, dict[str, float]],  # existing embedding-based ranking
    query_texts: dict[str, str],
    corpus_texts: dict[str, str],
) -> dict[str, list[str]]:
    """For each query, take its top-10 embedding candidates and reorder them by
    dumb_score alone. Returns query_id -> reranked top-10 id list."""
    reranked: dict[str, list[str]] = {}
    for qid in query_ids:
        top10 = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)[:10]
        qtext = query_texts[qid]
        scored = [(cid, dumb_score(qtext, corpus_texts[cid])) for cid, _ in top10]
        scored.sort(key=lambda kv: kv[1], reverse=True)
        reranked[qid] = [cid for cid, _ in scored]
    return reranked
