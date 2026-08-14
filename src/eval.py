"""BEIR-based Recall@k scoring (strict + lenient) and rank-1 failure taxonomy."""
from __future__ import annotations

from collections import Counter

import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval

from data import parse_corpus_id

K_VALUES = [1, 5, 10]


def build_results(
    query_ids: list[str],
    query_matrix: np.ndarray,  # (n_queries, dim), L2-normalized rows
    corpus_ids: list[str],
    corpus_matrix: np.ndarray,  # (n_corpus, dim), L2-normalized rows
    top_n: int = 200,
) -> dict[str, dict[str, float]]:
    """Full-corpus cosine similarity ranking for every sampled query -- every
    query IS scored against the entire corpus (no pool subsampling, per the
    plan's explicit instruction). Each query's stored result dict is then
    truncated to its top_n candidates; this only affects memory, not
    Recall@1/5/10 correctness, since a gold item outside the top 200 is
    provably outside the top 10 too."""
    sims = query_matrix @ corpus_matrix.T  # (n_queries, n_corpus) -- full-corpus scoring

    results: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(query_ids):
        row = sims[i]
        top_idx = np.argpartition(-row, top_n)[:top_n]
        top_idx = top_idx[np.argsort(-row[top_idx])]
        results[qid] = {corpus_ids[j]: float(row[j]) for j in top_idx}
    return results


def run_beir_eval(
    qrels: dict[str, dict[str, int]],
    results: dict[str, dict[str, float]],
) -> dict[str, float]:
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(qrels, results, K_VALUES)
    return {**ndcg, **_map, **recall, **precision}


def hit_at_k(
    query_ids: list[str],
    results: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int] = K_VALUES,
) -> dict[str, float]:
    """Binary per-query hit rate: is AT LEAST ONE relevant item in the top-k?
    This is what 'strict vs lenient' is actually meant to measure when a query
    can have multiple acceptable gold answers (the lenient ::eq:: siblings) --
    unlike BEIR's standard Recall@k, it isn't mechanically penalized by having
    more than one relevant item, since it doesn't divide by the relevant count."""
    out = {f"Hit@{k}": 0.0 for k in k_values}
    n = len(query_ids)
    for qid in query_ids:
        gold = set(qrels[qid].keys())
        ranked = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)
        ranked_ids = [cid for cid, _ in ranked]
        for k in k_values:
            if gold & set(ranked_ids[:k]):
                out[f"Hit@{k}"] += 1
    for k in k_values:
        out[f"Hit@{k}"] = round(out[f"Hit@{k}"] / n, 5)
    return out


def failure_taxonomy(
    query_ids: list[str],
    results: dict[str, dict[str, float]],
    qrels_strict: dict[str, dict[str, int]],
) -> tuple[Counter, list[dict]]:
    """For strict misses (rank-1 != gold), categorize what won instead."""
    categories: Counter = Counter()
    details: list[dict] = []
    for qid in query_ids:
        gold = set(qrels_strict[qid].keys())
        ranked = sorted(results[qid].items(), key=lambda kv: kv[1], reverse=True)
        top_id, top_score = ranked[0]
        if top_id in gold:
            continue  # correct at rank 1, not a miss

        q_base, _, _ = parse_corpus_id(qid)
        top_base, top_kind, top_suffix = parse_corpus_id(top_id)

        if top_base == q_base and top_kind == "eq":
            cat = "sibling_eq_variant"
        elif top_base == q_base and top_kind == "nm":
            cat = "own_nm_near_miss"
        elif top_kind is None:
            cat = "unsuffixed_distractor"
        else:
            cat = "other_base_item"

        categories[cat] += 1
        details.append(
            {
                "query_id": qid,
                "gold_id": next(iter(gold)),
                "top1_id": top_id,
                "top1_score": top_score,
                "category": cat,
            }
        )
    return categories, details


def to_matrix(ids: list[str], cache) -> np.ndarray:
    """cache: a VectorCache. Pulls only the requested ids into memory, not
    the whole cache."""
    m = cache.get_matrix(ids)
    m = m / np.linalg.norm(m, axis=1, keepdims=True)
    return m
