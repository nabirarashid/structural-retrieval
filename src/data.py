"""Load MathNet-Retrieve (easy tier) and build strict/lenient qrels + query samples."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

ID_PATTERN = re.compile(r"^(.+)::(eq|nm)::(.+)$")


def load_jsonl(path: Path) -> dict[str, str]:
    """id -> text"""
    out = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            out[d["_id"]] = d["text"]
    return out


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    """query-id -> {corpus-id: score}"""
    out: dict[str, dict[str, int]] = {}
    with open(path) as f:
        next(f)  # header
        for line in f:
            qid, cid, score = line.rstrip("\n").split("\t")
            out.setdefault(qid, {})[cid] = int(score)
    return out


def parse_corpus_id(cid: str) -> tuple[str, str | None, str | None]:
    """Returns (base_id, kind, suffix). kind is 'eq'/'nm'/None (unsuffixed)."""
    m = ID_PATTERN.match(cid)
    if m:
        base, kind, suf = m.groups()
        return base, kind, suf
    return cid, None, None


@dataclass
class MathNetEasy:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels_strict: dict[str, dict[str, int]]
    tier: str = "easy"
    _base_index: dict[str, list[str]] | None = None

    @classmethod
    def load(cls, tier: str = "easy") -> "MathNetEasy":
        # corpus.jsonl and queries.jsonl are byte-identical across tiers
        # (verified via sha1) -- only qrels differs, so which tier's folder
        # we read corpus/queries from doesn't matter, only qrels does.
        corpus = load_jsonl(DATA_DIR / "easy" / "corpus.jsonl")
        queries = load_jsonl(DATA_DIR / "easy" / "queries.jsonl")
        qrels = load_qrels(DATA_DIR / tier / "qrels" / "test.tsv")
        return cls(corpus=corpus, queries=queries, qrels_strict=qrels, tier=tier)

    def qrels_for_tier(self, tier: str) -> dict[str, dict[str, int]]:
        """Load a different tier's strict qrels without reloading corpus/queries."""
        return load_qrels(DATA_DIR / tier / "qrels" / "test.tsv")

    def base_index(self) -> dict[str, list[str]]:
        """base_id -> [corpus ids] for eq/nm entries only, built once."""
        if self._base_index is None:
            idx: dict[str, list[str]] = {}
            for cid in self.corpus:
                base, kind, _ = parse_corpus_id(cid)
                if kind in ("eq", "nm"):
                    idx.setdefault(base, []).append(cid)
            self._base_index = idx
        return self._base_index

    def sample_queries(self, n: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        all_qids = sorted(self.queries.keys())  # sort first: dict order isn't a stable seed input
        return rng.sample(all_qids, n)

    def lenient_qrels_for(self, query_ids: list[str]) -> dict[str, dict[str, int]]:
        """Any ::eq:: sibling of the query's own base counts as relevant, not just
        the strict tier-specific target."""
        lenient: dict[str, dict[str, int]] = {}
        for qid in query_ids:
            base, _, _ = parse_corpus_id(qid)
            siblings = {
                cid: 1
                for cid in (f"{base}::eq::easy", f"{base}::eq::medium", f"{base}::eq::hard")
                if cid in self.corpus
            }
            lenient[qid] = siblings
        return lenient

    def small_smoke_pool(self, query_ids: list[str], seed: int, target_size: int = 1500) -> dict[str, str]:
        """A reduced corpus for pipeline smoke-testing only -- NOT valid for the
        validation gate (Recall@k is pool-size-sensitive)."""
        rng = random.Random(seed)
        idx = self.base_index()
        required: set[str] = set()
        for qid in query_ids:
            base, _, _ = parse_corpus_id(qid)
            required.update(idx.get(base, []))
        remaining_budget = max(target_size - len(required), 0)
        pool_rest = [cid for cid in self.corpus if cid not in required]
        filler = rng.sample(pool_rest, min(remaining_budget, len(pool_rest)))
        keep_ids = required | set(filler)
        return {cid: self.corpus[cid] for cid in keep_ids}
