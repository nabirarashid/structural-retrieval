"""Internal lab embedding deployment (lab-hosted, vLLM-served Qwen3-Embedding-8B) client: batched,
memmap-cached, adaptive backoff, resumable. Same model/dim as embed_deepinfra.py --
used to check whether the two deployments agree on identical text (pooling /
normalization / quantization differences would show up as divergence).

Base URL is internal infra and lives only in .env (MANTIS_EMBEDDINGS_BASE_URL) --
never hardcode it, since this repo goes public."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from vector_cache import VectorCache

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"  # repo root
CACHE_DIR = Path(__file__).resolve().parent.parent / "embeddings_cache" / "labembed"
MODEL = "Qwen/Qwen3-Embedding-8B"
DIM = 4096

MAX_BATCH = 64
MAX_RETRIES = 6


def _read_env_var(name: str) -> str | None:
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _load_base_url() -> str:
    url = _read_env_var("MANTIS_EMBEDDINGS_BASE_URL")
    if not url:
        raise RuntimeError(f"MANTIS_EMBEDDINGS_BASE_URL not found in {ENV_PATH}")
    return url.rstrip("/")


def _load_api_key() -> str:
    # No real key was issued for this endpoint; a dummy Bearer token is accepted.
    # If MANTIS_EMBEDDINGS_API_KEY is later added to .env, prefer it.
    return _read_env_var("MANTIS_EMBEDDINGS_API_KEY") or "dummy"


class _HttpError(Exception):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:300]}")
        self.status = status
        self.text = text


def _call_batch(base_url: str, api_key: str, texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        f"{base_url}/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"input": texts, "model": MODEL, "encoding_format": "float"},
        timeout=120,
    )
    if resp.status_code == 200:
        data = resp.json()
        ordered = sorted(data["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in ordered]
    raise _HttpError(resp.status_code, resp.text)


def embed_all(items: dict[str, str], cache_name: str, log=print) -> VectorCache:
    ids_order = list(items.keys())
    cache = VectorCache(CACHE_DIR, cache_name, dim=DIM, capacity=len(items))
    todo = cache.missing(ids_order)
    log(f"[labembed/{cache_name}] {len(ids_order) - len(todo)} cached, {len(todo)} to embed")

    base_url = _load_base_url()
    api_key = _load_api_key()
    batch_size = MAX_BATCH
    i = 0
    while i < len(todo):
        chunk_ids = todo[i : i + batch_size]
        texts = [items[i_] for i_ in chunk_ids]
        attempt = 0
        while True:
            try:
                vecs = _call_batch(base_url, api_key, texts)
                cache.put_batch(chunk_ids, vecs)
                i += len(chunk_ids)
                if (i // batch_size) % 20 == 0 or i >= len(todo):
                    log(f"[labembed/{cache_name}] {i}/{len(todo)} embedded")
                break
            except _HttpError as e:
                if e.status == 400 and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    chunk_ids = todo[i : i + batch_size]
                    texts = [items[i_] for i_ in chunk_ids]
                    log(f"[labembed/{cache_name}] 400 error, shrinking batch to {batch_size}")
                    continue
                if e.status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    wait = min(60, 2**attempt)
                    log(f"[labembed/{cache_name}] {e.status}, retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise

    return cache
