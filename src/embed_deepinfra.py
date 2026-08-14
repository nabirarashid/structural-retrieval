"""DeepInfra (Qwen3-Embedding-8B) client: batched, memmap-cached (not
RAM-loaded), adaptive backoff, resumable."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from vector_cache import VectorCache

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"  # repo root
CACHE_DIR = Path(__file__).resolve().parent.parent / "embeddings_cache" / "deepinfra"
MODEL = "Qwen/Qwen3-Embedding-8B"
API_URL = "https://api.deepinfra.com/v1/openai/embeddings"
DIM = 4096

MAX_BATCH = 64
MAX_RETRIES = 6


def _load_api_key() -> str:
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith("DEEPINFRA_API_KEY"):
                return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"DEEPINFRA_API_KEY not found in {ENV_PATH}")


class _HttpError(Exception):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:300]}")
        self.status = status
        self.text = text


def _call_batch(api_key: str, texts: list[str]) -> list[list[float]]:
    resp = requests.post(
        API_URL,
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
    log(f"[deepinfra/{cache_name}] {len(ids_order) - len(todo)} cached, {len(todo)} to embed")

    api_key = _load_api_key()
    batch_size = MAX_BATCH
    i = 0
    while i < len(todo):
        chunk_ids = todo[i : i + batch_size]
        texts = [items[i_] for i_ in chunk_ids]
        attempt = 0
        while True:
            try:
                vecs = _call_batch(api_key, texts)
                cache.put_batch(chunk_ids, vecs)
                i += len(chunk_ids)
                if (i // batch_size) % 20 == 0 or i >= len(todo):
                    log(f"[deepinfra/{cache_name}] {i}/{len(todo)} embedded")
                break
            except _HttpError as e:
                if e.status == 400 and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    chunk_ids = todo[i : i + batch_size]
                    texts = [items[i_] for i_ in chunk_ids]
                    log(f"[deepinfra/{cache_name}] 400 error, shrinking batch to {batch_size}")
                    continue
                if e.status in (429, 500, 502, 503, 504) and attempt < MAX_RETRIES:
                    wait = min(60, 2**attempt)
                    log(f"[deepinfra/{cache_name}] {e.status}, retry {attempt+1}/{MAX_RETRIES} in {wait}s")
                    time.sleep(wait)
                    attempt += 1
                    continue
                raise

    return cache
