"""Disk-backed (memmap) embedding cache.

The naive approach -- JSONL with the whole cache loaded into a Python dict of
float lists to check what's done -- materializes gigabytes of Python float
objects in RAM for a corpus this size (117k items x 3-4k dims). This version
keeps only a small id->row-index manifest in memory; the actual vectors live
in a memory-mapped .npy-style binary file and are never fully loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class VectorCache:
    def __init__(self, cache_dir: Path, name: str, dim: int, capacity: int):
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.dir / f"{name}.manifest.json"
        self.vectors_path = self.dir / f"{name}.vectors.dat"
        self.dim = dim
        self.capacity = capacity

        self.manifest: dict[str, int] = {}
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)

        mode = "r+" if self.vectors_path.exists() else "w+"
        self._mm = np.memmap(self.vectors_path, dtype=np.float32, mode=mode, shape=(capacity, dim))

    def has(self, id_: str) -> bool:
        return id_ in self.manifest

    def missing(self, ids: list[str]) -> list[str]:
        return [i for i in ids if i not in self.manifest]

    def put_batch(self, ids: list[str], vectors: list[list[float]]) -> None:
        for id_, vec in zip(ids, vectors):
            if id_ in self.manifest:
                continue
            row = len(self.manifest)
            if row >= self.capacity:
                raise RuntimeError(f"cache capacity {self.capacity} exceeded")
            self._mm[row] = np.asarray(vec, dtype=np.float32)
            self.manifest[id_] = row
        self._mm.flush()
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f)

    def get_matrix(self, ids: list[str]) -> np.ndarray:
        """Rows for the given ids, in that order, as a fresh in-memory array
        (only as large as `ids`, not the whole cache)."""
        idx = [self.manifest[i] for i in ids]
        return np.asarray(self._mm[idx])
