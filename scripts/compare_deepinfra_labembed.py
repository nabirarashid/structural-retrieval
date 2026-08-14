"""Compare DeepInfra vs. the lab deployment embeddings of identical Qwen3-Embedding-8B
text -- same model id (Qwen/Qwen3-Embedding-8B, confirmed via each server's /models),
same declared dimensionality (4096). If the two deployments agree, cosine similarity
between corresponding vectors should be ~1.0. Divergence would point to a config
difference (pooling, normalization, quantization) between the two servers."""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

from data import MathNetEasy
from embed_deepinfra import embed_all as embed_deepinfra
from embed_labembed import embed_all as embed_labembed

SEED = 42
N_SAMPLE = 500
RESULTS_DIR = Path("results")

ds = MathNetEasy.load(tier="easy")
all_ids = sorted(ds.corpus.keys())
sample_ids = random.Random(SEED).sample(all_ids, N_SAMPLE)
sample_items = {cid: ds.corpus[cid] for cid in sample_ids}

# DeepInfra: full corpus is already fully cached under "full_corpus" from the
# baseline runs, so this is a cache hit -- no new API calls.
di_cache = embed_deepinfra(ds.corpus, cache_name="full_corpus")
di_matrix = di_cache.get_matrix(sample_ids).astype(np.float64)

# the lab deployment: new cache, only the 500-item sample needs embedding.
mc_cache = embed_labembed(sample_items, cache_name="compare_sample")
mc_matrix = mc_cache.get_matrix(sample_ids).astype(np.float64)

di_norms = np.linalg.norm(di_matrix, axis=1)
mc_norms = np.linalg.norm(mc_matrix, axis=1)

di_unit = di_matrix / di_norms[:, None]
mc_unit = mc_matrix / mc_norms[:, None]
cosines = np.sum(di_unit * mc_unit, axis=1)

result = {
    "n_compared": N_SAMPLE,
    "model": "Qwen/Qwen3-Embedding-8B",
    "declared_dim": 4096,
    "cosine_similarity": {
        "mean": float(cosines.mean()),
        "median": float(np.median(cosines)),
        "std": float(cosines.std()),
        "min": float(cosines.min()),
        "max": float(cosines.max()),
    },
    "pct_below_threshold": {
        "0.999": float((cosines < 0.999).mean() * 100),
        "0.99": float((cosines < 0.99).mean() * 100),
        "0.95": float((cosines < 0.95).mean() * 100),
        "0.90": float((cosines < 0.90).mean() * 100),
    },
    "raw_norm": {
        "deepinfra_mean": float(di_norms.mean()), "deepinfra_std": float(di_norms.std()),
        "labembed_mean": float(mc_norms.mean()), "labembed_std": float(mc_norms.std()),
    },
    "seed": SEED,
}

RESULTS_DIR.mkdir(exist_ok=True)
with open(RESULTS_DIR / "deepinfra_vs_labembed.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
