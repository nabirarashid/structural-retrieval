"""One-time join: download all per-country MathNet-Solve parquet shards
(ShadenA/MathNet, ~350MB total across ~58 files, small enough to just grab
all of them rather than build a fragile prefix->country-folder name mapping),
then for each of our 100 pilot query problems, find its official solution via
exact/near-exact text match against problem_markdown.

Queries in MathNet-Retrieve are unsuffixed base_ids (unlike corpus items,
which are always LLM-paraphrased eq/nm variants) -- confirmed by a spot check
to be verbatim-or-near-verbatim original MathNet-Solve problems, which is why
a plain text join is reliable here specifically (it would NOT be reliable for
corpus items, which are deliberately paraphrased, more heavily so at hard
tier -- that's why we're not attempting solutions for retrieved candidates,
only for the 100 query problems we need to grade against).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

sys.path.insert(0, "src")
from data import MathNetEasy

CACHE_DIR = Path("data/.hf_cache/mathnet_solve")  # local HF download cache, gitignored under data/
OUT_PATH = Path("data/solve_solution_index.json")

N_PILOT = 250
SEED = 42

ds = MathNetEasy.load(tier="hard")
query_ids = ds.sample_queries(500, seed=SEED)[:N_PILOT]
query_texts = {qid: ds.queries[qid] for qid in query_ids}

print(f"Need solutions for {len(query_ids)} query problems", flush=True)

api = HfApi()
info = api.dataset_info("ShadenA/MathNet", files_metadata=False)
country_files = [
    s.rfilename for s in info.siblings
    if s.rfilename.startswith("data/") and not s.rfilename.startswith("data/all/")
    and s.rfilename.endswith(".parquet")
]
print(f"Downloading {len(country_files)} per-country parquet files (~350MB total)...", flush=True)

frames = []
for i, fn in enumerate(country_files, 1):
    path = hf_hub_download(repo_id="ShadenA/MathNet", filename=fn, repo_type="dataset", cache_dir=str(CACHE_DIR))
    df = pd.read_parquet(path, columns=["id", "country", "competition", "problem_markdown", "solutions_markdown"])
    frames.append(df)
    if i % 10 == 0 or i == len(country_files):
        print(f"  {i}/{len(country_files)} downloaded", flush=True)

solve_df = pd.concat(frames, ignore_index=True)
print(f"MathNet-Solve rows loaded: {len(solve_df)}", flush=True)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


solve_df["_norm"] = solve_df["problem_markdown"].map(normalize)

results = {}
n_exact, n_norm, n_miss = 0, 0, 0
for qid, qtext in query_texts.items():
    qtext_norm = normalize(qtext)
    exact_hits = solve_df[solve_df["problem_markdown"].str.contains(re.escape(qtext), regex=True, na=False)]
    if len(exact_hits) == 1:
        row = exact_hits.iloc[0]
        n_exact += 1
    else:
        norm_hits = solve_df[solve_df["_norm"].str.contains(re.escape(qtext_norm), regex=True, na=False)]
        if len(norm_hits) == 1:
            row = norm_hits.iloc[0]
            n_norm += 1
        elif len(exact_hits) > 1 or len(norm_hits) > 1:
            # ambiguous -- take first but flag
            row = (exact_hits if len(exact_hits) else norm_hits).iloc[0]
            n_norm += 1
        else:
            n_miss += 1
            continue

    solutions = list(row["solutions_markdown"]) if row["solutions_markdown"] is not None else []
    results[qid] = {
        "solve_id": row["id"], "country": row["country"], "competition": row["competition"],
        "solutions_markdown": solutions,
    }

print(f"\nMatched: {len(results)}/{N_PILOT}  (exact={n_exact}, normalized/ambiguous={n_norm}, miss={n_miss})")
if n_miss:
    missed = [qid for qid in query_ids if qid not in results]
    print("Unmatched query IDs:", missed)

Path("data").mkdir(exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved to {OUT_PATH}")
