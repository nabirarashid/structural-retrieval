"""Task 3: fetch script for the math-domain retrieval data.

Source: the MathNet-Retrieve BEIR-format split, a SEPARATE Hugging Face
dataset repo (`ShadenA/MathNet-Retrieve`) from the raw MathNet problem
corpus (`ShadenA/MathNet`) -- this was the missing piece flagged in
README.md's "Data" section: the raw corpus alone has no equivalence/
near-miss pairing fields (verified directly against a downloaded sample
row), and re-deriving them would require re-running whatever LLM-
paraphrase pipeline built them, which would NOT reproduce the exact
original text. The pre-built retrieval split resolves this -- found by
checking Hugging Face directly for a repo matching the paper's own
"MathNet-Retrieve" task name, not by guessing at a URL.

Every hash below was computed by diffing a fresh download against this
project's existing data/ directory (the data every math-domain result in
this repo was actually computed on) -- all 9 files matched byte-exact,
verified 2026-08-17. Any future drift in the upstream repo is a loud
failure here, not a silent data change.
"""
import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "ShadenA/MathNet-Retrieve"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

EXPECTED_SHA256 = {
    "easy/corpus.jsonl": "a2de7a1408efe65c8ac064d4e4eb2dbc6c405dbb2c28fccad4126a2d7c03288f",
    "easy/queries.jsonl": "d303c7da9783da8dd441d871f9000ee387fde5982e90c6c701f2f60a93c7b799",
    "easy/qrels/test.tsv": "6b08bd32f008631754bf7532b694201ee37ed06ae50341e3c072a3ad29e868ce",
    "medium/corpus.jsonl": "a2de7a1408efe65c8ac064d4e4eb2dbc6c405dbb2c28fccad4126a2d7c03288f",
    "medium/queries.jsonl": "d303c7da9783da8dd441d871f9000ee387fde5982e90c6c701f2f60a93c7b799",
    "medium/qrels/test.tsv": "48d3f41bcd221358405600a4a7f40c3a10ddc8023965aa484800cdd9da4086d0",
    "hard/corpus.jsonl": "a2de7a1408efe65c8ac064d4e4eb2dbc6c405dbb2c28fccad4126a2d7c03288f",
    "hard/queries.jsonl": "d303c7da9783da8dd441d871f9000ee387fde5982e90c6c701f2f60a93c7b799",
    "hard/qrels/test.tsv": "bb5578c74f81ef84dbfb616a86bc25281c61230c6df634cb04671b1ce3372caf",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    mismatches = []
    for rel_path, expected_hash in EXPECTED_SHA256.items():
        print(f"[fetch-mathnet-retrieve] downloading {rel_path} ...", flush=True)
        downloaded = hf_hub_download(repo_id=REPO_ID, filename=rel_path, repo_type="dataset")
        actual_hash = sha256(Path(downloaded))
        if actual_hash != expected_hash:
            mismatches.append((rel_path, expected_hash, actual_hash))
            print(f"  MISMATCH: expected {expected_hash[:16]}..., got {actual_hash[:16]}...", file=sys.stderr)
            continue

        dest = DATA_DIR / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(downloaded).read_bytes())
        print(f"  OK ({actual_hash[:16]}...) -> {dest.relative_to(DATA_DIR.parent)}", flush=True)

    if mismatches:
        print(f"\n*** {len(mismatches)} file(s) failed SHA256 verification -- "
              f"upstream data has changed since this project's results were computed. "
              f"DO NOT use this data until you've confirmed whether it changes any result. ***",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n[fetch-mathnet-retrieve] all {len(EXPECTED_SHA256)} files verified byte-exact and written to {DATA_DIR}")


if __name__ == "__main__":
    main()
