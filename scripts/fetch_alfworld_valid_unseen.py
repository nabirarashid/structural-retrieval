"""Task 3: fetch script for the 78-new-query ALFWorld valid_unseen source.

Source: hkust-nlp/agentboard, a Hugging Face dataset repo, file
data/alfworld/test.jsonl -- the "lightest public source found" for
ALFWorld's valid_unseen split, per scripts/task1_expand_queries.py's own
docstring (which explains the shortfall against the ~150-new-query target:
only 78 unique goal strings exist in this file across 134 episodes, and
pulling in the full `alfworld` package for a `valid_seen` equivalent was
not pursued -- it drags in a heavy simulation stack for what's only
needed as text).

Written to /tmp/alfworld_test.jsonl, the exact path
scripts/task1_expand_queries.py already reads. Verified against this
project's own stored task1_new_query_labels.json: all 78 stored query
texts must appear in the fetched file's goal strings (confirmed
2026-08-17 -- 78/78 exact match), not just a count/hash match, since this
file's downstream use is text content, not byte-identity of the whole
file (episode ordering/fields beyond `goal` are not load-bearing here).
"""
import hashlib
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "hkust-nlp/agentboard"
FILENAME = "data/alfworld/test.jsonl"
DEST = Path("/tmp/alfworld_test.jsonl")
STORED_LABELS_PATH = Path(__file__).resolve().parent.parent / "results" / "task1_new_query_labels.json"

EXPECTED_SHA256 = "5ebe0ca46a1f82bab31c6dd34dd1d5a9b6ca1c375d6a3c7361c1d100c9aac53b"
EXPECTED_EPISODES = 134
EXPECTED_UNIQUE_GOALS = 78


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    print(f"[fetch-alfworld] downloading {REPO_ID}/{FILENAME} ...", flush=True)
    downloaded = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset"))
    actual_hash = sha256(downloaded)
    if actual_hash != EXPECTED_SHA256:
        print(f"*** SHA256 mismatch: expected {EXPECTED_SHA256[:16]}..., got {actual_hash[:16]}... "
              f"-- upstream file has changed. Stopping before writing or verifying further. ***", file=sys.stderr)
        sys.exit(1)

    lines = [json.loads(l) for l in open(downloaded)]
    goals = set(l["goal"] for l in lines)
    if len(lines) != EXPECTED_EPISODES:
        print(f"*** episode count mismatch: expected {EXPECTED_EPISODES}, got {len(lines)} ***", file=sys.stderr)
        sys.exit(1)
    if len(goals) != EXPECTED_UNIQUE_GOALS:
        print(f"*** unique-goal count mismatch: expected {EXPECTED_UNIQUE_GOALS}, got {len(goals)} ***", file=sys.stderr)
        sys.exit(1)
    print(f"[fetch-alfworld] verified: {len(lines)} episodes, {len(goals)} unique goal strings", flush=True)

    if STORED_LABELS_PATH.exists():
        stored = json.load(open(STORED_LABELS_PATH))
        stored_texts = set(e["query_text"] for e in stored["new_labeled_queries"])
        overlap = stored_texts & goals
        if len(overlap) != len(stored_texts):
            missing = stored_texts - goals
            print(f"*** {len(missing)}/{len(stored_texts)} stored query texts not found in fetched goals -- "
                  f"this would silently change which queries the trajectory domain's new_78 subset contains. "
                  f"Sample missing: {list(missing)[:3]} ***", file=sys.stderr)
            sys.exit(1)
        print(f"[fetch-alfworld] cross-checked against {STORED_LABELS_PATH.name}: "
              f"{len(overlap)}/{len(stored_texts)} stored query texts confirmed present", flush=True)
    else:
        print(f"[fetch-alfworld] {STORED_LABELS_PATH} not found -- skipping content cross-check "
              f"(hash + count verification above still passed)", flush=True)

    DEST.write_bytes(downloaded.read_bytes())
    print(f"\n[fetch-alfworld] written to {DEST}")


if __name__ == "__main__":
    main()
