"""Task 3: fetch script for the trajectory-domain corpus.

Source: github.com/qpiai/Proced_mem_bench (Ishant and Krishnan, 2025,
arXiv:2511.21730). Clones into /tmp/proced_mem_bench_check, the path every
trajectory-domain script in this repo already expects -- deliberately
kept as-is rather than relocated into the repo, since a broad path change
touches ~15 scripts for a benefit (surviving a /tmp cleanup) this fetch
script itself already provides by being trivially re-runnable.

Why this matters: this exact scratch directory went missing mid-project
(a routine /tmp cleanup between sessions cleared it, including the
cloned repo's own .git/config -- see the 2026-08-17 entry in
JOURNEY_LOG.md) with no fetch script to restore it. This script is that
fix. Pinned to the exact commit this project's trajectory-domain results
were computed against; verified by SHA256 + count on the two files every
downstream script reads.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/qpiai/Proced_mem_bench.git"
PINNED_COMMIT = "0804875216391ae98aae8b8109657fd404faf61f"  # HEAD as of 2026-03-18, verified against this project's results
TARGET_DIR = Path("/tmp/proced_mem_bench_check")

TRAJECTORIES_PATH = "procedural_memory_benchmark/data/corpus/agentinstruct_trajectories.json"
QUERY_BANK_PATH = "procedural_memory_benchmark/benchmark/data/query_bank.json"

EXPECTED = {
    TRAJECTORIES_PATH: {
        "sha256": "4aee80a266220cd8b7a17bc6341345589536182f263c56b86e27008d1876243c",
        "count_field": "trajectories", "expected_count": 336,
    },
    QUERY_BANK_PATH: {
        "sha256": "1ca7dcca6cbf778f78175f208c9f4e55a5b80ca57896055fd2a6ec8fa1c02d37",
        "count_field": "queries", "expected_count": 40,
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def main():
    if TARGET_DIR.exists():
        print(f"[fetch-proced-mem-bench] {TARGET_DIR} already exists -- removing before re-clone "
              f"(this is scratch data, not the repo's own working tree)", flush=True)
        shutil.rmtree(TARGET_DIR)

    print(f"[fetch-proced-mem-bench] cloning {REPO_URL} -> {TARGET_DIR} ...", flush=True)
    subprocess.run(["git", "clone", "-q", REPO_URL, str(TARGET_DIR)], check=True)
    subprocess.run(["git", "-C", str(TARGET_DIR), "checkout", "-q", PINNED_COMMIT], check=True)
    actual_commit = subprocess.run(
        ["git", "-C", str(TARGET_DIR), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if actual_commit != PINNED_COMMIT:
        print(f"*** commit mismatch after checkout: expected {PINNED_COMMIT}, got {actual_commit} ***", file=sys.stderr)
        sys.exit(1)
    print(f"[fetch-proced-mem-bench] checked out pinned commit {PINNED_COMMIT}", flush=True)

    mismatches = []
    for rel_path, spec in EXPECTED.items():
        full_path = TARGET_DIR / rel_path
        if not full_path.exists():
            mismatches.append((rel_path, "FILE MISSING", ""))
            print(f"  MISSING: {rel_path}", file=sys.stderr)
            continue
        actual_hash = sha256(full_path)
        data = json.load(open(full_path))
        actual_count = len(data[spec["count_field"]])
        ok = actual_hash == spec["sha256"] and actual_count == spec["expected_count"]
        if not ok:
            mismatches.append((rel_path, spec["sha256"][:16], actual_hash[:16]))
            print(f"  MISMATCH {rel_path}: sha256 expected {spec['sha256'][:16]}... got {actual_hash[:16]}..., "
                  f"count expected {spec['expected_count']} got {actual_count}", file=sys.stderr)
        else:
            print(f"  OK {rel_path}: sha256={actual_hash[:16]}... count={actual_count}", flush=True)

    if mismatches:
        print(f"\n*** {len(mismatches)} file(s) failed verification at the pinned commit -- "
              f"this should not happen (commit is pinned); investigate before trusting this data. ***",
              file=sys.stderr)
        sys.exit(1)

    print(f"\n[fetch-proced-mem-bench] all files verified. Corpus ready at {TARGET_DIR}")


if __name__ == "__main__":
    main()
