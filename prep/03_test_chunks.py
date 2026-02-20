from __future__ import annotations

import json
import sys
from pathlib import Path

CHUNKS_PATH = Path("data/training/chunks.jsonl")


def _fail(msg: str, code: int) -> None:
    print(msg)
    sys.exit(code)


def is_valid_row(r: dict) -> bool:
    sample_id = r.get("sample_id")
    chunk_id = r.get("chunk_id")
    start = r.get("start")
    end = r.get("end")
    text = r.get("text")

    if not isinstance(sample_id, str):
        return False
    if not isinstance(chunk_id, int) or chunk_id < 0:
        return False
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        return False
    if not isinstance(text, str) or len(text) < 2:
        return False
    if len(text) != (end - start):
        return False

    return True


def main() -> None:
    if not CHUNKS_PATH.exists():
        _fail(f"[FAIL] Missing {CHUNKS_PATH.as_posix()} (run ml/02_chunk_dataset.py)", 1)

    rows = 0
    bad = 0

    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rows += 1
            r = json.loads(line)
            if not is_valid_row(r):
                bad += 1

    if rows == 0:
        _fail("[FAIL] chunks.jsonl is empty", 2)

    if bad:
        _fail(f"[FAIL] Invalid chunks: {bad}/{rows}", 3)

    print(f"[OK] Chunks validated: {rows} rows.")


if __name__ == "__main__":
    main()