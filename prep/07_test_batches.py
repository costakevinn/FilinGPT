from __future__ import annotations

import json
import sys
from pathlib import Path

BATCHES_PATH = Path("data/training/batches.jsonl")

SEQ_LEN = 256
VOCAB_SIZE = 258


def _fail(msg: str, code: int) -> None:
    print(msg)
    sys.exit(code)


def is_valid_row(r: dict) -> bool:
    x = r.get("x")
    y = r.get("y")

    if not isinstance(r.get("batch_id"), int):
        return False
    if not isinstance(r.get("sample_id"), str):
        return False
    if not isinstance(r.get("chunk_id"), int):
        return False

    offset = r.get("offset")
    if not isinstance(offset, int) or offset < 0:
        return False

    if not isinstance(x, list) or not isinstance(y, list):
        return False
    if len(x) != SEQ_LEN or len(y) != SEQ_LEN:
        return False

    if any((not isinstance(t, int) or t < 0 or t >= VOCAB_SIZE) for t in x):
        return False
    if any((not isinstance(t, int) or t < 0 or t >= VOCAB_SIZE) for t in y):
        return False

    return True


def main() -> None:
    if not BATCHES_PATH.exists():
        _fail(f"[FAIL] Missing {BATCHES_PATH.as_posix()} (run ml/06_build_batches.py)", 1)

    rows = 0
    bad = 0

    with BATCHES_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rows += 1
            r = json.loads(line)
            if not is_valid_row(r):
                bad += 1

    if rows == 0:
        _fail("[FAIL] batches.jsonl is empty", 2)

    if bad:
        _fail(f"[FAIL] Invalid batch rows: {bad}/{rows}", 3)

    print(f"[OK] Batches validated: {rows} rows. seq_len={SEQ_LEN} vocab_size={VOCAB_SIZE}")


if __name__ == "__main__":
    main()