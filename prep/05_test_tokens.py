from __future__ import annotations

import json
import sys
from pathlib import Path

TOKENS_PATH = Path("data/training/tokens.jsonl")

BOS = 256
EOS = 257
VOCAB_SIZE = 258
MIN_LEN = 3  # BOS + at least 1 byte + EOS


def _fail(msg: str, code: int) -> None:
    print(msg)
    sys.exit(code)


def check_row(r: dict) -> list[str]:
    errs: list[str] = []

    sid = r.get("sample_id")
    cid = r.get("chunk_id")
    toks = r.get("tokens")

    if not isinstance(sid, str) or not isinstance(cid, int):
        return ["Missing/invalid sample_id or chunk_id"]

    if not isinstance(toks, list):
        return [f"tokens is not a list (chunk_id={cid})"]

    if len(toks) < MIN_LEN:
        return [f"tokens too short (len={len(toks)}) (chunk_id={cid})"]

    if toks[0] != BOS or toks[-1] != EOS:
        errs.append(f"Missing BOS/EOS (chunk_id={cid})")

    if any((not isinstance(t, int)) or t < 0 or t >= VOCAB_SIZE for t in toks):
        errs.append(f"Token out of range (chunk_id={cid})")

    return errs


def main() -> None:
    if not TOKENS_PATH.exists():
        _fail(f"[FAIL] Missing: {TOKENS_PATH.as_posix()}", 1)

    rows = 0
    failed = False

    with TOKENS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            rows += 1
            r = json.loads(line)
            errs = check_row(r)

            if errs:
                failed = True
                for e in errs:
                    print("[FAIL]", e)

    if rows == 0:
        _fail("[FAIL] No rows found in tokens.jsonl", 2)

    if failed:
        _fail(f"\n[FAIL] Token validation failed ({rows} rows checked).", 3)

    print(f"[OK] Tokens validated: {rows} rows.")


if __name__ == "__main__":
    main()