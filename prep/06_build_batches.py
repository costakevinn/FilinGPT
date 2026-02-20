from __future__ import annotations

import json
from pathlib import Path

TRAIN_DIR = Path("data/training")
IN_PATH = TRAIN_DIR / "tokens.jsonl"
OUT_PATH = TRAIN_DIR / "batches.jsonl"

SEQ_LEN = 256
STRIDE = 256
VOCAB_SIZE = 258


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def is_valid_tokens(tokens: object) -> bool:
    if not isinstance(tokens, list) or len(tokens) < (SEQ_LEN + 2):
        return False
    return all(isinstance(t, int) and 0 <= t < VOCAB_SIZE for t in tokens)


def is_valid_row(row: dict) -> bool:
    if not isinstance(row.get("sample_id"), str):
        return False
    if not isinstance(row.get("chunk_id"), int):
        return False
    return is_valid_tokens(row.get("tokens"))


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"[ERR] Missing {IN_PATH.as_posix()}")

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for row in iter_jsonl(IN_PATH):
            if not is_valid_row(row):
                skipped += 1
                continue

            sample_id = row["sample_id"]
            chunk_id = row["chunk_id"]
            tokens = row["tokens"]

            max_start = len(tokens) - (SEQ_LEN + 1)
            for offset in range(0, max_start + 1, STRIDE):
                x = tokens[offset : offset + SEQ_LEN]
                y = tokens[offset + 1 : offset + SEQ_LEN + 1]

                out.write(
                    json.dumps(
                        {
                            "batch_id": written,
                            "sample_id": sample_id,
                            "chunk_id": chunk_id,
                            "offset": offset,
                            "seq_len": SEQ_LEN,
                            "x": x,
                            "y": y,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1

    print(f"[OK] Wrote {written} sequences -> {OUT_PATH.as_posix()}")
    print(f"[OK] seq_len={SEQ_LEN} stride={STRIDE} vocab_size={VOCAB_SIZE}")
    if skipped:
        print(f"[WARN] Skipped {skipped} token-rows due to schema/quality issues.")


if __name__ == "__main__":
    main()