from __future__ import annotations

import json
from pathlib import Path

IN_PATH = Path("data/training/chunks.jsonl")
OUT_PATH = Path("data/training/tokens.jsonl")

BOS = 256
EOS = 257
VOCAB_SIZE = 258


def encode(text: str) -> list[int]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    b = text.encode("utf-8", errors="replace")
    return [BOS, *b, EOS]


def is_valid_row(r: dict) -> bool:
    sample_id = r.get("sample_id")
    chunk_id = r.get("chunk_id")
    start = r.get("start")
    end = r.get("end")
    text = r.get("text")

    if not isinstance(sample_id, str) or not isinstance(chunk_id, int):
        return False
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        return False
    if not isinstance(text, str) or len(text) < 2:
        return False

    return True


def main() -> None:
    if not IN_PATH.exists():
        raise SystemExit(f"[ERR] Missing {IN_PATH.as_posix()}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0

    with IN_PATH.open("r", encoding="utf-8") as fin, OUT_PATH.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            r = json.loads(line)
            if not is_valid_row(r):
                skipped += 1
                continue

            text = r["text"]
            tokens = encode(text)

            fout.write(
                json.dumps(
                    {
                        "sample_id": r["sample_id"],
                        "chunk_id": r["chunk_id"],
                        "start": r["start"],
                        "end": r["end"],
                        "n_chars": len(text),
                        "n_tokens": len(tokens),
                        "bos": BOS,
                        "eos": EOS,
                        "vocab_size": VOCAB_SIZE,
                        "tokens": tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"[OK] Wrote {written} tokenized chunks -> {OUT_PATH.as_posix()}")
    print(f"[OK] vocab_size={VOCAB_SIZE} bos={BOS} eos={EOS}")
    if skipped:
        print(f"[WARN] Skipped {skipped} rows due to schema issues.")


if __name__ == "__main__":
    main()