from __future__ import annotations

import json
from pathlib import Path

DATASET_DIR = Path("data/training")
IN_PATH = DATASET_DIR / "samples.jsonl"
OUT_PATH = DATASET_DIR / "chunks.jsonl"

CHUNK_SIZE = 2048
OVERLAP = 256


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    if OVERLAP >= CHUNK_SIZE:
        raise SystemExit("[ERR] OVERLAP must be smaller than CHUNK_SIZE")
    if not IN_PATH.exists():
        raise SystemExit(f"[ERR] Missing {IN_PATH.as_posix()} (run ml/00_build_dataset.py)")

    rows = load_jsonl(IN_PATH)
    if not rows:
        raise SystemExit("[ERR] samples.jsonl is empty")

    step = CHUNK_SIZE - OVERLAP
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for r in rows:
            sample_id = r.get("sample_id")
            text = r.get("text")
            if not isinstance(sample_id, str) or not isinstance(text, str):
                continue

            text = text.replace("\r\n", "\n").replace("\r", "\n")
            if len(text) < 2:
                continue

            chunk_id = 0
            for start in range(0, len(text), step):
                end = min(start + CHUNK_SIZE, len(text))
                chunk = text[start:end]
                if len(chunk) < 2:
                    break

                out.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "chunk_id": chunk_id,
                            "start": start,
                            "end": end,
                            "text": chunk,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                total += 1
                chunk_id += 1

                if end >= len(text):
                    break

    print(f"[OK] Wrote {total} chunks -> {OUT_PATH.as_posix()}")
    print(f"[OK] chunk_size={CHUNK_SIZE} overlap={OVERLAP} step={step}")


if __name__ == "__main__":
    main()