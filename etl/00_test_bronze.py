from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

BRONZE_DIR = Path("data/bronze")

REQUIRED = (
    "apple_2023_10k.txt",
    "amazon_2022_10k.txt",
    "microsoft_2023_10k.txt",
)

RE_DOCUMENT = re.compile(r"(?is)<DOCUMENT>.*?</DOCUMENT>")
RE_TYPE_10K = re.compile(r"(?im)^\s*<TYPE>\s*10-k\b")
RE_ACCESSION = re.compile(r"(?i)\bACCESSION NUMBER:\s*\d{10}-\d{2}-\d{6}\b")
RE_CIK = re.compile(r"(?i)\bCENTRAL INDEX KEY:\s*\d+\b")

MIN_CHARS = 50_000


def preview(text: str, n: int = 300) -> tuple[str, str]:
    head = text[:n].replace("\n", "\\n")
    tail = text[-n:].replace("\n", "\\n")
    return head, tail


def analyze(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="ignore")

    docs = sum(1 for _ in RE_DOCUMENT.finditer(raw))
    return {
        "file": path.name,
        "size": len(raw),
        "documents": docs,
        "has_type_10k": bool(RE_TYPE_10K.search(raw)),
        "has_accession": bool(RE_ACCESSION.search(raw)),
        "has_cik": bool(RE_CIK.search(raw)),
        "too_small": len(raw) < MIN_CHARS,
        "preview": preview(raw),
    }


def is_invalid(r: dict[str, Any]) -> bool:
    return bool(
        r["too_small"]
        or r["documents"] == 0
        or not r["has_accession"]
        or not r["has_cik"]
    )


def main() -> None:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)

    missing = [name for name in REQUIRED if not (BRONZE_DIR / name).exists()]
    if missing:
        print("[FAIL] Missing filings:")
        for name in missing:
            print("  -", name)
        sys.exit(1)

    results = [analyze(BRONZE_DIR / name) for name in REQUIRED]

    failed = False
    for r in results:
        print(f"\n== {r['file']} ==")
        print(f"chars: {r['size']:,}")
        print(f"<DOCUMENT> blocks: {r['documents']}")
        print("has <TYPE>10-K:", r["has_type_10k"])
        print("has ACCESSION:", r["has_accession"])
        print("has CIK:", r["has_cik"])
        print("too small:", r["too_small"])

        if is_invalid(r):
            failed = True
            print("[WARN] Structural validation failed.")

        head, tail = r["preview"]
        print("-- HEAD --")
        print(head)
        print("-- TAIL --")
        print(tail)

    if failed:
        print("\n[FAIL] Bronze validation failed.")
        sys.exit(2)

    print("\n[OK] Bronze layer validated.")


if __name__ == "__main__":
    main()