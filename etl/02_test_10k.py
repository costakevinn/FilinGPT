from __future__ import annotations

import re
import sys
from pathlib import Path

SILVER_DIR = Path("data/silver")

REQUIRED = (
    "apple_2023_10k__10k.txt",
    "amazon_2022_10k__10k.txt",
    "microsoft_2023_10k__10k.txt",
)

RE_PART_I = re.compile(r"(?im)^\s*part\s+i\b")
RE_ITEM1 = re.compile(r"(?im)^\s*item\s+1\b")
RE_BUSINESS = re.compile(r"(?im)\bbusiness\b")
RE_ITEM7 = re.compile(r"(?im)^\s*item\s+7\b")
RE_ITEM8 = re.compile(r"(?im)^\s*item\s+8\b")

RE_XBRL_TOKEN = re.compile(r"(?i)\b(us-gaap|dei|xbrli|iso4217):")


def analyze(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    head = text[:5000]
    xbrl_head_hits = sum(1 for _ in RE_XBRL_TOKEN.finditer(head))

    return {
        "file": path.name,
        "len": len(text),
        "has_part_i": bool(RE_PART_I.search(text)),
        "has_item1": bool(RE_ITEM1.search(text)),
        "has_business_word": bool(RE_BUSINESS.search(text)),
        "has_item7": bool(RE_ITEM7.search(text)),
        "has_item8": bool(RE_ITEM8.search(text)),
        "xbrl_head_hits": xbrl_head_hits,
        "head": text[:320].replace("\n", "\\n"),
    }


def is_invalid(r: dict) -> bool:
    # Minimal narrative checks + guard against XBRL leakage
    return bool(r["len"] < 80_000 or r["xbrl_head_hits"] >= 5 or not r["has_item7"])


def main() -> None:
    missing = [name for name in REQUIRED if not (SILVER_DIR / name).exists()]
    if missing:
        print("[FAIL] Missing silver files:")
        for name in missing:
            print("  -", name)
        sys.exit(1)

    failed = False

    for fname in REQUIRED:
        r = analyze(SILVER_DIR / fname)

        print(f"\n== {r['file']} ==")
        print({k: r[k] for k in r if k not in {"file", "head"}})
        print("-- HEAD --")
        print(r["head"])

        if is_invalid(r):
            failed = True
            print("[WARN] Silver narrative validation failed.")

    if failed:
        print("\n[FAIL] Silver validation failed.")
        sys.exit(2)

    print("\n[OK] Silver layer validated.")


if __name__ == "__main__":
    main()