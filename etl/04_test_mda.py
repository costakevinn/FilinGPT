from __future__ import annotations

import re
import sys
from pathlib import Path

GOLD_DIR = Path("data/gold")

MIN_LEN = 5000
TITLE_SCAN = 12_000
HEAD_PREVIEW = 420

TITLE = re.compile(
    r"management(?:&#8217;|’|')s discussion and analysis of financial condition and results of operations",
    re.IGNORECASE,
)

SEP = r"(?:\s|&#160;|&nbsp;|\xa0)+"
ITEM7A_HEADING = re.compile(rf"(?im)^[ \t]*item{SEP}?7a\b[ \t]*[.\-:]?")
ITEM8_HEADING = re.compile(rf"(?im)^[ \t]*item{SEP}?8\b[ \t]*[.\-:]?")


def summarize_checks(txt: str) -> dict:
    return {
        "len": len(txt),
        "has_title": bool(TITLE.search(txt[:TITLE_SCAN])),
        "has_item7a_heading_inside": bool(ITEM7A_HEADING.search(txt)),
        "has_item8_heading_inside": bool(ITEM8_HEADING.search(txt)),
    }


def is_invalid(info: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    if info["len"] < MIN_LEN or not info["has_title"]:
        reasons.append("Too small or missing MD&A title near the start.")

    if info["has_item7a_heading_inside"] or info["has_item8_heading_inside"]:
        reasons.append("Contains Item 7A or Item 8 heading inside MD&A (bad end boundary).")

    return (len(reasons) > 0, reasons)


def main() -> None:
    files = sorted(GOLD_DIR.glob("*__mda.txt"))
    if not files:
        print(f"[FAIL] No MD&A files found in {GOLD_DIR} (expected *__mda.txt)")
        sys.exit(1)

    any_fail = False

    for p in files:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        info = summarize_checks(txt)

        print(f"\n== {p.name} ==")
        print(info)
        print("-- HEAD --")
        head = (txt[:HEAD_PREVIEW] + "...") if len(txt) > HEAD_PREVIEW else txt
        print(head.replace("\n", "\\n"))

        bad, reasons = is_invalid(info)
        if bad:
            any_fail = True
            for msg in reasons:
                print("[FAIL]", msg)

    if any_fail:
        print("\n[FAIL] MD&A validation failed.")
        sys.exit(2)

    print("\n[OK] MD&A validation passed.")


if __name__ == "__main__":
    main()