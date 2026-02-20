from __future__ import annotations

import re
from pathlib import Path

IN_DIR = Path("data/silver")
OUT_DIR = Path("data/gold")

SEP = r"(?:\s|&#160;|&nbsp;|\xa0)+"

# Line-anchored headings
ITEM7_HEADING = re.compile(rf"(?im)^[ \t]*item{SEP}?7\b[ \t]*[.\-:]?")
ITEM7A_HEADING = re.compile(rf"(?im)^[ \t]*item{SEP}?7a\b[ \t]*[.\-:]?")
ITEM8_HEADING = re.compile(rf"(?im)^[ \t]*item{SEP}?8\b[ \t]*[.\-:]?")

MDA_TITLE = re.compile(
    r"management(?:&#8217;|’|')s discussion and analysis of financial condition and results of operations",
    re.IGNORECASE,
)

RE_ITEM_WORD = re.compile(r"\bitem\b", re.IGNORECASE)


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    text = text.replace("&nbsp;", " ").replace("&#160;", " ")
    return text


def is_toc_like(txt: str, pos: int) -> bool:
    after = txt[pos : pos + 1800].lower()
    if "table of contents" in after:
        return True
    return sum(1 for _ in RE_ITEM_WORD.finditer(after)) >= 8


def pick_start_via_title(txt: str) -> int | None:
    best: tuple[int, int] | None = None  # (score, start)

    for t in MDA_TITLE.finditer(txt):
        lookback_start = max(0, t.start() - 1200)
        before = txt[lookback_start : t.start()]

        last_heading = None
        for m in ITEM7_HEADING.finditer(before):
            last_heading = m
        if not last_heading:
            continue

        start = lookback_start + last_heading.start()
        after = txt[start : start + 5000].lower()

        score = 0
        if is_toc_like(txt, start):
            score -= 10
        if "forward-looking statements" in after:
            score += 8
        if MDA_TITLE.search(txt[start : start + 2000]):
            score += 10

        if best is None or score > best[0]:
            best = (score, start)

    return best[1] if best and best[0] >= 0 else None


def pick_start_fallback(txt: str) -> int | None:
    best: tuple[int, int] | None = None  # (score, pos)

    for m in ITEM7_HEADING.finditer(txt):
        pos = m.start()
        after = txt[pos : pos + 4000].lower()

        score = 0
        if is_toc_like(txt, pos):
            score -= 10
        if MDA_TITLE.search(after):
            score += 10
        if "forward-looking statements" in after:
            score += 8
        if "this annual report on form 10-k" in after:
            score += 3

        if best is None or score > best[0]:
            best = (score, pos)

    return best[1] if best and best[0] >= 0 else None


def pick_end(txt: str, start: int) -> int | None:
    m7a = ITEM7A_HEADING.search(txt, start + 1)
    m8 = ITEM8_HEADING.search(txt, start + 1)

    if m7a and m8:
        return min(m7a.start(), m8.start())
    if m7a:
        return m7a.start()
    if m8:
        return m8.start()
    return None


def extract_mda(txt: str) -> str | None:
    txt = normalize(txt)

    start = pick_start_via_title(txt) or pick_start_fallback(txt)
    if start is None:
        return None

    end = pick_end(txt, start)
    if end is None or end <= start:
        return None

    chunk = txt[start:end].strip()
    if len(chunk) < 5000:
        return None

    if not MDA_TITLE.search(chunk[:8000]):
        return None

    return chunk


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*__10k.txt"))
    if not files:
        raise SystemExit(f"[ERR] No files found in {IN_DIR} (expected *__10k.txt)")

    for src in files:
        raw = src.read_text(encoding="utf-8", errors="ignore")

        out_name = src.name.replace("__10k.txt", "__mda.txt")
        out_path = OUT_DIR / out_name

        print(f"Processing {src.name}...")
        mda = extract_mda(raw)

        if not mda:
            if out_path.exists():
                out_path.unlink()
                print(f"  -> MD&A not found. Removed stale {out_path.as_posix()}")
            else:
                print("  -> MD&A not found.")
            continue

        out_path.write_text(mda, encoding="utf-8")
        print(f"  -> Saved {out_path.as_posix()} ({len(mda):,} chars)")

    print("MD&A extraction complete.")


if __name__ == "__main__":
    main()