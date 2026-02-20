from __future__ import annotations

import html
import re
from pathlib import Path

BRONZE_DIR = Path("data/bronze")
SILVER_DIR = Path("data/silver")

RAW_FILES = (
    "apple_2023_10k.txt",
    "amazon_2022_10k.txt",
    "microsoft_2023_10k.txt",
)

DOC_RE = re.compile(r"(?is)<DOCUMENT>(.*?)</DOCUMENT>")
TYPE_RE = re.compile(r"(?is)<TYPE>\s*([^\s<]+)")
TEXT_RE = re.compile(r"(?is)<TEXT>(.*?)</TEXT>")

TAG_RE = re.compile(r"(?is)<[^>]+>")
SPACE_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")

RE_MDA_TITLE = re.compile(
    r"(?i)management(?:’|&#8217;|')s discussion and analysis of financial condition and results of operations"
)
RE_XBRL = re.compile(r"(?i)\b(us-gaap|dei|xbrli|iso4217|fasb\.org)\b")

RE_PART_I = re.compile(r"(?i)\bpart\s+i\b")
RE_PART_II = re.compile(r"(?i)\bpart\s+ii\b")
RE_ITEM_DOT = re.compile(r"(?i)\bitem\s*(\d+[a-z]?)\s*\.")


def _doc_type(doc: str) -> str:
    m = TYPE_RE.search(doc)
    return (m.group(1) if m else "").strip().lower()


def _doc_text(doc: str) -> str:
    m = TEXT_RE.search(doc)
    return m.group(1) if m else doc


def _inject_section_newlines(s: str) -> str:
    s = RE_PART_I.sub("\nPART I\n", s)
    s = RE_PART_II.sub("\nPART II\n", s)
    s = RE_ITEM_DOT.sub(r"\nItem \1.", s)
    return s


def _clean_text(s: str) -> str:
    s = html.unescape(s)
    s = s.replace("\r", "\n").replace("\xa0", " ")
    s = TAG_RE.sub(" ", s)
    s = _inject_section_newlines(s)
    s = SPACE_RE.sub(" ", s)
    s = MULTI_NL_RE.sub("\n\n", s)
    return s.strip()


def _trim_to_narrative(txt: str) -> str:
    # Cut at the earliest plausible narrative anchor.
    starts: list[int] = []

    m = RE_PART_I.search(txt)
    if m:
        starts.append(m.start())

    m = re.search(r"(?i)\bitem\s*1\s*\.", txt)
    if m:
        starts.append(m.start())

    if not starts:
        return txt.strip()

    return txt[min(starts) :].lstrip()


def _score(txt: str) -> int:
    score = 0

    if RE_MDA_TITLE.search(txt):
        score += 10

    if re.search(r"(?im)^\s*part\s+i\b", txt):
        score += 8
    if re.search(r"(?im)^\s*item\s*1\b", txt):
        score += 8
    if re.search(r"(?im)^\s*item\s*7\b", txt):
        score += 10
    if re.search(r"(?im)^\s*item\s*8\b", txt):
        score += 6

    head = txt[:8000]
    score -= min(200, 5 * len(RE_XBRL.findall(head)))

    if len(txt) < 80_000:
        score -= 10

    return score


def extract_narrative_10k(raw: str) -> str:
    docs = DOC_RE.findall(raw)

    pool = [
        d
        for d in docs
        if "10-k" in _doc_type(d) or _doc_type(d) in {"10k", "10-k405", "10k405"}
    ]
    if not pool:
        pool = docs

    best_txt = ""
    best_score = -10**9

    for d in pool:
        txt = _trim_to_narrative(_clean_text(_doc_text(d)))
        sc = _score(txt)
        if sc > best_score:
            best_score = sc
            best_txt = txt

    if not best_txt:
        return ""
    if not re.search(r"(?im)^\s*item\s*7\b", best_txt):
        return ""
    if not RE_MDA_TITLE.search(best_txt):
        return ""

    return best_txt.strip()


def main() -> None:
    SILVER_DIR.mkdir(parents=True, exist_ok=True)

    for fname in RAW_FILES:
        src = BRONZE_DIR / fname
        if not src.exists():
            print(f"[ERR] missing: {src}")
            continue

        raw = src.read_text(encoding="utf-8", errors="ignore")
        out = extract_narrative_10k(raw)

        if not out:
            print(f"[ERR] {fname}: narrative 10-K not found")
            continue

        out_path = SILVER_DIR / fname.replace(".txt", "__10k.txt")
        out_path.write_text(out, encoding="utf-8")
        print(f"[OK] {fname} -> {out_path} ({len(out):,} chars)")


if __name__ == "__main__":
    main()