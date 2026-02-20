from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

GOLD_DIR = Path("data/gold")
OUT_DIR = Path("data/training")

BEGIN_IN = "<BEGIN_INPUT>"
END_IN = "<END_INPUT>"
BEGIN_OUT = "<BEGIN_OUTPUT>"
END_OUT = "<END_OUTPUT>"

FIN_TERMS = (
    "revenue", "net sales", "sales", "income", "operating income", "operating loss",
    "margin", "gross margin", "eps", "earnings", "cash flow", "free cash flow",
    "guidance", "outlook", "demand", "cost", "expenses", "inflation", "interest rate",
    "foreign exchange", "fx", "currency", "capex", "capital expenditures",
)

RISK_TERMS = (
    "risk", "uncertaint", "headwind", "challenge", "pressure", "litigation",
    "regulator", "compliance", "security", "cyber", "macroeconomic",
    "recession", "inflation", "interest rate", "foreign exchange", "supply chain",
)

POS_WORDS = (
    "increase", "increased", "growth", "grew", "improved", "improvement",
    "strong", "strength", "record", "resilient", "positive", "benefit",
)

NEG_WORDS = (
    "decrease", "decreased", "decline", "declined", "lower", "reduced", "reduce",
    "weak", "weaker", "negative", "loss", "impairment", "material weakness",
    "adverse", "uncertain", "pressure", "headwind",
)

WS = re.compile(r"\s+")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\(\[])")  # good-enough splitter
NUM_RE = re.compile(r"(?<!\w)(\$?\d[\d,]*\.?\d*%?)(?!\w)")
WORD_RE = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class Sample:
    sample_id: str
    source_file: str
    md5: str
    input_text: str
    output_text: str
    combined: str


def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8", errors="ignore")).hexdigest()


def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    s = WS.sub(" ", s)
    s = s.replace(" \n ", "\n").replace(" \n", "\n").replace("\n ", "\n")
    return s.strip()


def to_sentences(s: str) -> list[str]:
    s = WS.sub(" ", s.replace("\n", " ")).strip()
    if not s:
        return []

    parts = [p.strip() for p in SENT_SPLIT.split(s) if p.strip()]
    if len(parts) < 4:
        parts = [p.strip() for p in re.split(r"\.\s+", s) if p.strip()]
        parts = [p + ("" if p.endswith((".", "!", "?")) else ".") for p in parts]

    return [p for p in parts if len(p) >= 20]


def score_highlight(sent: str) -> int:
    s = sent.lower()
    score = 3 * len(NUM_RE.findall(sent))

    for t in FIN_TERMS:
        if t in s:
            score += 2

    if "due to" in s or "because" in s or "driven by" in s:
        score += 2

    if "forward-looking" in s:
        score -= 5

    return score


def score_risk(sent: str) -> int:
    s = sent.lower()
    score = 0

    for t in RISK_TERMS:
        if t in s:
            score += 2

    if "may" in s or "could" in s or "might" in s:
        score += 1

    if "forward-looking" in s:
        score -= 2

    return score


def pick_business(sents: list[str]) -> str:
    chosen: list[str] = []
    for sent in sents[:12]:
        if "forward-looking" in sent.lower():
            continue
        chosen.append(sent)
        if len(chosen) == 2:
            break

    if not chosen and sents:
        chosen = [sents[0]]

    return " ".join(chosen).strip()


def pick_top(sentences: list[str], score_fn: Callable[[str], int], k: int) -> list[str]:
    scored = [(score_fn(s), i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    picked: list[str] = []
    seen: set[str] = set()

    for sc, _, s in scored:
        if sc <= 0:
            continue
        key = s.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        picked.append(s)
        if len(picked) == k:
            break

    if len(picked) < k:
        for s in sentences:
            if s not in picked:
                picked.append(s)
                if len(picked) == k:
                    break

    return picked[:k]


def infer_tone(text: str) -> str:
    words = [w.lower() for w in WORD_RE.findall(text)]
    pos = sum(1 for w in words if w in POS_WORDS)
    neg = sum(1 for w in words if w in NEG_WORDS)

    if neg >= pos + 3:
        return "negative"
    if pos >= neg + 3:
        return "positive"
    return "mixed"


def format_output(business: str, highlights: list[str], risks: list[str], tone: str) -> str:
    lines: list[str] = []

    lines.append("Business:")
    lines.append(business.strip())
    lines.append("")

    lines.append("Highlights:")
    for h in highlights[:3]:
        lines.append(f"- {h.strip()}")
    lines.append("")

    lines.append("Risks:")
    for r in risks[:2]:
        lines.append(f"- {r.strip()}")
    lines.append("")

    lines.append(f"Tone: {tone}")
    lines.append("###")

    return "\n".join(lines).strip() + "\n"


def build_sample(src_path: Path) -> Sample | None:
    raw = src_path.read_text(encoding="utf-8", errors="ignore")
    raw = clean_text(raw)
    if len(raw) < 5_000:
        return None

    sents = to_sentences(raw)
    if len(sents) < 5:
        return None

    business = pick_business(sents)
    highlights = pick_top(sents, score_highlight, 3)
    risks = pick_top(sents, score_risk, 2)
    tone = infer_tone(raw)

    inp = "\n".join([BEGIN_IN, "MD&A:", raw, END_IN]).strip() + "\n"
    out = "\n".join([BEGIN_OUT, format_output(business, highlights, risks, tone), END_OUT]).strip() + "\n"

    combined = inp + out
    sample_id = md5_text(src_path.name + "\n" + md5_text(raw))[:12]

    return Sample(
        sample_id=sample_id,
        source_file=src_path.name,
        md5=md5_text(raw),
        input_text=inp,
        output_text=out,
        combined=combined,
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gold_files = sorted(GOLD_DIR.glob("*__mda.txt"))
    if not gold_files:
        raise SystemExit(f"[ERR] No files found in {GOLD_DIR} (expected *__mda.txt)")

    samples: list[Sample] = []
    for p in gold_files:
        s = build_sample(p)
        if not s:
            print(f"[SKIP] {p.name} (too short / not enough sentences)")
            continue
        samples.append(s)
        print(f"[OK] {p.name} -> sample_id={s.sample_id}")

    if not samples:
        raise SystemExit("[ERR] No samples built.")

    manifest_path = OUT_DIR / "manifest.jsonl"
    samples_path = OUT_DIR / "samples.jsonl"

    _write_jsonl(
        manifest_path,
        [{"sample_id": s.sample_id, "source_file": s.source_file, "md5": s.md5} for s in samples],
    )
    _write_jsonl(
        samples_path,
        [{"sample_id": s.sample_id, "text": s.combined} for s in samples],
    )

    print(f"[OK] Wrote {len(samples)} samples -> {samples_path.as_posix()}")
    print(f"[OK] Wrote manifest -> {manifest_path.as_posix()}")


if __name__ == "__main__":
    main()