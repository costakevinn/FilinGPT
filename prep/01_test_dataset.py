from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DATASET_DIR = Path("data/training")
SAMPLES = DATASET_DIR / "samples.jsonl"
MANIFEST = DATASET_DIR / "manifest.jsonl"

BEGIN_IN = "<BEGIN_INPUT>"
END_IN = "<END_INPUT>"
BEGIN_OUT = "<BEGIN_OUTPUT>"
END_OUT = "<END_OUTPUT>"

RE_TONE = re.compile(r"(?m)^Tone:\s*(positive|mixed|negative)\s*$")
RE_HIGHLIGHTS = re.compile(r"(?m)^Highlights:\s*$")
RE_RISKS = re.compile(r"(?m)^Risks:\s*$")
RE_BULLET = re.compile(r"(?m)^-\s+.+$")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_text(text: str) -> list[str]:
    errs: list[str] = []

    for tag in (BEGIN_IN, END_IN, BEGIN_OUT, END_OUT):
        if tag not in text:
            errs.append(f"missing {tag}")

    if "###" not in text:
        errs.append("missing ### delimiter")

    if BEGIN_OUT not in text or END_OUT not in text:
        return errs

    out = text.split(BEGIN_OUT, 1)[1].split(END_OUT, 1)[0]

    if "Business:" not in out:
        errs.append("missing Business:")
    if not RE_HIGHLIGHTS.search(out):
        errs.append("missing Highlights:")
    if not RE_RISKS.search(out):
        errs.append("missing Risks:")
    if not RE_TONE.search(out):
        errs.append("invalid or missing Tone")

    bullets = RE_BULLET.findall(out)
    if len(bullets) != 5:
        errs.append(f"expected 5 bullets, got {len(bullets)}")

    if not out.strip().endswith("###"):
        errs.append("output does not end with ###")

    return errs


def _require_file(path: Path, hint: str) -> None:
    if not path.exists():
        print(f"[FAIL] Missing {path.as_posix()} ({hint})")
        sys.exit(1)


def main() -> None:
    _require_file(SAMPLES, "run etl/build_dataset.py")
    _require_file(MANIFEST, "run etl/build_dataset.py")

    rows = load_jsonl(SAMPLES)
    manifest = load_jsonl(MANIFEST)

    if not rows:
        print("[FAIL] samples.jsonl is empty")
        sys.exit(2)

    ids_samples = {r.get("sample_id") for r in rows}
    ids_manifest = {r.get("sample_id") for r in manifest}

    failed = False

    if ids_samples != ids_manifest:
        failed = True
        print("[FAIL] sample_id mismatch between samples and manifest")

    for r in rows:
        sid = r.get("sample_id")
        text = r.get("text", "")
        errs = validate_text(text)
        if errs:
            failed = True
            print(f"\n[FAIL] sample_id={sid}")
            for e in errs:
                print("  -", e)
            print("-- HEAD --")
            print(text[:500].replace("\n", "\\n"))

    if failed:
        print("\n[FAIL] Dataset validation failed.")
        sys.exit(3)

    print(f"[OK] Dataset validated: {len(rows)} samples.")


if __name__ == "__main__":
    main()