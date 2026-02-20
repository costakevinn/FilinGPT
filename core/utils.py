# core/utils.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_model_npz(model: dict[str, np.ndarray], out_path: str) -> None:
    p = Path(out_path)
    _ensure_parent(p)
    np.savez(p, **model)
    print(f"[OK] Saved model -> {p.as_posix()}")


def save_history_json(history: dict, out_path: str) -> None:
    p = Path(out_path)
    _ensure_parent(p)
    with p.open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[OK] Saved history -> {p.as_posix()}")