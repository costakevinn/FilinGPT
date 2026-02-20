# core/infer.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.model import CTX_LEN, VOCAB_SIZE, forward

BOS = 256
EOS = 257

_REQUIRED_KEYS = {"W_embed", "W1", "b1", "W2", "b2"}


def load_model_npz(path: str) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[ERR] Missing model: {p.as_posix()}")

    with np.load(p, allow_pickle=False) as d:
        model = {k: d[k].astype(np.float32, copy=False) for k in d.files}

    missing = _REQUIRED_KEYS.difference(model.keys())
    if missing:
        raise SystemExit(f"[ERR] model missing key(s): {sorted(missing)}")

    return model


def text_to_tokens(text: str) -> list[int]:
    # Byte-level UTF-8 encoding (0..255).
    b = text.replace("\r\n", "\n").replace("\r", "\n").encode("utf-8", errors="replace")
    return list(b)


def tokens_to_text(tokens: list[int]) -> str:
    # Only 0..255 are valid byte tokens.
    bb = bytearray(t for t in tokens if 0 <= t < 256)
    return bb.decode("utf-8", errors="replace")


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    expz = np.exp(z)
    return expz / (expz.sum() + 1e-12)


def _sample_from_probs(probs: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(probs.shape[0], p=probs))


def _top_k_filter(probs: np.ndarray, k: int) -> np.ndarray:
    v = probs.shape[0]
    if k <= 0 or k >= v:
        return probs

    idx = np.argpartition(probs, -k)[-k:]
    out = np.zeros_like(probs)
    out[idx] = probs[idx]

    s = out.sum()
    return probs if s <= 0 else (out / s)


def _next_token(
    model: dict[str, np.ndarray],
    ctx_tokens: list[int],
    temperature: float,
    top_k: int,
    rng: np.random.Generator,
) -> int:
    x = np.array([ctx_tokens], dtype=np.int32)
    logits, _ = forward(model, x)

    # forward returns [B, V]; here B=1
    scores = logits[0].astype(np.float64, copy=False)
    scores[BOS] = -1e9

    if temperature <= 0:
        return int(np.argmax(scores))

    scores = scores / float(temperature)
    probs = _softmax(scores)
    probs = _top_k_filter(probs, top_k)
    return _sample_from_probs(probs, rng)


def generate(
    model: dict[str, np.ndarray],
    prompt: str,
    max_new_tokens: int = 300,
    temperature: float = 0.9,
    top_k: int = 80,
    seed: int = 123,
) -> str:
    rng = np.random.default_rng(seed)

    prompt_tokens = text_to_tokens(prompt)

    ctx: list[int] = [BOS] * CTX_LEN
    for t in prompt_tokens:
        ctx = (ctx + [int(t)])[-CTX_LEN:]

    gen_tokens: list[int] = []
    for _ in range(max_new_tokens):
        nxt = _next_token(model, ctx, temperature=temperature, top_k=top_k, rng=rng)
        if nxt == EOS:
            break
        gen_tokens.append(nxt)
        ctx = (ctx + [nxt])[-CTX_LEN:]

    return tokens_to_text(gen_tokens)