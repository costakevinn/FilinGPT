# core/train.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.model import CTX_LEN, VOCAB_SIZE, backward, forward, init_model

BOS = 256


def load_batches(path: str) -> tuple[np.ndarray, np.ndarray]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"[ERR] Missing: {p.as_posix()}")

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            r = json.loads(line)
            x = r.get("x")
            y = r.get("y")
            if not isinstance(x, list) or not isinstance(y, list) or len(x) != len(y):
                continue

            xs.append(np.asarray(x, dtype=np.int32))
            ys.append(np.asarray(y, dtype=np.int32))

    if not xs:
        raise SystemExit("[ERR] No sequences found in batches.jsonl")

    X = np.stack(xs, axis=0)
    Y = np.stack(ys, axis=0)
    return X, Y


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    # logits: [B, V], targets: [B]
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    probs = expz / expz.sum(axis=1, keepdims=True)

    b = targets.shape[0]
    loss = -np.log(probs[np.arange(b), targets] + 1e-12).mean()

    probs[np.arange(b), targets] -= 1.0
    probs /= b
    return float(loss), probs.astype(np.float32, copy=False)


def sgd_step(model: dict, grads: dict, lr: float) -> None:
    for k, g in grads.items():
        model[k] -= lr * g


def make_ctx_batch(X: np.ndarray, n_idx: np.ndarray, t_idx: np.ndarray) -> np.ndarray:
    # Builds a [B, CTX_LEN] context window with BOS left-padding.
    b = n_idx.shape[0]
    k = CTX_LEN
    out = np.full((b, k), BOS, dtype=np.int32)

    for i in range(b):
        seq = X[n_idx[i]]
        pos = int(t_idx[i])
        start = max(0, pos - (k - 1))
        ctx = seq[start : pos + 1]
        out[i, k - len(ctx) :] = ctx

    return out


def train_loop(
    model: dict,
    X: np.ndarray,
    Y: np.ndarray,
    steps: int = 50_000,
    lr: float = 0.05,
    seed: int = 42,
    log_every: int = 500,
    batch_size: int = 128,
) -> tuple[dict, dict]:
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape:
        raise SystemExit("[ERR] X/Y must be [N, T] with the same shape")

    n, t = X.shape
    rng = np.random.default_rng(seed)

    losses: list[float] = []

    for step in range(1, steps + 1):
        n_idx = rng.integers(0, n, size=batch_size)
        t_idx = rng.integers(0, t, size=batch_size)

        x_ctx = make_ctx_batch(X, n_idx, t_idx)
        targets = Y[n_idx, t_idx].astype(np.int32, copy=False)

        logits, cache = forward(model, x_ctx)
        loss, dlogits = softmax_cross_entropy(logits, targets)
        grads = backward(model, cache, dlogits)
        sgd_step(model, grads, lr)

        losses.append(loss)

        # percent-based logging
        log_interval = max(1, int(steps * (log_every / 100.0)))

        if step == 1 or step % log_interval == 0 or step == steps:
            pct = (step / steps) * 100.0
            print(f"[{pct:6.2f}%] {step}/{steps} loss={loss:.4f}")

    history = {
        "losses": losses,
        "final_loss": float(losses[-1]) if losses else None,
        "steps": int(steps),
        "lr": float(lr),
        "seed": int(seed),
        "batch_size": int(batch_size),
        "vocab_size": int(VOCAB_SIZE),
        "ctx_len": int(CTX_LEN),
        "n_sequences": int(n),
        "seq_len": int(t),
    }

    return model, history


def train(
    batches_path: str = "data/training/batches.jsonl",
    steps: int = 2000,
    lr: float = 1e-2,
    seed: int = 42,
    log_every: int = 50,
    batch_size: int = 32,
) -> tuple[dict, dict]:
    X, Y = load_batches(batches_path)
    model = init_model(seed=seed)

    model, history = train_loop(
        model=model,
        X=X,
        Y=Y,
        steps=steps,
        lr=lr,
        seed=seed,
        log_every=log_every,
        batch_size=batch_size,
    )

    history["batches_path"] = batches_path
    return model, history