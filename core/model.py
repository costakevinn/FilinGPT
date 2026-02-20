# core/model.py
from __future__ import annotations

import numpy as np

VOCAB_SIZE = 258
EMBED_DIM = 64
HIDDEN_DIM = 128
CTX_LEN = 16


def init_model(seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    in_dim = CTX_LEN * EMBED_DIM
    w1 = rng.normal(0.0, 1.0, size=(in_dim, HIDDEN_DIM)) * np.sqrt(2.0 / in_dim)
    w2 = rng.normal(0.0, 1.0, size=(HIDDEN_DIM, VOCAB_SIZE)) * np.sqrt(2.0 / HIDDEN_DIM)

    return {
        "W_embed": rng.normal(0.0, 0.01, size=(VOCAB_SIZE, EMBED_DIM)).astype(np.float32),
        "W1": w1.astype(np.float32),
        "b1": np.zeros((HIDDEN_DIM,), dtype=np.float32),
        "W2": w2.astype(np.float32),
        "b2": np.zeros((VOCAB_SIZE,), dtype=np.float32),
    }


def relu_forward(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.maximum(0, x)
    return out, x


def relu_backward(dout: np.ndarray, x: np.ndarray) -> np.ndarray:
    return dout * (x > 0)


def linear_forward(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    out = x @ w + b
    return out, (x, w)


def linear_backward(dout: np.ndarray, cache: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, w = cache
    dx = dout @ w.T
    dw = x.T @ dout
    db = dout.sum(axis=0)
    return dx, dw, db


def embed_forward(w_embed: np.ndarray, token_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # token_ids: [B, K]
    token_ids = token_ids.astype(np.int32, copy=False)
    e = w_embed[token_ids]  # [B, K, D]
    return e, token_ids


def embed_backward(de: np.ndarray, token_ids: np.ndarray, vocab_size: int) -> np.ndarray:
    # de: [B, K, D]
    dw = np.zeros((vocab_size, de.shape[-1]), dtype=de.dtype)
    np.add.at(dw, token_ids.reshape(-1), de.reshape(-1, de.shape[-1]))
    return dw


def forward(model: dict[str, np.ndarray], token_ids: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    # token_ids: [B, K]
    e, emb_ids = embed_forward(model["W_embed"], token_ids)
    b, k, d = e.shape
    x = e.reshape(b, k * d)

    h_pre, l1 = linear_forward(x, model["W1"], model["b1"])
    h, relu_x = relu_forward(h_pre)
    logits, l2 = linear_forward(h, model["W2"], model["b2"])

    cache: dict[str, object] = {
        "emb_ids": emb_ids,
        "shape": (b, k, d),
        "l1": l1,
        "relu_x": relu_x,
        "l2": l2,
    }
    return logits, cache


def backward(model: dict[str, np.ndarray], cache: dict[str, object], dlogits: np.ndarray) -> dict[str, np.ndarray]:
    grads: dict[str, np.ndarray] = {}

    dh, dW2, db2 = linear_backward(dlogits, cache["l2"])  # type: ignore[arg-type]
    grads["W2"] = dW2
    grads["b2"] = db2

    dh_pre = relu_backward(dh, cache["relu_x"])  # type: ignore[arg-type]
    dX, dW1, db1 = linear_backward(dh_pre, cache["l1"])  # type: ignore[arg-type]
    grads["W1"] = dW1
    grads["b1"] = db1

    b, k, d = cache["shape"]  # type: ignore[misc]
    de = dX.reshape(b, k, d)

    emb_ids = cache["emb_ids"]  # type: ignore[assignment]
    grads["W_embed"] = embed_backward(de, emb_ids, vocab_size=VOCAB_SIZE)

    return grads