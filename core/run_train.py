# core/run_train.py
from core.train import train
from core.utils import save_model_npz, save_history_json


def run():
    model, hist = train(
        batches_path="data/training/batches.jsonl",
        steps=int(100000),
        lr=2e-2,
        seed=42,
        log_every=10,
        batch_size=32,
    )

    save_model_npz(model, "data/artifacts/filingpt_mlp_financial_v1.npz")
    save_history_json(hist, "data/artifacts/filingpt_mlp_financial_v1.json")
    print("[OK] Training finished & saved.")


if __name__ == "__main__":
    run()