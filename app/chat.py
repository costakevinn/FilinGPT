# app/chat.py
from pathlib import Path

from core.infer import load_model_npz, generate

ARTIFACTS_DIR = Path("data/artifacts")

# Generation defaults (kept fixed for reproducibility)
MAX_NEW_TOKENS = 300
TEMPERATURE = 0.9
TOP_K = 80
SEED = 123


def list_models() -> list[Path]:
    """Return available .npz model files."""
    if not ARTIFACTS_DIR.exists():
        return []
    return sorted(ARTIFACTS_DIR.glob("*.npz"))


def main() -> None:
    models = list_models()
    if not models:
        raise SystemExit("[ERR] No .npz models found in data/artifacts")

    print("\nAvailable models:")
    for i, p in enumerate(models, 1):
        print(f"  {i:02d}) {p.name}")

    try:
        idx = int(input("\nSelect model number: ").strip()) - 1
    except ValueError:
        raise SystemExit("[ERR] Invalid selection")

    if not 0 <= idx < len(models):
        raise SystemExit("[ERR] Out of range")

    chosen = models[idx]
    model = load_model_npz(str(chosen))

    print(f"\n[OK] Loaded: {chosen.name}")
    print("Type your prompt. Empty line quits.\n")

    while True:
        try:
            prompt = input("You> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt.strip():
            break

        out = generate(
            model,
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            seed=SEED,
        )

        print("Model>", out)
        print()


if __name__ == "__main__":
    main()