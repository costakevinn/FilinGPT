#!/usr/bin/env python3
"""
make_comparison_report.py

Generates a clean baseline vs financial comparison report:
- reports/compare_loss_first_200.png
- reports/compare_ppl_first_200.png
- reports/financial_loss_full.png
- reports/financial_ppl_full.png
- reports/compare_loss_logx.png
- reports/compare_ppl_logx.png
- reports/metrics.md
- reports/metrics.json

Assumes artifacts:
- data/artifacts/filingpt_mlp_baseline_v1.history.json
- data/artifacts/filingpt_mlp_financial_v1.json

Run:
  python scripts/make_comparison_report.py
or:
  python scripts/make_comparison_report.py --out reports
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_losses(obj: Dict[str, Any]) -> List[float]:
    # We expect "losses" but keep it robust.
    for k in ("losses", "train_losses", "loss"):
        v = obj.get(k)
        if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            return [float(x) for x in v]
    raise ValueError("Could not find a numeric 'losses' list in JSON.")


def _summarize(obj: Dict[str, Any], losses: List[float]) -> Dict[str, Any]:
    final_loss = float(obj.get("final_loss", losses[-1]))
    steps = int(obj.get("steps", len(losses)))
    lr = obj.get("lr", None)
    seed = obj.get("seed", None)
    batch_size = obj.get("batch_size", None)
    vocab_size = obj.get("vocab_size", None)
    ctx_len = obj.get("ctx_len", None)
    seq_len = obj.get("seq_len", None)
    n_sequences = obj.get("n_sequences", None)
    batches_path = obj.get("batches_path", None)

    start_loss = float(losses[0])
    start_ppl = float(math.exp(start_loss))
    final_ppl = float(math.exp(final_loss))
    ppl_reduction_pct = float((1.0 - (final_ppl / start_ppl)) * 100.0)

    return {
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "batch_size": batch_size,
        "vocab_size": vocab_size,
        "ctx_len": ctx_len,
        "seq_len": seq_len,
        "n_sequences": n_sequences,
        "batches_path": batches_path,
        "start_loss": start_loss,
        "final_loss": final_loss,
        "start_ppl": start_ppl,
        "final_ppl": final_ppl,
        "ppl_reduction_pct": ppl_reduction_pct,
        "n_logged_points": len(losses),
    }


def _plot_line(
    x: List[int],
    y_series: List[Tuple[str, List[float]]],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    x_log: bool = False,
) -> None:
    plt.figure()
    for label, y in y_series:
        plt.plot(x[: len(y)], y, label=label)
    if x_log:
        plt.xscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def _exp_list(xs: List[float]) -> List[float]:
    return [math.exp(x) for x in xs]


def _write_metrics_md(out_path: Path, baseline: Dict[str, Any], financial: Dict[str, Any]) -> None:
    def fmt(x: Any, nd: int = 4) -> str:
        if x is None:
            return "—"
        if isinstance(x, (int,)):
            return str(x)
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    lines = []
    lines.append("## Baseline vs Financial\n")
    lines.append("| Field | baseline_v1 | financial_v1 |")
    lines.append("|---|---:|---:|")
    lines.append(f"| steps | {fmt(baseline['steps'],0)} | {fmt(financial['steps'],0)} |")
    lines.append(f"| lr | {fmt(baseline['lr'],4)} | {fmt(financial['lr'],4)} |")
    lines.append(f"| seed | {fmt(baseline['seed'],0)} | {fmt(financial['seed'],0)} |")
    lines.append(f"| batch_size | {fmt(baseline['batch_size'],0)} | {fmt(financial['batch_size'],0)} |")
    lines.append(f"| vocab_size | {fmt(baseline['vocab_size'],0)} | {fmt(financial['vocab_size'],0)} |")
    lines.append(f"| ctx_len | {fmt(baseline['ctx_len'],0)} | {fmt(financial['ctx_len'],0)} |")
    lines.append(f"| seq_len | {fmt(baseline['seq_len'],0)} | {fmt(financial['seq_len'],0)} |")
    lines.append(f"| n_sequences | {fmt(baseline['n_sequences'],0)} | {fmt(financial['n_sequences'],0)} |")
    lines.append(f"| start_loss | {fmt(baseline['start_loss'],4)} | {fmt(financial['start_loss'],4)} |")
    lines.append(f"| final_loss | {fmt(baseline['final_loss'],4)} | {fmt(financial['final_loss'],4)} |")
    lines.append(f"| start_ppl (exp(loss)) | {fmt(baseline['start_ppl'],2)} | {fmt(financial['start_ppl'],2)} |")
    lines.append(f"| final_ppl (exp(loss)) | {fmt(baseline['final_ppl'],2)} | {fmt(financial['final_ppl'],2)} |")
    lines.append(f"| ppl_reduction_pct | {fmt(baseline['ppl_reduction_pct'],2)}% | {fmt(financial['ppl_reduction_pct'],2)}% |")
    lines.append("")
    lines.append("### Plots\n")
    lines.append("- `compare_loss_first_200.png`: baseline vs financial (first 200 steps)")
    lines.append("- `compare_ppl_first_200.png`: perplexity baseline vs financial (first 200 steps)")
    lines.append("- `financial_loss_full.png`: financial loss (full training)")
    lines.append("- `financial_ppl_full.png`: financial perplexity (full training)")
    lines.append("- `compare_loss_logx.png`: baseline vs financial with log-scaled x-axis")
    lines.append("- `compare_ppl_logx.png`: perplexity baseline vs financial with log-scaled x-axis")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--baseline",
        type=str,
        default="data/artifacts/filingpt_mlp_baseline_v1.history.json",
        help="Path to baseline history JSON",
    )
    ap.add_argument(
        "--financial",
        type=str,
        default="data/artifacts/filingpt_mlp_financial_v1.json",
        help="Path to financial history JSON",
    )
    ap.add_argument("--out", type=str, default="reports", help="Output directory")
    ap.add_argument("--first_n", type=int, default=200, help="Steps to compare in the early plot")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    financial_path = Path(args.financial)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_path.exists():
        raise SystemExit(f"[ERR] Missing baseline JSON: {baseline_path.as_posix()}")
    if not financial_path.exists():
        raise SystemExit(f"[ERR] Missing financial JSON: {financial_path.as_posix()}")

    b_obj = _read_json(baseline_path)
    f_obj = _read_json(financial_path)

    b_losses = _get_losses(b_obj)
    f_losses = _get_losses(f_obj)

    b_sum = _summarize(b_obj, b_losses)
    f_sum = _summarize(f_obj, f_losses)

    # ---- Early comparison (first N) ----
    n = min(args.first_n, len(b_losses), len(f_losses))
    x_early = list(range(1, n + 1))  # start at 1 so log-scale plot doesn't hit 0
    b_early = b_losses[:n]
    f_early = f_losses[:n]

    _plot_line(
        x=x_early,
        y_series=[(f"baseline_v1 ({b_sum['steps']} steps)", b_early), (f"financial_v1 ({f_sum['steps']} steps)", f_early)],
        title=f"Training Loss Comparison (first {n} steps)",
        xlabel="Step",
        ylabel="Loss",
        out_path=out_dir / "compare_loss_first_200.png",
        x_log=False,
    )

    _plot_line(
        x=x_early,
        y_series=[
            (f"baseline_v1 ({b_sum['steps']} steps)", _exp_list(b_early)),
            (f"financial_v1 ({f_sum['steps']} steps)", _exp_list(f_early)),
        ],
        title=f"Perplexity Comparison (first {n} steps)",
        xlabel="Step",
        ylabel="Perplexity (exp(loss))",
        out_path=out_dir / "compare_ppl_first_200.png",
        x_log=False,
    )

    # ---- Full financial only ----
    x_fin = list(range(1, len(f_losses) + 1))
    _plot_line(
        x=x_fin,
        y_series=[(f"financial_v1 ({f_sum['steps']} steps)", f_losses)],
        title="Financial Checkpoint Training Loss (full)",
        xlabel="Step",
        ylabel="Loss",
        out_path=out_dir / "financial_loss_full.png",
        x_log=False,
    )
    _plot_line(
        x=x_fin,
        y_series=[(f"financial_v1 ({f_sum['steps']} steps)", _exp_list(f_losses))],
        title="Financial Checkpoint Perplexity (full)",
        xlabel="Step",
        ylabel="Perplexity (exp(loss))",
        out_path=out_dir / "financial_ppl_full.png",
        x_log=False,
    )

    # ---- Log-x comparison (nice paper-style) ----
    # Build x so both can be plotted together: use their own lengths, but x starts at 1
    x_b = list(range(1, len(b_losses) + 1))
    x_f = list(range(1, len(f_losses) + 1))

    # For log-x plot, we need a single x list. We'll plot each with its own x via plt directly.
    plt.figure()
    plt.plot(x_b, b_losses, label=f"baseline_v1 ({b_sum['steps']} steps)")
    plt.plot(x_f, f_losses, label=f"financial_v1 ({f_sum['steps']} steps)")
    plt.xscale("log")
    plt.title("Training Loss Comparison (log-scaled steps)")
    plt.xlabel("Step (log scale)")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_loss_logx.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(x_b, _exp_list(b_losses), label=f"baseline_v1 ({b_sum['steps']} steps)")
    plt.plot(x_f, _exp_list(f_losses), label=f"financial_v1 ({f_sum['steps']} steps)")
    plt.xscale("log")
    plt.title("Perplexity Comparison (log-scaled steps)")
    plt.xlabel("Step (log scale)")
    plt.ylabel("Perplexity (exp(loss))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_ppl_logx.png", dpi=160)
    plt.close()

    # ---- Write metrics ----
    metrics = {"baseline_v1": b_sum, "financial_v1": f_sum}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    _write_metrics_md(out_dir / "metrics.md", b_sum, f_sum)

    print("[OK] Wrote report files to:", out_dir.as_posix())
    for name in (
        "compare_loss_first_200.png",
        "compare_ppl_first_200.png",
        "financial_loss_full.png",
        "financial_ppl_full.png",
        "compare_loss_logx.png",
        "compare_ppl_logx.png",
        "metrics.md",
        "metrics.json",
    ):
        print(" -", (out_dir / name).as_posix())


if __name__ == "__main__":
    main()