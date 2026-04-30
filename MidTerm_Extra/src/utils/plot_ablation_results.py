from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


MODEL_ORDER = ["CsiNet", "CsiNet-LSTM", "DA-TCFNet"]
MODEL_COLORS = {
    "CsiNet": "#7A7A7A",
    "CsiNet-LSTM": "#3A7D44",
    "DA-TCFNet": "#1F4E79",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rho_label(value: float) -> str:
    if abs(value - 0.5) < 0.02:
        return "rho=0.50\nHigh Doppler"
    if abs(value - 0.8) < 0.02:
        return "rho=0.80\nMedium Doppler"
    if abs(value - 0.95) < 0.02:
        return "rho=0.95\nLow Doppler"
    return f"rho={value:.2f}"


def plot_nmse_grouped(rows: list[dict[str, str]], output: Path) -> None:
    rho_values = sorted({float(r["rho"]) for r in rows})
    x = np.arange(len(rho_values))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for idx, model in enumerate(MODEL_ORDER):
        values = []
        for rho in rho_values:
            match = next(r for r in rows if r["model"] == model and abs(float(r["rho"]) - rho) < 1e-9)
            values.append(float(match["nmse_db"]))
        ax.bar(x + (idx - 1) * width, values, width, label=model, color=MODEL_COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels([rho_label(v) for v in rho_values], fontsize=10)
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Ablation Comparison on Generated Time-Varying CSI Sequences")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_nmse_line(rows: list[dict[str, str]], output: Path) -> None:
    rho_values = sorted({float(r["rho"]) for r in rows})
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    for model in MODEL_ORDER:
        xs = []
        ys = []
        for rho in rho_values:
            match = next(r for r in rows if r["model"] == model and abs(float(r["rho"]) - rho) < 1e-9)
            xs.append(rho)
            ys.append(float(match["nmse_db"]))
        ax.plot(xs, ys, marker="o", linewidth=2.2, markersize=7, label=model, color=MODEL_COLORS[model])

    ax.set_xlabel("Temporal correlation coefficient rho")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("NMSE versus Temporal Correlation")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_inference_latency(rows: list[dict[str, str]], output: Path) -> None:
    unique_models = []
    for model in MODEL_ORDER:
        if any(r["model"] == model for r in rows):
            unique_models.append(model)

    latency_values = []
    for model in unique_models:
        model_rows = [r for r in rows if r["model"] == model]
        latency_values.append(np.mean([float(r["infer_seconds_per_sample"]) for r in model_rows]) * 1000.0)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bars = ax.bar(unique_models, latency_values, color=[MODEL_COLORS[m] for m in unique_models])
    ax.set_ylabel("Inference time per sample (ms)")
    ax.set_title("Model Inference Latency")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, val in zip(bars, latency_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_training_curve(history_rows: list[dict[str, str]], output: Path) -> None:
    epochs = [int(r["epoch"]) for r in history_rows]
    loss = [float(r["loss"]) for r in history_rows]
    val_loss = [float(r["val_loss"]) for r in history_rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(epochs, loss, marker="o", linewidth=2.0, label="Train loss", color="#1F4E79")
    ax.plot(epochs, val_loss, marker="s", linewidth=2.0, linestyle="--", label="Val loss", color="#C75D2C")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("DA-TCFNet Training Curve")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-csv", default="result/ablation_medium_triplet/ablation_results.csv")
    parser.add_argument("--history-csv", default="result/medium/history_da_tcfnet.csv")
    parser.add_argument("--figure-dir", default="result/figures")
    args = parser.parse_args()

    figure_dir = ROOT / args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    ablation_rows = read_csv(ROOT / args.ablation_csv)
    plot_nmse_grouped(ablation_rows, figure_dir / "ablation_nmse_grouped.png")
    plot_nmse_line(ablation_rows, figure_dir / "ablation_nmse_line.png")
    plot_inference_latency(ablation_rows, figure_dir / "ablation_latency.png")

    history_path = ROOT / args.history_csv
    if history_path.exists():
        history_rows = read_csv(history_path)
        plot_training_curve(history_rows, figure_dir / "da_tcfnet_training_curve.png")

    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
