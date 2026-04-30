from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def rho_label(value: float) -> str:
    if abs(value - 0.5) < 0.02:
        return "rho=0.50\nHigh Doppler"
    if abs(value - 0.8) < 0.02:
        return "rho=0.80\nMedium Doppler"
    if abs(value - 0.95) < 0.02:
        return "rho=0.95\nLow Doppler"
    return f"rho={value:.2f}"


def plot_sample_counts(datasets: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    split_names = ["train", "val", "test"]
    counts = [len(datasets[split]["x_seq"]) for split in split_names]

    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bars = ax.bar(split_names, counts, color=["#1F4E79", "#3A7D44", "#C75D2C"])
    ax.set_ylabel("Number of sequences")
    ax.set_title("Sequence Counts per Split")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.01, str(count), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_rho_counts(test_data: dict[str, np.ndarray], output: Path) -> None:
    rho_values, counts = np.unique(test_data["doppler"].reshape(-1), return_counts=True)
    labels = [rho_label(float(v)) for v in rho_values]

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    bars = ax.bar(labels, counts, color="#5B8E7D")
    ax.set_ylabel("Number of sequences")
    ax.set_title("Sequence Counts by Temporal Correlation Setting")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, count + max(counts) * 0.02, str(int(count)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_temporal_decay(time_steps: int, rho_values: list[float], output: Path) -> None:
    lags = np.arange(time_steps)

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    colors = ["#1F4E79", "#3A7D44", "#C75D2C"]
    for rho, color in zip(rho_values, colors, strict=False):
        decay = rho**lags
        ax.plot(lags, decay, marker="o", linewidth=2.2, color=color, label=rho_label(rho).replace("\n", " "))
    ax.set_xlabel("Time lag")
    ax.set_ylabel(r"Approximate correlation $\rho^{lag}$")
    ax.set_title("Temporal Correlation Settings Used for Sequence Synthesis")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sequence_submit")
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--figure-dir", default="result/figures_submit")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    figure_dir = ROOT / args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        split: load_npz(data_dir / f"{split}_sequences_t{args.time_steps}.npz")
        for split in ("train", "val", "test")
    }
    rho_values = sorted(float(v) for v in np.unique(datasets["test"]["doppler"].reshape(-1)))

    plot_sample_counts(datasets, figure_dir / "sequence_sample_counts.png")
    plot_rho_counts(datasets["test"], figure_dir / "sequence_rho_counts.png")
    plot_temporal_decay(args.time_steps, rho_values, figure_dir / "temporal_correlation_settings.png")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
