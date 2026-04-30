from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "models"))

from da_tcfnet import build_da_tcfnet  # noqa: E402
from lstm_baseline import build_lstm_baseline  # noqa: E402
from single_frame_baseline import build_single_frame_baseline  # noqa: E402


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is required to plot reconstruction examples.") from exc
    return tf


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["x_seq"] = payload["x_seq"].astype(np.float32)
    payload["x_target"] = payload["x_target"].astype(np.float32)
    payload["doppler"] = payload["doppler"].astype(np.float32)
    return payload


def csi_magnitude(x: np.ndarray) -> np.ndarray:
    return np.abs((x[:, :, 0] - 0.5) + 1j * (x[:, :, 1] - 0.5))


def rho_label(value: float) -> str:
    if abs(value - 0.5) < 0.02:
        return "rho=0.50 (High Doppler)"
    if abs(value - 0.8) < 0.02:
        return "rho=0.80 (Medium Doppler)"
    if abs(value - 0.95) < 0.02:
        return "rho=0.95 (Low Doppler)"
    return f"rho={value:.2f}"


def select_example(data: dict[str, np.ndarray], rho: float) -> dict[str, np.ndarray]:
    mask = np.isclose(data["doppler"].reshape(-1), rho, atol=0.02)
    idx = int(np.flatnonzero(mask)[0])
    return {
        "x_seq": data["x_seq"][idx : idx + 1],
        "x_target": data["x_target"][idx],
        "doppler": data["doppler"][idx : idx + 1],
    }


def build_models(tf, time_steps: int, encoded_dim: int):
    return {
        "CsiNet": build_single_frame_baseline(tf, time_steps=time_steps, encoded_dim=encoded_dim),
        "CsiNet-LSTM": build_lstm_baseline(tf, time_steps=time_steps, encoded_dim=encoded_dim),
        "DA-TCFNet": build_da_tcfnet(tf, time_steps=time_steps, encoded_dim=encoded_dim),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="data/sequence_submit/test_sequences_t10.npz")
    parser.add_argument("--model-dir", default="result/ablation_submit_triplet/models")
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--encoded-dim", type=int, default=128)
    parser.add_argument("--rho-values", nargs="+", type=float, default=[0.95, 0.5])
    parser.add_argument("--figure-dir", default="result/figures_submit")
    args = parser.parse_args()

    tf = import_tensorflow()
    data = load_npz(ROOT / args.data_file)
    figure_dir = ROOT / args.figure_dir
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / args.model_dir

    model_paths = {
        "CsiNet": model_dir / "csinet.weights.h5",
        "CsiNet-LSTM": model_dir / "csinet_lstm.weights.h5",
        "DA-TCFNet": model_dir / "da_tcfnet.weights.h5",
    }
    for name, path in model_paths.items():
        if not path.exists():
            raise SystemExit(f"Missing weights for {name}: {path}")

    models = build_models(tf, time_steps=args.time_steps, encoded_dim=args.encoded_dim)
    for name, model in models.items():
        model.load_weights(model_paths[name])

    row_names = ["Original", "CsiNet", "CsiNet-LSTM", "DA-TCFNet"]
    col_values = args.rho_values
    fig, axes = plt.subplots(len(row_names), len(col_values), figsize=(4.0 * len(col_values), 3.2 * len(row_names)))
    if len(col_values) == 1:
        axes = np.expand_dims(axes, axis=1)

    for col_idx, rho in enumerate(col_values):
        example = select_example(data, rho)
        inputs = {"x_seq": example["x_seq"], "doppler": example["doppler"]}
        target_mag = csi_magnitude(example["x_target"])
        predictions = {
            "CsiNet": csi_magnitude(models["CsiNet"].predict(inputs, verbose=0)[0]),
            "CsiNet-LSTM": csi_magnitude(models["CsiNet-LSTM"].predict(inputs, verbose=0)[0]),
            "DA-TCFNet": csi_magnitude(models["DA-TCFNet"].predict(inputs, verbose=0)[0]),
        }

        images = {
            "Original": target_mag,
            "CsiNet": predictions["CsiNet"],
            "CsiNet-LSTM": predictions["CsiNet-LSTM"],
            "DA-TCFNet": predictions["DA-TCFNet"],
        }

        for row_idx, row_name in enumerate(row_names):
            ax = axes[row_idx, col_idx]
            ax.imshow(images[row_name].T, cmap="gray", origin="lower")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(rho_label(rho), fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=10)

    fig.suptitle("CSI Reconstruction Visualization Across Doppler Conditions", fontsize=13)
    fig.tight_layout()
    output = figure_dir / "reconstruction_comparison.png"
    fig.savefig(output, dpi=220)
    plt.close(fig)
    print(f"Wrote figure to {output}")


if __name__ == "__main__":
    main()
