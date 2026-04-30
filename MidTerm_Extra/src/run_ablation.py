from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "utils"))

from da_tcfnet import build_da_tcfnet  # noqa: E402
from lstm_baseline import build_lstm_baseline  # noqa: E402
from single_frame_baseline import build_single_frame_baseline  # noqa: E402
from complexity import average_inference_seconds, count_trainable_params  # noqa: E402
from metrics import cosine_similarity, nmse_db  # noqa: E402


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is required to run the ablation script.") from exc
    return tf


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    payload["x_seq"] = payload["x_seq"].astype(np.float32)
    payload["x_target"] = payload["x_target"].astype(np.float32)
    payload["doppler"] = payload["doppler"].astype(np.float32)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/sequence")
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--encoded-dim", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--result-dir", default="result/ablation")
    parser.add_argument("--save-model-dir", default=None)
    parser.add_argument("--seed", type=int, default=535100)
    return parser.parse_args()


def select_by_rho(data: dict[str, np.ndarray], rho: float, atol: float = 1e-6) -> dict[str, np.ndarray]:
    mask = np.isclose(data["doppler"].reshape(-1), rho, atol=atol)
    return {
        "x_seq": data["x_seq"][mask],
        "x_target": data["x_target"][mask],
        "doppler": data["doppler"][mask],
    }


def main() -> None:
    args = parse_args()
    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = ROOT / args.data_dir
    print("Starting ablation run")
    print(f"Sequence data dir    : {data_dir}")
    print(f"Time steps           : {args.time_steps}")
    print(f"Encoded dim          : {args.encoded_dim}")
    print(f"Epochs               : {args.epochs}")
    print(f"Batch size           : {args.batch_size}")

    train_data = load_npz(data_dir / f"train_sequences_t{args.time_steps}.npz")
    val_data = load_npz(data_dir / f"val_sequences_t{args.time_steps}.npz")
    test_data = load_npz(data_dir / f"test_sequences_t{args.time_steps}.npz")
    print(
        "Loaded datasets      : "
        f"train={train_data['x_seq'].shape}, "
        f"val={val_data['x_seq'].shape}, "
        f"test={test_data['x_seq'].shape}"
    )

    builders = {
        "CsiNet": lambda: build_single_frame_baseline(tf, time_steps=args.time_steps, encoded_dim=args.encoded_dim),
        "DA-TCFNet": lambda: build_da_tcfnet(tf, time_steps=args.time_steps, encoded_dim=args.encoded_dim),
        "CsiNet-LSTM": lambda: build_lstm_baseline(tf, time_steps=args.time_steps, encoded_dim=args.encoded_dim),
    }
    result_dir = ROOT / args.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / args.save_model_dir if args.save_model_dir else result_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    total_models = len(builders)
    for model_index, (name, builder) in enumerate(builders.items(), start=1):
        print(f"\n[{model_index}/{total_models}] Training {name}")
        model = builder()
        model.fit(
            {"x_seq": train_data["x_seq"], "doppler": train_data["doppler"]},
            train_data["x_target"],
            validation_data=(
                {"x_seq": val_data["x_seq"], "doppler": val_data["doppler"]},
                val_data["x_target"],
            ),
            epochs=args.epochs,
            batch_size=args.batch_size,
            shuffle=True,
            verbose=2,
        )
        model_filename = name.lower().replace("-", "_").replace(" ", "_") + ".weights.h5"
        model_path = model_dir / model_filename
        model.save_weights(model_path)
        print(f"[{model_index}/{total_models}] Saved weights to {model_path}")

        rho_values = sorted(np.unique(test_data["doppler"].reshape(-1)).tolist())
        print(f"[{model_index}/{total_models}] Evaluating {name} on rho values: {rho_values}")
        for rho in rho_values:
            split = select_by_rho(test_data, rho)
            inputs = {"x_seq": split["x_seq"], "doppler": split["doppler"]}
            pred = model.predict(inputs, batch_size=args.batch_size, verbose=0)
            nmse_value = round(nmse_db(split["x_target"], pred), 4)
            cos_value = round(cosine_similarity(split["x_target"], pred), 6)
            infer_value = average_inference_seconds(model, inputs)
            print(
                f"  rho={rho:.4f} | "
                f"samples={len(split['x_seq'])} | "
                f"NMSE={nmse_value} dB | "
                f"cosine={cos_value}"
            )
            rows.append(
                {
                    "model": name,
                    "rho": rho,
                    "nmse_db": nmse_value,
                    "cosine_similarity": cos_value,
                    "trainable_params": count_trainable_params(model),
                    "infer_seconds_per_sample": f"{infer_value:.8f}",
                }
            )

    output = result_dir / "ablation_results.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote ablation CSV    : {output}")
    print("Finished ablation run.")


if __name__ == "__main__":
    main()
