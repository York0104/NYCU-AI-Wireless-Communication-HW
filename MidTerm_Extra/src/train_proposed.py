from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "data"))
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "utils"))

from da_tcfnet import build_da_tcfnet  # noqa: E402
from complexity import average_inference_seconds, count_trainable_params  # noqa: E402
from metrics import cosine_similarity, nmse_db  # noqa: E402


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow is not installed. Use the same TensorFlow environment as Q7, "
            "for example `conda run -n csinet_tf python ...`."
        ) from exc
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--residual-num", type=int, default=2)
    parser.add_argument("--quant-bits", type=int, default=8)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--save-dir", default="saved_model/proposed")
    parser.add_argument("--result-dir", default="result")
    parser.add_argument("--seed", type=int, default=535100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.seed)

    data_dir = ROOT / args.data_dir
    print("Starting DA-TCFNet training")
    print(f"Sequence data dir    : {data_dir}")
    print(f"Time steps           : {args.time_steps}")
    print(f"Encoded dim          : {args.encoded_dim}")
    print(f"Epochs               : {args.epochs}")
    print(f"Batch size           : {args.batch_size}")
    print(f"Freeze encoder       : {args.freeze_encoder}")

    train_data = load_npz(data_dir / f"train_sequences_t{args.time_steps}.npz")
    val_data = load_npz(data_dir / f"val_sequences_t{args.time_steps}.npz")
    test_data = load_npz(data_dir / f"test_sequences_t{args.time_steps}.npz")
    print(
        "Loaded datasets      : "
        f"train={train_data['x_seq'].shape}, "
        f"val={val_data['x_seq'].shape}, "
        f"test={test_data['x_seq'].shape}"
    )

    model = build_da_tcfnet(
        tf=tf,
        time_steps=args.time_steps,
        encoded_dim=args.encoded_dim,
        residual_num=args.residual_num,
        quant_bits=args.quant_bits,
        freeze_encoder=args.freeze_encoder,
    )

    save_dir = ROOT / args.save_dir
    result_dir = ROOT / args.result_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=save_dir / "da_tcfnet_best.weights.h5",
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        )
    ]

    print("\nTraining model...")
    history = model.fit(
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
        callbacks=callbacks,
    )

    print("\nLoading best checkpoint for evaluation...")
    model.load_weights(save_dir / "da_tcfnet_best.weights.h5")
    test_inputs = {"x_seq": test_data["x_seq"], "doppler": test_data["doppler"]}
    print("Running test-set inference...")
    pred = model.predict(test_inputs, batch_size=args.batch_size, verbose=0)

    metrics_row = {
        "model": "DA-TCFNet",
        "time_steps": args.time_steps,
        "encoded_dim": args.encoded_dim,
        "epochs": args.epochs,
        "nmse_db": round(nmse_db(test_data["x_target"], pred), 4),
        "cosine_similarity": round(cosine_similarity(test_data["x_target"], pred), 6),
        "trainable_params": count_trainable_params(model),
        "infer_seconds_per_sample": f"{average_inference_seconds(model, test_inputs):.8f}",
    }

    history_path = result_dir / "history_da_tcfnet.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "val_loss"])
        for idx, (loss, val_loss) in enumerate(zip(history.history["loss"], history.history["val_loss"]), start=1):
            writer.writerow([idx, loss, val_loss])

    result_path = result_dir / "da_tcfnet_metrics.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_row.keys()))
        writer.writeheader()
        writer.writerow(metrics_row)

    model.save_weights(save_dir / "da_tcfnet_last.weights.h5")
    print(f"Wrote history CSV     : {history_path}")
    print(f"Wrote metrics CSV     : {result_path}")
    print(f"Saved final weights   : {save_dir / 'da_tcfnet_last.weights.h5'}")
    print("Finished DA-TCFNet training.")


if __name__ == "__main__":
    main()
