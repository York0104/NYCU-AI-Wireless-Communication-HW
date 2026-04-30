from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
sys.path.insert(0, str(SRC_ROOT / "models"))
sys.path.insert(0, str(SRC_ROOT / "utils"))

from da_tcfnet import build_da_tcfnet  # noqa: E402
from metrics import cosine_similarity, nmse_db  # noqa: E402


def import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorFlow is required to run the proposed model test script.") from exc
    return tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", default="data/sequence/test_sequences_t10.npz")
    parser.add_argument("--weights", default="saved_model/proposed/da_tcfnet_best.weights.h5")
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--encoded-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tf = import_tensorflow()
    print("Starting DA-TCFNet evaluation")
    print(f"Data file            : {ROOT / args.data_file}")
    print(f"Weights              : {ROOT / args.weights}")
    print(f"Time steps           : {args.time_steps}")
    print(f"Encoded dim          : {args.encoded_dim}")
    print(f"Batch size           : {args.batch_size}")

    with np.load(ROOT / args.data_file, allow_pickle=True) as data:
        x_seq = data["x_seq"].astype(np.float32)
        x_target = data["x_target"].astype(np.float32)
        doppler = data["doppler"].astype(np.float32)
    print(
        "Loaded test dataset  : "
        f"x_seq={x_seq.shape}, "
        f"x_target={x_target.shape}, "
        f"doppler={doppler.shape}"
    )

    print("Building model...")
    model = build_da_tcfnet(tf=tf, time_steps=args.time_steps, encoded_dim=args.encoded_dim)
    print("Loading weights...")
    model.load_weights(ROOT / args.weights)
    print("Running inference on test set...")
    pred = model.predict({"x_seq": x_seq, "doppler": doppler}, batch_size=args.batch_size, verbose=0)
    print(f"NMSE(dB): {nmse_db(x_target, pred):.4f}")
    print(f"Cosine similarity: {cosine_similarity(x_target, pred):.6f}")
    print("Finished DA-TCFNet evaluation.")


if __name__ == "__main__":
    main()
