from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from load_cost2100 import DATASET_SPECS, load_ht, reshape_ht


def synthesize_temporal_sequence(
    frames: np.ndarray,
    time_steps: int,
    rho: float,
    limit: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if limit is not None:
        frames = frames[:limit]

    count = len(frames)
    if count == 0:
        raise ValueError("No frames available to synthesize sequences.")

    seq = np.empty((count, time_steps, *frames.shape[1:]), dtype=np.float32)
    target = np.empty((count, *frames.shape[1:]), dtype=np.float32)
    doppler = np.full((count, 1), rho, dtype=np.float32)

    random_indices = rng.integers(0, count, size=(count, time_steps))
    prev = frames[random_indices[:, 0]]
    seq[:, 0] = prev
    for step in range(1, time_steps):
        fresh = frames[random_indices[:, step]]
        mixed = rho * prev + np.sqrt(max(1e-6, 1.0 - rho**2)) * fresh
        mixed = np.clip(mixed, 0.0, 1.0)
        seq[:, step] = mixed
        prev = mixed

    target[:] = seq[:, -1]
    return seq, target, doppler


def build_sequences_for_split(
    data_dir: Path,
    datasets: list[str],
    split: str,
    time_steps: int,
    rho_values: list[float],
    limit_per_dataset: int | None,
    seed: int,
) -> dict[str, np.ndarray]:
    seq_parts = []
    target_parts = []
    doppler_parts = []
    labels = []

    for dataset_idx, dataset in enumerate(datasets):
        frames = reshape_ht(load_ht(data_dir, dataset, split))
        for rho_idx, rho in enumerate(rho_values):
            seq, target, doppler = synthesize_temporal_sequence(
                frames=frames,
                time_steps=time_steps,
                rho=rho,
                limit=limit_per_dataset,
                seed=seed + dataset_idx * 100 + rho_idx,
            )
            seq_parts.append(seq)
            target_parts.append(target)
            doppler_parts.append(doppler)
            labels.append(np.full((len(seq),), dataset, dtype="<U32"))

    return {
        "x_seq": np.concatenate(seq_parts, axis=0),
        "x_target": np.concatenate(target_parts, axis=0),
        "doppler": np.concatenate(doppler_parts, axis=0),
        "dataset": np.concatenate(labels, axis=0),
    }


def save_npz(output_path: Path, payload: dict[str, np.ndarray]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="../../MidTerm_Q7/data/cost2100_official")
    parser.add_argument("--output-dir", default="../../MidTerm_Extra/data/sequence")
    parser.add_argument("--time-steps", type=int, default=10)
    parser.add_argument("--rho-list", nargs="+", type=float, default=[0.95, 0.8, 0.5])
    parser.add_argument("--train-limit", type=int, default=300)
    parser.add_argument("--val-limit", type=int, default=100)
    parser.add_argument("--test-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=535100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    datasets = [spec.name for spec in DATASET_SPECS]

    print("Starting synthetic time-varying CSI sequence generation")
    print(f"Input data directory : {data_dir}")
    print(f"Output directory     : {output_dir}")
    print(f"Time steps           : {args.time_steps}")
    print(f"Rho list             : {args.rho_list}")
    print(
        "Sample limits        : "
        f"train={args.train_limit}, val={args.val_limit}, test={args.test_limit}"
    )

    split_to_limit = {
        "train": args.train_limit,
        "val": args.val_limit,
        "test": args.test_limit,
    }
    for split, limit in split_to_limit.items():
        print(f"\nBuilding {split} split sequences...")
        payload = build_sequences_for_split(
            data_dir=data_dir,
            datasets=datasets,
            split=split,
            time_steps=args.time_steps,
            rho_values=args.rho_list,
            limit_per_dataset=limit,
            seed=args.seed,
        )
        payload["x_seq"] = payload["x_seq"].astype(np.float16)
        payload["x_target"] = payload["x_target"].astype(np.float16)
        payload["doppler"] = payload["doppler"].astype(np.float16)
        output_path = output_dir / f"{split}_sequences_t{args.time_steps}.npz"
        save_npz(output_path, payload)
        print(
            f"Wrote {output_path} | "
            f"x_seq shape={payload['x_seq'].shape}, "
            f"x_target shape={payload['x_target'].shape}"
        )

    print("\nFinished sequence generation.")


if __name__ == "__main__":
    main()
