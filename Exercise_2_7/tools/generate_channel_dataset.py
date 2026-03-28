import argparse
from pathlib import Path

import numpy as np


def build_power_delay_profile(num_taps: int, decay: float) -> np.ndarray:
    """Create a normalized exponential power delay profile."""
    taps = np.arange(num_taps, dtype=np.float64)
    pdp = np.exp(-decay * taps)
    pdp /= pdp.sum()
    return pdp


def generate_channels(num_samples: int, num_taps: int, decay: float, rng: np.random.Generator) -> np.ndarray:
    """Generate normalized complex Rayleigh channel samples."""
    pdp = build_power_delay_profile(num_taps, decay)
    sigma = np.sqrt(pdp / 2.0).astype(np.float32)
    real = rng.standard_normal((num_samples, num_taps), dtype=np.float32)
    imag = rng.standard_normal((num_samples, num_taps), dtype=np.float32)
    channels = (real + 1j * imag) * sigma[np.newaxis, :]

    # Normalize each sample to unit average channel power to stabilize SNR behavior.
    power = np.sum(np.abs(channels) ** 2, axis=1, keepdims=True)
    channels = channels / np.sqrt(np.maximum(power, 1e-12))
    return channels.astype(np.complex64)


def save_dataset(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate channel_train.npy and channel_test.npy for Exercise 2.7."
    )
    parser.add_argument("--train-size", type=int, default=100000, help="Number of training channels.")
    parser.add_argument("--test-size", type=int, default=390000, help="Number of test channels.")
    parser.add_argument("--num-taps", type=int, default=16, help="Length of the channel impulse response.")
    parser.add_argument(
        "--decay",
        type=float,
        default=0.25,
        help="Exponential power-delay-profile decay factor.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where channel_train.npy and channel_test.npy will be saved.",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(
        f"Generating channels: train={args.train_size}, test={args.test_size}, "
        f"taps={args.num_taps}, decay={args.decay}, seed={args.seed}"
    )

    channel_train = generate_channels(args.train_size, args.num_taps, args.decay, rng)
    channel_test = generate_channels(args.test_size, args.num_taps, args.decay, rng)

    train_path = args.output_dir / "channel_train.npy"
    test_path = args.output_dir / "channel_test.npy"
    save_dataset(train_path, channel_train)
    save_dataset(test_path, channel_test)

    print(f"Saved training set to: {train_path}")
    print(f"Saved test set to: {test_path}")
    print(f"Training shape: {channel_train.shape}, dtype: {channel_train.dtype}")
    print(f"Test shape: {channel_test.shape}, dtype: {channel_test.dtype}")


if __name__ == "__main__":
    main()
