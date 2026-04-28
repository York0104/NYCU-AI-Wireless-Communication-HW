"""
Dataset utilities for Exercise 2.15.

The original exercise asks for COST 2100 channel datasets.  When the
official MATLAB COST 2100 generator is available, the generated .mat files
can be replaced directly as long as they keep the same keys:

  HT     : normalized angular-delay CSI, shape [samples, 2048]
  HF_all : complex frequency-domain CSI, shape [samples, 32, 125]

This module provides a deterministic COST-2100-style surrogate generator so
the complete experiment pipeline can be reproduced on a plain Python setup.
It preserves the key CsiNet assumptions used in the reference paper: 32 BS
antennas, 1024 OFDM subcarriers, sparse angular-delay channels, and normalized
32 x 32 x 2 CsiNet inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.io as sio


IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 2
FEATURES = IMG_HEIGHT * IMG_WIDTH * IMG_CHANNELS
FREQ_BINS = 125
FFT_SIZE_FOR_METRIC = 257


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    environment: str
    area_length_m: float
    carrier_ghz: float
    user_distribution: str
    clusters: int
    delay_spread: float
    angle_spread: float


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec("D1_indoor_uniform", "indoor", 20.0, 5.3, "uniform", 7, 1.0, 1.0),
    DatasetSpec("D2_indoor_center", "indoor", 20.0, 5.3, "center", 5, 0.7, 0.7),
    DatasetSpec("D3_indoor_edge", "indoor", 20.0, 5.3, "edge", 8, 1.3, 1.1),
    DatasetSpec("D4_indoor_ring", "indoor", 20.0, 5.3, "ring", 6, 1.0, 1.4),
    DatasetSpec("D5_outdoor_uniform", "outdoor", 400.0, 0.3, "uniform", 9, 1.6, 1.2),
    DatasetSpec("D6_outdoor_clustered", "outdoor", 400.0, 0.3, "clustered", 10, 1.8, 1.5),
)


def spec_by_name(name: str) -> DatasetSpec:
    for spec in DATASET_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"Unknown dataset: {name}")


def sample_users(spec: DatasetSpec, count: int, rng: np.random.Generator) -> np.ndarray:
    half = spec.area_length_m / 2.0
    if spec.user_distribution == "uniform":
        return rng.uniform(-half, half, size=(count, 2))
    if spec.user_distribution == "center":
        pts = rng.normal(0.0, spec.area_length_m / 9.0, size=(count, 2))
        return np.clip(pts, -half, half)
    if spec.user_distribution == "edge":
        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        radius = rng.uniform(0.65 * half, half, size=count)
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    if spec.user_distribution == "ring":
        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        radius = rng.normal(0.55 * half, 0.08 * half, size=count)
        radius = np.clip(radius, 0.30 * half, 0.85 * half)
        return np.column_stack((radius * np.cos(angles), radius * np.sin(angles)))
    if spec.user_distribution == "clustered":
        centers = np.array(
            [
                [-0.45 * half, -0.25 * half],
                [0.35 * half, 0.35 * half],
                [0.20 * half, -0.45 * half],
            ]
        )
        idx = rng.integers(0, len(centers), size=count)
        pts = centers[idx] + rng.normal(0.0, spec.area_length_m / 12.0, size=(count, 2))
        return np.clip(pts, -half, half)
    raise ValueError(f"Unsupported distribution: {spec.user_distribution}")


def _add_gaussian_blob(grid: np.ndarray, row: float, col: float, amp: complex, sig_r: float, sig_c: float) -> None:
    rr = np.arange(IMG_HEIGHT, dtype=np.float32)[:, None]
    cc = np.arange(IMG_WIDTH, dtype=np.float32)[None, :]
    blob = np.exp(-0.5 * (((rr - row) / sig_r) ** 2 + ((cc - col) / sig_c) ** 2))
    grid += amp * blob


def generate_channel_matrices(spec: DatasetSpec, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    users = sample_users(spec, count, rng)
    h_ad = np.zeros((count, IMG_HEIGHT, IMG_WIDTH), dtype=np.complex64)

    max_distance = np.sqrt(2.0) * spec.area_length_m / 2.0 + 1e-6
    distances = np.linalg.norm(users, axis=1) + 1.0
    distance_factor = np.clip(distances / max_distance, 0.05, 1.5)

    for i, (x_pos, y_pos) in enumerate(users):
        azimuth = (np.arctan2(y_pos, x_pos) + np.pi) / (2.0 * np.pi)
        base_angle = azimuth * (IMG_WIDTH - 1)
        pathloss = 1.0 / (distance_factor[i] ** (1.2 if spec.environment == "indoor" else 1.7))

        clusters = max(2, int(rng.poisson(spec.clusters - 1) + 1))
        for _ in range(clusters):
            delay_mean = rng.gamma(shape=1.5 * spec.delay_spread, scale=3.2)
            delay_row = np.clip(delay_mean + 3.0 * distance_factor[i], 0, IMG_HEIGHT - 1)
            angle_col = np.clip(base_angle + rng.normal(0.0, 3.0 * spec.angle_spread), 0, IMG_WIDTH - 1)
            amp_mag = pathloss * rng.rayleigh(scale=0.8) * np.exp(-delay_row / (9.0 + 2.0 * spec.delay_spread))
            amp_phase = rng.uniform(-np.pi, np.pi)
            amp = amp_mag * np.exp(1j * amp_phase)
            sig_r = rng.uniform(0.45, 1.2 + 0.3 * spec.delay_spread)
            sig_c = rng.uniform(0.45, 1.2 + 0.3 * spec.angle_spread)
            _add_gaussian_blob(h_ad[i], delay_row, angle_col, amp, sig_r, sig_c)

        noise = (rng.normal(0, 0.01, size=(IMG_HEIGHT, IMG_WIDTH)) + 1j * rng.normal(0, 0.01, size=(IMG_HEIGHT, IMG_WIDTH)))
        h_ad[i] += noise.astype(np.complex64)

    scale = np.percentile(np.abs(h_ad), 99.5)
    h_ad = h_ad / max(scale, 1e-8)
    h_ad = np.clip(h_ad.real, -1.0, 1.0) + 1j * np.clip(h_ad.imag, -1.0, 1.0)

    ht = np.stack((h_ad.real + 0.5, h_ad.imag + 0.5), axis=1)
    ht = np.clip(ht, 0.0, 1.0).astype(np.float32)
    ht = ht.reshape(count, FEATURES)

    hf = np.fft.fft(
        np.concatenate(
            (h_ad, np.zeros((count, IMG_HEIGHT, FFT_SIZE_FOR_METRIC - IMG_WIDTH), dtype=np.complex64)),
            axis=2,
        ),
        axis=2,
    )[:, :, :FREQ_BINS].astype(np.complex64)
    return ht, hf


def save_split(output_dir: Path, spec: DatasetSpec, split: str, ht: np.ndarray, hf: np.ndarray | None = None) -> None:
    dataset_dir = output_dir / spec.name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    sio.savemat(dataset_dir / f"DATA_H{split}.mat", {"HT": ht})
    if split == "test" and hf is not None:
        sio.savemat(dataset_dir / "DATA_HtestF_all.mat", {"HF_all": hf})


def generate_all(
    output_dir: Path,
    train_samples: int = 3000,
    val_samples: int = 800,
    test_samples: int = 1000,
    seed: int = 535100,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for spec_idx, spec in enumerate(DATASET_SPECS):
        for split, count, offset in (("train", train_samples, 0), ("val", val_samples, 10_000), ("test", test_samples, 20_000)):
            ht, hf = generate_channel_matrices(spec, count, seed + spec_idx * 1000 + offset)
            save_split(output_dir, spec, split, ht, hf if split == "test" else None)


def load_ht(data_dir: Path, dataset: str, split: str) -> np.ndarray:
    mat = sio.loadmat(data_dir / dataset / f"DATA_H{split}.mat")
    return mat["HT"].astype(np.float32)


def load_hf_test(data_dir: Path, dataset: str) -> np.ndarray:
    mat = sio.loadmat(data_dir / dataset / "DATA_HtestF_all.mat")
    return mat["HF_all"]


def mixed_ht(data_dir: Path, datasets: Iterable[str], split: str, limit_per_dataset: int | None = None) -> np.ndarray:
    arrays = []
    for name in datasets:
        arr = load_ht(data_dir, name, split)
        if limit_per_dataset is not None:
            arr = arr[:limit_per_dataset]
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)

