from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


QPSK_TABLE = {
    (0, 0): (-1 - 1j) / math.sqrt(2),
    (0, 1): (-1 + 1j) / math.sqrt(2),
    (1, 1): (1 + 1j) / math.sqrt(2),
    (1, 0): (1 - 1j) / math.sqrt(2),
}

GRAY_3BIT_TO_LEVEL = {
    (0, 0, 0): -7,
    (0, 0, 1): -5,
    (0, 1, 1): -3,
    (0, 1, 0): -1,
    (1, 1, 0): 1,
    (1, 1, 1): 3,
    (1, 0, 1): 5,
    (1, 0, 0): 7,
}
LEVEL_TO_GRAY_3BIT = {v: k for k, v in GRAY_3BIT_TO_LEVEL.items()}
QAM64_LEVELS = np.array(sorted(LEVEL_TO_GRAY_3BIT.keys()))
QAM64_SCALE = math.sqrt(42)


@dataclass
class OfdmConfig:
    k: int = 64
    cp: int = 16
    num_paths: int = 24
    max_delay: int = 16
    modulation: str = "qpsk"
    pilots: int = 64
    snr_db: int = 20
    with_cp: bool = True
    random_seed: int = 7
    channel_dataset_path: str | None = str(ROOT / "tmp_OFDM_DNN" / "H_dataset_extracted" / "H_dataset")

    @property
    def mu(self) -> int:
        return 2 if self.modulation == "qpsk" else 6

    @property
    def pilot_carriers(self) -> np.ndarray:
        if self.pilots <= 0:
            return np.array([], dtype=int)
        if self.pilots >= self.k:
            return np.arange(self.k)
        return np.arange(0, self.k, self.k // self.pilots, dtype=int)

    @property
    def data_carriers_in_pilot_block(self) -> np.ndarray:
        return np.setdiff1d(np.arange(self.k), self.pilot_carriers)


@dataclass
class RuntimeConfig:
    train_samples_all: int | None = None
    max_iter_all: int | None = None
    part_b_train_samples_64: int = 12000
    part_b_train_samples_8: int = 30000
    part_b_max_iter: int = 100
    part_c_train_samples_64: int = 12000
    part_c_train_samples_8: int = 30000
    part_c_max_iter: int = 100
    part_d_small_train_samples: int = 12000
    part_d_small_max_iter: int = 100
    part_d_full_train_samples: int = 12000
    part_d_full_max_iter: int = 100

    def effective_train_samples(self, default_value: int) -> int:
        return self.train_samples_all if self.train_samples_all is not None else default_value

    def effective_max_iter(self, default_value: int) -> int:
        return self.max_iter_all if self.max_iter_all is not None else default_value


@lru_cache(maxsize=4)
def load_channel_dataset(dataset_path: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(dataset_path)
    train_channels = []
    test_channels = []
    for idx in range(1, 301):
        with (root / f"{idx}.txt").open() as handle:
            for line in handle:
                vals = np.fromstring(line, sep=" ")
                half = vals.size // 2
                train_channels.append(vals[:half] + 1j * vals[half:])
    for idx in range(301, 401):
        with (root / f"{idx}.txt").open() as handle:
            for line in handle:
                vals = np.fromstring(line, sep=" ")
                half = vals.size // 2
                test_channels.append(vals[:half] + 1j * vals[half:])
    return np.asarray(train_channels), np.asarray(test_channels)


def bits_to_symbols(bits: np.ndarray, modulation: str) -> np.ndarray:
    if modulation == "qpsk":
        reshaped = bits.reshape(-1, 2)
        return np.array([QPSK_TABLE[tuple(pair)] for pair in reshaped], dtype=np.complex128)

    reshaped = bits.reshape(-1, 6)
    i_levels = np.array([GRAY_3BIT_TO_LEVEL[tuple(triple)] for triple in reshaped[:, :3]], dtype=float)
    q_levels = np.array([GRAY_3BIT_TO_LEVEL[tuple(triple)] for triple in reshaped[:, 3:]], dtype=float)
    return (i_levels + 1j * q_levels) / QAM64_SCALE


def symbols_to_bits(symbols: np.ndarray, modulation: str) -> np.ndarray:
    if modulation == "qpsk":
        bits = []
        for sym in symbols:
            re = 1 if np.real(sym) >= 0 else 0
            im = 1 if np.imag(sym) >= 0 else 0
            if (re, im) == (1, 0):
                bits.extend((1, 0))
            elif (re, im) == (1, 1):
                bits.extend((1, 1))
            elif (re, im) == (0, 1):
                bits.extend((0, 1))
            else:
                bits.extend((0, 0))
        return np.asarray(bits, dtype=np.int8)

    scaled = symbols * QAM64_SCALE
    bits = []
    for sym in scaled:
        i_level = QAM64_LEVELS[np.argmin(np.abs(np.real(sym) - QAM64_LEVELS))]
        q_level = QAM64_LEVELS[np.argmin(np.abs(np.imag(sym) - QAM64_LEVELS))]
        bits.extend(LEVEL_TO_GRAY_3BIT[int(i_level)])
        bits.extend(LEVEL_TO_GRAY_3BIT[int(q_level)])
    return np.asarray(bits, dtype=np.int8)


def random_channel(cfg: OfdmConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.channel_dataset_path and Path(cfg.channel_dataset_path).exists():
        train_channels, _ = load_channel_dataset(cfg.channel_dataset_path)
        return train_channels[rng.integers(0, len(train_channels))]
    h = np.zeros(cfg.max_delay, dtype=np.complex128)
    delays = rng.integers(0, cfg.max_delay, size=cfg.num_paths)
    gains = (rng.standard_normal(cfg.num_paths) + 1j * rng.standard_normal(cfg.num_paths)) / math.sqrt(2 * cfg.num_paths)
    np.add.at(h, delays, gains)
    norm = np.linalg.norm(h)
    return h / norm if norm > 0 else h


def add_cp(x: np.ndarray, cp: int, with_cp: bool) -> np.ndarray:
    prefix = x[-cp:] if with_cp else np.zeros(cp, dtype=np.complex128)
    return np.concatenate([prefix, x])


def through_channel(x: np.ndarray, h: np.ndarray, snr_db: float, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    y = np.convolve(x, h)
    signal_power = np.mean(np.abs(y) ** 2)
    noise_var = signal_power * (10 ** (-snr_db / 10.0))
    noise = math.sqrt(noise_var / 2.0) * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))
    return y + noise, noise_var


def rx_no_cp(freq_symbol: np.ndarray, h: np.ndarray, cfg: OfdmConfig, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    time_signal = np.fft.ifft(freq_symbol)
    tx = add_cp(time_signal, cfg.cp, cfg.with_cp)
    rx, noise_var = through_channel(tx, h, cfg.snr_db, rng)
    return rx[cfg.cp: cfg.cp + cfg.k], noise_var


def build_frame(cfg: OfdmConfig, rng: np.random.Generator, pilot_symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pilot_block = np.zeros(cfg.k, dtype=np.complex128)
    if cfg.pilots > 0:
        pilot_block[cfg.pilot_carriers] = pilot_symbols
    if cfg.data_carriers_in_pilot_block.size:
        aux_bits = rng.integers(0, 2, size=cfg.data_carriers_in_pilot_block.size * cfg.mu, dtype=np.int8)
        pilot_block[cfg.data_carriers_in_pilot_block] = bits_to_symbols(aux_bits, cfg.modulation)
    data_bits = rng.integers(0, 2, size=cfg.k * cfg.mu, dtype=np.int8)
    data_block = bits_to_symbols(data_bits, cfg.modulation)
    return pilot_block, data_bits, data_block


def sample_dataset(
    cfg: OfdmConfig,
    n_samples: int,
    pred_range: np.ndarray,
    pilot_bits: np.ndarray,
    rng: np.random.Generator,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    pilot_symbols = bits_to_symbols(pilot_bits, cfg.modulation)
    x = np.zeros((n_samples, 4 * cfg.k), dtype=np.float32)
    y = np.zeros((n_samples, len(pred_range)), dtype=np.int8)
    dataset_channels = None
    if cfg.channel_dataset_path and Path(cfg.channel_dataset_path).exists():
        train_channels, test_channels = load_channel_dataset(cfg.channel_dataset_path)
        dataset_channels = train_channels if split == "train" else test_channels
    for idx in range(n_samples):
        h = dataset_channels[rng.integers(0, len(dataset_channels))] if dataset_channels is not None else random_channel(cfg, rng)
        pilot_block, data_bits, data_block = build_frame(cfg, rng, pilot_symbols)
        rx_pilot, _ = rx_no_cp(pilot_block, h, cfg, rng)
        rx_data, _ = rx_no_cp(data_block, h, cfg, rng)
        x[idx] = np.concatenate(
            [np.real(rx_pilot), np.imag(rx_pilot), np.real(rx_data), np.imag(rx_data)]
        )
        y[idx] = data_bits[pred_range]
    return x, y


def train_dnn(
    cfg: OfdmConfig,
    pred_range: np.ndarray,
    pilot_bits: np.ndarray,
    train_samples: int,
    test_samples: int,
    hidden_layers: tuple[int, ...],
    random_seed: int,
    max_iter: int = 80,
    early_stopping: bool = True,
) -> dict:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for train_dnn(). Activate the csinet_tf environment or install tensorflow 2.x."
        ) from exc

    rng_train = np.random.default_rng(random_seed)
    rng_test = np.random.default_rng(random_seed + 1)
    x_train, y_train = sample_dataset(cfg, train_samples, pred_range, pilot_bits, rng_train, split="train")
    x_test, y_test = sample_dataset(cfg, test_samples, pred_range, pilot_bits, rng_test, split="test")
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    tf.keras.backend.clear_session()
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(x_train.shape[1],)))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation="relu"))
    model.add(tf.keras.layers.Dense(y_train.shape[1], activation="sigmoid"))

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss="mse")

    callbacks = []
    validation_split = 0.1 if early_stopping else 0.0
    if early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=8,
                restore_best_weights=True,
            )
        )

    history = model.fit(
        x_train,
        y_train.astype(np.float32),
        batch_size=256,
        epochs=max_iter,
        verbose=0,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True,
    )

    pred_prob = model.predict(x_test, verbose=0)
    pred = (pred_prob >= 0.5).astype(np.int8)
    ber = float(np.mean(pred != y_test))
    loss_curve = history.history.get("loss", [])
    return {
        "ber": ber,
        "loss_curve_len": len(loss_curve),
        "loss_curve": [float(x) for x in loss_curve],
    }


def log_progress(message: str) -> None:
    print(message, flush=True)


def save_training_curve(loss_curve: list[float], title: str, output_path: Path) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(np.arange(1, len(loss_curve) + 1), loss_curve, linewidth=2)
    plt.grid(True, linestyle=":")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_grouped_training_curves(curves: dict[str, list[float]], title: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for label, loss_curve in curves.items():
        plt.plot(np.arange(1, len(loss_curve) + 1), loss_curve, linewidth=2, label=label)
    plt.grid(True, linestyle=":")
    plt.xlabel("Iteration")
    plt.ylabel("Training Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def periodic_linear_interp(k: int, pilot_idx: np.ndarray, pilot_vals: np.ndarray) -> np.ndarray:
    if len(pilot_idx) == k:
        return pilot_vals
    xp = np.concatenate([pilot_idx, [pilot_idx[0] + k]])
    fp_r = np.concatenate([np.real(pilot_vals), [np.real(pilot_vals[0])]])
    fp_i = np.concatenate([np.imag(pilot_vals), [np.imag(pilot_vals[0])]])
    x = np.arange(k)
    return np.interp(x, xp, fp_r) + 1j * np.interp(x, xp, fp_i)


def estimate_channel_tap_covariance(cfg: OfdmConfig, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    if cfg.channel_dataset_path and Path(cfg.channel_dataset_path).exists():
        train_channels, _ = load_channel_dataset(cfg.channel_dataset_path)
        channels = train_channels[:n_channels]
        return (channels.conj().T @ channels) / channels.shape[0]
    hs = np.zeros((n_channels, cfg.max_delay), dtype=np.complex128)
    for idx in range(n_channels):
        hs[idx] = random_channel(cfg, rng)
    return (hs.conj().T @ hs) / n_channels


def baseline_ber(
    cfg: OfdmConfig,
    pred_range: np.ndarray,
    pilot_bits: np.ndarray,
    n_samples: int,
    covariance: np.ndarray,
    random_seed: int,
) -> dict:
    rng = np.random.default_rng(random_seed)
    pilot_symbols = bits_to_symbols(pilot_bits, cfg.modulation)
    pilot_idx = cfg.pilot_carriers
    ber_ls = []
    ber_mmse = []
    if len(pilot_idx) == 0:
        return {"ls": None, "mmse": None}

    ell = np.arange(covariance.shape[0])
    fourier_partial = np.exp(-1j * 2 * np.pi * np.outer(pilot_idx, ell) / cfg.k)
    pilot_matrix = pilot_symbols[:, None] * fourier_partial

    for _ in range(n_samples):
        if cfg.channel_dataset_path and Path(cfg.channel_dataset_path).exists():
            _, test_channels = load_channel_dataset(cfg.channel_dataset_path)
            h = test_channels[rng.integers(0, len(test_channels))]
        else:
            h = random_channel(cfg, rng)
        pilot_block, data_bits, data_block = build_frame(cfg, rng, pilot_symbols)
        rx_pilot_td, noise_var = rx_no_cp(pilot_block, h, cfg, rng)
        rx_data_td, _ = rx_no_cp(data_block, h, cfg, rng)
        y_pilot = np.fft.fft(rx_pilot_td)
        y_data = np.fft.fft(rx_data_td)

        h_ls_pilot = y_pilot[pilot_idx] / pilot_symbols
        h_ls = periodic_linear_interp(cfg.k, pilot_idx, h_ls_pilot)
        sigma_freq = cfg.k * noise_var
        h_mmse_taps = covariance @ pilot_matrix.conj().T @ np.linalg.solve(
            pilot_matrix @ covariance @ pilot_matrix.conj().T + sigma_freq * np.eye(len(pilot_idx)),
            y_pilot[pilot_idx],
        )
        h_mmse = np.fft.fft(np.pad(h_mmse_taps, (0, cfg.k - covariance.shape[0])))

        bits_ls = symbols_to_bits(y_data / h_ls, cfg.modulation)[pred_range]
        bits_mmse = symbols_to_bits(y_data / h_mmse, cfg.modulation)[pred_range]
        target = data_bits[pred_range]
        ber_ls.append(np.mean(bits_ls != target))
        ber_mmse.append(np.mean(bits_mmse != target))

    return {"ls": float(np.mean(ber_ls)), "mmse": float(np.mean(ber_mmse))}


def make_pilot_bits(cfg: OfdmConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.pilots <= 0:
        return np.zeros(0, dtype=np.int8)
    return rng.integers(0, 2, size=cfg.pilots * cfg.mu, dtype=np.int8)


def run_part_b(runtime: RuntimeConfig) -> dict:
    snrs = [5, 10, 15, 20, 25]
    pilot_options = [64, 8]
    pred_range = np.arange(16, 32)
    results = {"metadata": {"pred_range": pred_range.tolist(), "modulation": "qpsk", "training": {}}, "curves": {}, "training_curves": {}}
    cov_rng = np.random.default_rng(123)
    log_progress("[Part b] Start QPSK Figure 3.3 reproduction")

    for pilots in pilot_options:
        cfg = OfdmConfig(modulation="qpsk", pilots=pilots)
        pilot_bits = make_pilot_bits(cfg, np.random.default_rng(1000 + pilots))
        covariance = estimate_channel_tap_covariance(cfg, n_channels=2500, rng=cov_rng)
        curves = {"dnn": [], "ls": [], "mmse": []}
        train_samples = runtime.effective_train_samples(runtime.part_b_train_samples_64 if pilots == 64 else runtime.part_b_train_samples_8)
        max_iter = runtime.effective_max_iter(runtime.part_b_max_iter)
        log_progress(f"[Part b] pilots={pilots}, train_samples={train_samples}, max_iter={max_iter}")
        results["metadata"]["training"][f"pilot_{pilots}"] = {
            "train_samples": train_samples,
            "test_samples": 1500,
            "hidden_layers": [500, 250, 120],
            "max_iter": max_iter,
            "early_stopping": True,
            "batch_size": 256,
            "learning_rate_init": 1e-3,
        }
        for snr in snrs:
            cfg.snr_db = snr
            log_progress(f"[Part b] Training DNN for pilots={pilots}, SNR={snr} dB")
            dnn_out = train_dnn(
                cfg=cfg,
                pred_range=pred_range,
                pilot_bits=pilot_bits,
                train_samples=train_samples,
                test_samples=1500,
                hidden_layers=(500, 250, 120),
                random_seed=snr * 10 + pilots,
                max_iter=max_iter,
                early_stopping=True,
            )
            log_progress(f"[Part b] Finished DNN for pilots={pilots}, SNR={snr} dB, BER={dnn_out['ber']:.6f}, loss_iters={dnn_out['loss_curve_len']}")
            log_progress(f"[Part b] Evaluating LS/MMSE for pilots={pilots}, SNR={snr} dB")
            baseline_out = baseline_ber(
                cfg=cfg,
                pred_range=pred_range,
                pilot_bits=pilot_bits,
                n_samples=600,
                covariance=covariance,
                random_seed=snr * 100 + pilots,
            )
            curves["dnn"].append(dnn_out["ber"])
            curves["ls"].append(baseline_out["ls"])
            curves["mmse"].append(baseline_out["mmse"])
            results["training_curves"][f"pilot_{pilots}_snr_{snr}"] = dnn_out["loss_curve"]
            log_progress(f"[Part b] Baseline done for pilots={pilots}, SNR={snr} dB, LS={baseline_out['ls']:.6f}, MMSE={baseline_out['mmse']:.6f}")
        results["curves"][f"pilot_{pilots}"] = curves

    save_path = RESULTS_DIR / "part_b_results.json"
    save_path.write_text(json.dumps(results, indent=2))
    plot_part_b(results, FIGURES_DIR / "part_b_figure33_like.png")
    save_grouped_training_curves(
        {
            "64 pilots, SNR=20 dB": results["training_curves"]["pilot_64_snr_20"],
            "8 pilots, SNR=20 dB": results["training_curves"]["pilot_8_snr_20"],
        },
        "Part (b) Training Curves",
        FIGURES_DIR / "part_b_training_curves.png",
    )
    log_progress("[Part b] Completed")
    return results


def plot_part_b(results: dict, output_path: Path) -> None:
    snrs = [5, 10, 15, 20, 25]
    plt.figure(figsize=(8, 5))
    styles = {
        ("pilot_64", "dnn"): ("o-", "DNN, 64 pilots"),
        ("pilot_64", "ls"): ("s--", "LS, 64 pilots"),
        ("pilot_64", "mmse"): ("^--", "MMSE, 64 pilots"),
        ("pilot_8", "dnn"): ("o-.", "DNN, 8 pilots"),
        ("pilot_8", "ls"): ("s:", "LS, 8 pilots"),
        ("pilot_8", "mmse"): ("^:", "MMSE, 8 pilots"),
    }
    for key, curves in results["curves"].items():
        for method in ("dnn", "ls", "mmse"):
            marker, label = styles[(key, method)]
            plt.semilogy(snrs, curves[method], marker, label=label)
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Figure 3.3 Reproduction (QPSK)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_part_c(runtime: RuntimeConfig) -> dict:
    snrs = [5, 10, 15, 20, 25]
    pred_range = np.arange(48, 96)
    results = {"metadata": {"pred_range": pred_range.tolist(), "modulation": "64qam", "training": {}}, "curves": {}, "training_curves": {}}
    cov_rng = np.random.default_rng(456)
    log_progress("[Part c] Start 64-QAM experiment")

    for pilots in (64, 8):
        cfg = OfdmConfig(modulation="64qam", pilots=pilots)
        pilot_bits = make_pilot_bits(cfg, np.random.default_rng(2000 + pilots))
        covariance = estimate_channel_tap_covariance(cfg, n_channels=2500, rng=cov_rng)
        curves = {"dnn": [], "ls": [], "mmse": []}
        train_samples = runtime.effective_train_samples(runtime.part_c_train_samples_64 if pilots == 64 else runtime.part_c_train_samples_8)
        max_iter = runtime.effective_max_iter(runtime.part_c_max_iter)
        log_progress(f"[Part c] pilots={pilots}, train_samples={train_samples}, max_iter={max_iter}")
        results["metadata"]["training"][f"pilot_{pilots}"] = {
            "train_samples": train_samples,
            "test_samples": 1000,
            "hidden_layers": [500, 250, 120],
            "max_iter": max_iter,
            "early_stopping": False,
            "batch_size": 256,
            "learning_rate_init": 1e-3,
        }
        for snr in snrs:
            cfg.snr_db = snr
            log_progress(f"[Part c] Training DNN for pilots={pilots}, SNR={snr} dB")
            dnn_out = train_dnn(
                cfg=cfg,
                pred_range=pred_range,
                pilot_bits=pilot_bits,
                train_samples=train_samples,
                test_samples=1000,
                hidden_layers=(500, 250, 120),
                random_seed=snr * 10 + pilots + 500,
                max_iter=max_iter,
                early_stopping=False,
            )
            log_progress(f"[Part c] Finished DNN for pilots={pilots}, SNR={snr} dB, BER={dnn_out['ber']:.6f}, loss_iters={dnn_out['loss_curve_len']}")
            log_progress(f"[Part c] Evaluating LS/MMSE for pilots={pilots}, SNR={snr} dB")
            baseline_out = baseline_ber(
                cfg=cfg,
                pred_range=pred_range,
                pilot_bits=pilot_bits,
                n_samples=600,
                covariance=covariance,
                random_seed=snr * 100 + pilots + 500,
            )
            curves["dnn"].append(dnn_out["ber"])
            curves["ls"].append(baseline_out["ls"])
            curves["mmse"].append(baseline_out["mmse"])
            results["training_curves"][f"pilot_{pilots}_snr_{snr}"] = dnn_out["loss_curve"]
            log_progress(f"[Part c] Baseline done for pilots={pilots}, SNR={snr} dB, LS={baseline_out['ls']:.6f}, MMSE={baseline_out['mmse']:.6f}")
        results["curves"][f"pilot_{pilots}"] = curves

    save_path = RESULTS_DIR / "part_c_results.json"
    save_path.write_text(json.dumps(results, indent=2))
    plot_part_c(results, FIGURES_DIR / "part_c_64qam.png")
    save_grouped_training_curves(
        {
            "64 pilots, SNR=25 dB": results["training_curves"]["pilot_64_snr_25"],
            "8 pilots, SNR=25 dB": results["training_curves"]["pilot_8_snr_25"],
        },
        "Part (c) Training Curves",
        FIGURES_DIR / "part_c_training_curves.png",
    )
    log_progress("[Part c] Completed")
    return results


def plot_part_c(results: dict, output_path: Path) -> None:
    snrs = [5, 10, 15, 20, 25]
    plt.figure(figsize=(8, 5))
    for key, curves in results["curves"].items():
        pilots = key.split("_")[-1]
        plt.semilogy(snrs, curves["dnn"], "o-", label=f"DNN, {pilots} pilots")
        plt.semilogy(snrs, curves["ls"], "s--", label=f"LS, {pilots} pilots")
        plt.semilogy(snrs, curves["mmse"], "^--", label=f"MMSE, {pilots} pilots")
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("64-QAM BER Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_part_d(runtime: RuntimeConfig) -> dict:
    snrs = [5, 10, 15, 20, 25]
    cfg = OfdmConfig(modulation="qpsk", pilots=64)
    pilot_bits = make_pilot_bits(cfg, np.random.default_rng(3000))
    log_progress("[Part d] Start small-vs-full DNN comparison")

    small_pred = np.arange(16, 32)
    full_pred = np.arange(0, 128)
    small_train_samples = runtime.effective_train_samples(runtime.part_d_small_train_samples)
    small_max_iter = runtime.effective_max_iter(runtime.part_d_small_max_iter)
    full_train_samples = runtime.effective_train_samples(runtime.part_d_full_train_samples)
    full_max_iter = runtime.effective_max_iter(runtime.part_d_full_max_iter)
    log_progress(f"[Part d] small_train_samples={small_train_samples}, small_max_iter={small_max_iter}, full_train_samples={full_train_samples}, full_max_iter={full_max_iter}")
    result = {
        "small_dnn": [],
        "full_dnn": [],
        "metadata": {
            "small_dnn_training": {
                "train_samples": small_train_samples,
                "test_samples": 1000,
                "hidden_layers": [500, 250, 120],
                "max_iter": small_max_iter,
                "early_stopping": False,
                "batch_size": 256,
                "learning_rate_init": 1e-3,
            },
            "full_dnn_training": {
                "train_samples": full_train_samples,
                "test_samples": 1000,
                "hidden_layers": [1024, 512, 256],
                "max_iter": full_max_iter,
                "early_stopping": False,
                "batch_size": 256,
                "learning_rate_init": 1e-3,
            },
        },
        "training_curves": {},
    }
    for snr in snrs:
        cfg.snr_db = snr
        log_progress(f"[Part d] Training small DNN, SNR={snr} dB")
        small_out = train_dnn(
            cfg=cfg,
            pred_range=small_pred,
            pilot_bits=pilot_bits,
            train_samples=small_train_samples,
            test_samples=1000,
            hidden_layers=(500, 250, 120),
            random_seed=900 + snr,
            max_iter=small_max_iter,
            early_stopping=False,
        )
        log_progress(f"[Part d] Finished small DNN, SNR={snr} dB, BER={small_out['ber']:.6f}, loss_iters={small_out['loss_curve_len']}")
        log_progress(f"[Part d] Training full DNN, SNR={snr} dB")
        full_out = train_dnn(
            cfg=cfg,
            pred_range=full_pred,
            pilot_bits=pilot_bits,
            train_samples=full_train_samples,
            test_samples=1000,
            hidden_layers=(1024, 512, 256),
            random_seed=1200 + snr,
            max_iter=full_max_iter,
            early_stopping=False,
        )
        log_progress(f"[Part d] Finished full DNN, SNR={snr} dB, BER={full_out['ber']:.6f}, loss_iters={full_out['loss_curve_len']}")
        result["small_dnn"].append(small_out["ber"])
        result["full_dnn"].append(full_out["ber"])
        result["training_curves"][f"small_snr_{snr}"] = small_out["loss_curve"]
        result["training_curves"][f"full_snr_{snr}"] = full_out["loss_curve"]

    save_path = RESULTS_DIR / "part_d_results.json"
    save_path.write_text(json.dumps(result, indent=2))
    plot_part_d(result, FIGURES_DIR / "part_d_full_vs_small.png")
    save_grouped_training_curves(
        {
            "Small 16-bit DNN, SNR=20 dB": result["training_curves"]["small_snr_20"],
            "Full 128-bit DNN, SNR=20 dB": result["training_curves"]["full_snr_20"],
        },
        "Part (d) Training Curves",
        FIGURES_DIR / "part_d_training_curves.png",
    )
    log_progress("[Part d] Completed")
    return result


def plot_part_d(results: dict, output_path: Path) -> None:
    snrs = [5, 10, 15, 20, 25]
    plt.figure(figsize=(8, 5))
    plt.semilogy(snrs, results["small_dnn"], "o-", label="One 16-bit FC-DNN")
    plt.semilogy(snrs, results["full_dnn"], "s-", label="One 128-bit FC-DNN")
    plt.grid(True, which="both", linestyle=":")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("Small vs Full-Output FC-DNN")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Reproduce Exercise 3.1 (b)(c)(d) with configurable train_samples and max_iter.")
    parser.add_argument("--train-samples-all", type=int, default=None, help="Override train_samples for every DNN experiment.")
    parser.add_argument("--max-iter-all", type=int, default=None, help="Override max_iter for every DNN experiment.")
    parser.add_argument("--part-b-train-samples-64", type=int, default=12000, help="train_samples for part (b), 64 pilots.")
    parser.add_argument("--part-b-train-samples-8", type=int, default=30000, help="train_samples for part (b), 8 pilots.")
    parser.add_argument("--part-b-max-iter", type=int, default=80, help="max_iter for part (b).")
    parser.add_argument("--part-c-train-samples", type=int, default=8000, help="train_samples for part (c).")
    parser.add_argument("--part-c-max-iter", type=int, default=40, help="max_iter for part (c).")
    parser.add_argument("--part-d-small-train-samples", type=int, default=8000, help="train_samples for part (d) small 16-bit DNN.")
    parser.add_argument("--part-d-small-max-iter", type=int, default=60, help="max_iter for part (d) small 16-bit DNN.")
    parser.add_argument("--part-d-full-train-samples", type=int, default=8000, help="train_samples for part (d) full 128-bit DNN.")
    parser.add_argument("--part-d-full-max-iter", type=int, default=60, help="max_iter for part (d) full 128-bit DNN.")
    args = parser.parse_args()
    return RuntimeConfig(
        train_samples_all=args.train_samples_all,
        max_iter_all=args.max_iter_all,
        part_b_train_samples_64=args.part_b_train_samples_64,
        part_b_train_samples_8=args.part_b_train_samples_8,
        part_b_max_iter=args.part_b_max_iter,
        part_c_train_samples_64=args.part_c_train_samples,
        part_c_train_samples_8=args.part_c_train_samples,
        part_c_max_iter=args.part_c_max_iter,
        part_d_small_train_samples=args.part_d_small_train_samples,
        part_d_small_max_iter=args.part_d_small_max_iter,
        part_d_full_train_samples=args.part_d_full_train_samples,
        part_d_full_max_iter=args.part_d_full_max_iter,
    )


def main() -> None:
    runtime = parse_args()
    log_progress("[Main] Starting full experiment")
    summary = {
        "part_b": run_part_b(runtime),
        "part_c": run_part_c(runtime),
        "part_d": run_part_d(runtime),
    }
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    log_progress(f"[Main] Saved results to {summary_path}")


if __name__ == "__main__":
    main()
