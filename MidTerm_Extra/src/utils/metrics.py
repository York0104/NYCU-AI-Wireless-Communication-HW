from __future__ import annotations

import math

import numpy as np


def complex_from_csi_image(x: np.ndarray) -> np.ndarray:
    real = x[:, :, :, 0].reshape(len(x), -1) - 0.5
    imag = x[:, :, :, 1].reshape(len(x), -1) - 0.5
    return real + 1j * imag


def nmse_db(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_c = complex_from_csi_image(y_true)
    pred_c = complex_from_csi_image(y_pred)
    power = np.sum(np.abs(true_c) ** 2, axis=1)
    mse = np.sum(np.abs(true_c - pred_c) ** 2, axis=1)
    return 10.0 * math.log10(float(np.mean(mse / np.maximum(power, 1e-12))))


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_c = complex_from_csi_image(y_true)
    pred_c = complex_from_csi_image(y_pred)
    num = np.abs(np.sum(np.conj(true_c) * pred_c, axis=1))
    den = np.sqrt(np.sum(np.abs(true_c) ** 2, axis=1)) * np.sqrt(np.sum(np.abs(pred_c) ** 2, axis=1))
    return float(np.mean(num / np.maximum(den, 1e-12)))
