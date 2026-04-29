from pathlib import Path

import matplotlib
import numpy as np
import scipy.io as sio

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SNR_DB = np.array([5, 10, 15, 20, 25, 30, 35, 40], dtype=np.int32)


def load_curve(mat_path: str, key: str) -> np.ndarray:
    data = sio.loadmat(mat_path)[key]
    return np.asarray(data).reshape(-1)


def main() -> None:
    dnn_cp = load_curve("MSE_dnn_4QAM.mat", "MSE_dnn_4QAM")
    mmse_cp = load_curve("MSE_mmse_4QAM.mat", "MSE_mmse_4QAM")
    dnn_nocp = load_curve("MSE_dnn_4QAM_CP_FREE.mat", "MSE_dnn_4QAM_CP_FREE")
    mmse_nocp = load_curve("MSE_mmse_4QAM_CP_FREE.mat", "MSE_mmse_4QAM_CP_FREE")

    plt.figure(figsize=(9, 6))
    plt.semilogy(SNR_DB, dnn_cp, "o-", linewidth=2, markersize=6, label="DNN with CP")
    plt.semilogy(SNR_DB, mmse_cp, "s-", linewidth=2, markersize=6, label="LMMSE with CP")
    plt.semilogy(SNR_DB, dnn_nocp, "o--", linewidth=2, markersize=6, label="DNN without CP")
    plt.semilogy(SNR_DB, mmse_nocp, "s--", linewidth=2, markersize=6, label="LMMSE without CP")

    plt.xlabel("SNR (dB)")
    plt.ylabel("MSE")
    plt.title("Exercise 2.7: SISO-OFDM Channel Estimation")
    plt.xticks(SNR_DB)
    plt.grid(True, which="both", linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()

    output_path = Path("figure_2_9_reproduced.png")
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
