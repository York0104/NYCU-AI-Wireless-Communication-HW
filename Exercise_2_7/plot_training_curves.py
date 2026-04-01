from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SNR_DB = [5, 10, 15, 20, 25, 30, 35, 40]


def load_loss(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    loss_history = np.asarray(data["loss_history"]).reshape(-1)
    test_step = int(np.asarray(data["test_step"]).item())
    return loss_history, test_step


def plot_one_group(prefix: str, title: str, output_name: str):
    plt.figure(figsize=(9, 6))

    found = False
    for snr in SNR_DB:
        npz_path = Path("dnn_ce") / f"{prefix}{snr}dB.npz"
        if not npz_path.exists():
            print(f"Skip missing file: {npz_path}")
            continue

        loss_history, test_step = load_loss(npz_path)
        epochs = np.arange(len(loss_history)) * test_step
        plt.plot(epochs, loss_history, linewidth=2, label=f"SNR={snr} dB")
        found = True

    if not found:
        print(f"No files found for prefix: {prefix}")
        return

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title(title)
    plt.grid(True, linestyle=":", linewidth=0.7)
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_name)
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path.resolve()}")
    plt.close()


def main():
    plot_one_group(
        prefix="CE_DNN_4QAM_SNR_",
        title="Training Curve of DNN Channel Estimator (with CP)",
        output_name="training_curve_with_cp.png",
    )

    plot_one_group(
        prefix="CE_DNN_CPFREE_4QAM_SNR_",
        title="Training Curve of DNN Channel Estimator (without CP)",
        output_name="training_curve_without_cp.png",
    )


if __name__ == "__main__":
    main()