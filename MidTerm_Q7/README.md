# Exercise 2.15: CSI Compression and Reconstruction using CsiNet and CS-CsiNet

This repository implements two neural network models (CsiNet and CS-CsiNet) for Channel State Information (CSI) compression and reconstruction in wireless communication systems. Both models support indoor/outdoor wireless environments, adjustable compression rates, and provide complete training & inference pipelines with performance evaluation and visualization.

## Completed Exercise 2.15 Solution

This workspace also includes a complete Exercise 2.15 experiment pipeline:

| File | Purpose |
|------|---------|
| `scripts/exercise_2_15_datasets.py` | Generates six COST-2100-style channel datasets with different user distributions. |
| `scripts/run_exercise_2_15_tf.py` | Runs the TensorFlow/Keras CsiNet single-dataset and mixed-dataset experiment. |
| `scripts/run_exercise_2_15_fast.py` | Runs a CPU-only PCA autoencoder proxy to verify the full pipeline when TensorFlow is unavailable. |
| `reports/exercise_2_15_report.md` | Chinese/English-ready report content for parts (a), (b), and (c), including result tables and discussion. |
| `result/exercise_2_15_fast_results.csv` | CPU-only proxy verification results. |
| `result/exercise_2_15_csinet_results.csv` | TensorFlow CsiNet results generated in the `csinet_tf` Conda environment. |

Current environment note: the base Python is 3.13, so a separate Conda environment named `csinet_tf` was created with Python 3.10 and TensorFlow 2.10.1. The TensorFlow runner uses `channels_last` internally because Windows CPU TensorFlow does not support `channels_first` Conv2D backpropagation.

Quick local verification:

```powershell
python scripts/run_exercise_2_15_fast.py --generate --train-samples 1200 --val-samples 300 --test-samples 400 --encoded-dim 128 --mix-limit 1200
```

TensorFlow CsiNet run:

```powershell
conda run -n csinet_tf python scripts/run_exercise_2_15_tf.py --encoded-dim 512 --epochs 20 --batch-size 100 --mix-limit 1200 --val-limit 300
```

## What You Need to Do

| Step | Task | Details |
| :---: | :--- | :--- |
| 1 | **Code Preparation** | Ensure you have installed the required dependencies (TensorFlow 1.x or 2.x, Keras, NumPy, SciPy, Matplotlib). Create directories `data`, `result`, and `saved_model`. |
| 2 | **Data Preparation** | Place MATLAB-formatted CSI datasets and random projection matrices in the `data/` directory. |
| 3 | **Model Training** | Run `CsiNet_train.py` and `CS-CsiNet_train.py` to train the models and save their architectures/weights to the `result/` folder. |
| 4 | **Model Inference** | Copy pre-trained model files to `saved_model/` and run `CsiNet_onlytest.py` or `CS-CsiNet_onlytest.py` to evaluate reconstruction performance. |

## File Structure

| File / Directory | Purpose |
|------|---------|
| `CsiNet_train.py` | CsiNet training pipeline: model building, training, evaluation, saving. |
| `CsiNet_onlytest.py` | CsiNet inference only: load pre-trained model, reconstruct CSI, evaluate. |
| `CS-CsiNet_train.py` | CS-CsiNet training pipeline: train decoder with fixed CS encoder. |
| `CS-CsiNet_onlytest.py` | CS-CsiNet inference only: load decoder, reconstruct CSI from compressed data. |
| `data/` | Directory containing MATLAB-formatted CSI datasets and random projection matrices. |
| `result/` | Output directory containing training logs, loss curves, reconstructed data, and model checkpoints. |
| `saved_model/` | Output directory containing pre-trained model architectures (`.json`) and weights (`.h5`) for inference. |

---

## Detailed Task Breakdown

### Part 1: CsiNet Training and Inference
An end-to-end residual-based autoencoder for CSI compression/reconstruction:
*   **Encoder:** Conv2D → Flatten → Dense (adaptive CSI feature compression). Fully trainable without needing a pre-defined compression matrix.
*   **Decoder:** Dense → Reshape → Stacked Residual Blocks → Conv2D (CSI reconstruction).

### Part 2: CS-CsiNet Training and Inference
A compressed sensing enhanced version of CsiNet:
*   **Encoder:** Fixed random projection matrix (non-trainable CS projection). Lightweight, no training needed.
*   **Decoder:** Residual-based structure (trainable for CSI reconstruction).

---

## Key Parameters (Modify at script top)

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `envir` | `indoor` | Wireless environment (`indoor`/`outdoor`). |
| `img_height/width` | `32/32` | CSI matrix spatial/frequency dimensions. |
| `img_channels` | `2` | CSI channels (real + imaginary parts). |
| `residual_num` | `2` | Number of residual blocks in decoder. |
| `encoded_dim` | `512` | Compression dimension (512=1/4, 128=1/16, 64=1/32, 32=1/64). |

---

## Performance Metrics & Important Notes

**Performance Metrics**
*   **Normalized Mean Square Error (NMSE, dB):** `10*log10(E[|CSI_orig - CSI_rec|²]/E[|CSI_orig|²])`. Smaller (more negative) = better reconstruction.
*   **Correlation Coefficient (rho):** Range: [0,1], closer to 1 = higher similarity between original/reconstructed CSI.

**Important Notes**
1. Inference scripts must use the same `envir` and `encoded_dim` as the pre-trained model.
2. CS-CsiNet requires matching random projection matrix (`A{encoded_dim}.mat`).
3. Remove `tf.reset_default_graph()` for TensorFlow 2.x compatibility.
4. Code uses "channels_first" data format (do not modify without adjusting network).
5. Default training epochs: 1000 (adjust based on dataset/hardware).
