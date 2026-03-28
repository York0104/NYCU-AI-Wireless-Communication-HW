# Exercise 2.7: Data-Driven SISO-OFDM Channel Estimation

This project reproduces the data-driven SISO-OFDM channel estimation experiment required in Exercise 2.7, and compares the MSE performance of a DNN-based channel estimator and an LMMSE channel estimator under different SNRs and CP settings.

This implementation corresponds to the two parts of the problem:

- `(a)` Under an OFDM system with cyclic prefix (CP), compare the DNN and LMMSE channel estimators and reproduce the solid-line results in Figure 2.9.
- `(b)` Remove the cyclic prefix (CP) and repeat the same experiment to demonstrate the impact of inter-symbol interference (ISI) on estimation performance, reproducing the dashed-line results.

The final reproduced figure is saved as `figure_2_9_reproduced.png`.

## 1. Problem Mapping and Experimental Objective

This problem requires channel estimation under a SISO-OFDM architecture with the following system settings:

- Number of subcarriers: `64`
- First OFDM symbol: `64` QPSK pilots
- Second OFDM symbol: data symbols
- SNR range: `5 dB` to `40 dB`
- SNR interval: `5 dB`
- Compared methods: `DNN` and `LMMSE`
- Additional condition: compare both `with CP` and `without CP`

Therefore, this project completes the following four experiments:

- DNN with CP
- LMMSE with CP
- DNN without CP
- LMMSE without CP

These four results together form the reproduction of Figure 2.9.

## 2. System Design

### 2.1 DNN Channel Estimator

The DNN channel estimator is implemented in `tools/networks.py` using a multi-layer perceptron (MLP). Its design is as follows:

- Input: concatenated vector of pilot received signal `Yp` and pilot symbols `Xp`
- Hidden layers:
  - `Dense(500, relu)`
  - `Dense(250, relu)`
- Output: real and imaginary parts of the estimated frequency-domain channel
- Loss function: `tf.nn.l2_loss`

This design allows the model to learn the mapping between pilot observations and the channel response directly from data, making it a data-driven channel estimation method.

### 2.2 LMMSE Channel Estimator

The LMMSE channel estimator is implemented in the `MMSE_CE()` function in `tools/raputil.py`. Its main steps are:

- Build an LS initial channel estimate using pilots
- Construct `Rhp` and `Rpp` based on the channel correlation model
- Compute the LMMSE weighting matrix
- Produce the final frequency-domain channel estimate

This is a traditional model-based channel estimation method and serves as the baseline for comparison with the DNN approach.

## 3. Channel Dataset Description

The original public repository did not provide `tools/channel_train.npy` and `tools/channel_test.npy`. Therefore, this project adds `tools/generate_channel_dataset.py` to generate a compatible channel dataset.

The generated dataset has the following properties:

- 16-tap complex Rayleigh channels
- Exponential power-delay profile
- Each channel sample is normalized to stabilize overall signal power

This dataset supports both DNN training and MSE evaluation, allowing the entire experiment pipeline to run completely. However, it should be noted that this dataset is a self-generated compatible version, so its statistical distribution may not be exactly identical to the textbook dataset or the author’s original private dataset.

## 4. Completed Implementation Items

Since the original reference repository was not directly executable, this project completes the following parts:

### 4.1 DNN Channel Estimator Implementation

Completed in `tools/networks.py`:

- input placeholder
- label placeholder
- two-layer MLP architecture
- output layer
- L2 loss

### 4.2 LMMSE Channel Estimator Implementation

Completed in `tools/raputil.py`:

- LS pilot estimation
- channel covariance weighting
- LMMSE estimation matrix
- final channel estimation result

### 4.3 Channel Dataset Generation

Added `tools/generate_channel_dataset.py` to create:

- `tools/channel_train.npy`
- `tools/channel_test.npy`

### 4.4 Command-Line Experiment Workflow

Modified `main.py` so that the full experiment pipeline can be executed using command-line arguments without manually editing the source code.

### 4.5 Result Integration and Plotting

Added `plot_results.py` to plot the four `.mat` result files in a single figure for reporting and analysis.

## 5. Main Files

- `main.py`: main entry point for training and testing
- `plot_results.py`: loads the four `.mat` result files and plots the final figure
- `tools/networks.py`: DNN channel estimator
- `tools/raputil.py`: OFDM utilities, channel simulation, LMMSE, and MSE evaluation
- `tools/generate_channel_dataset.py`: generates `channel_train.npy` and `channel_test.npy`
- `dnn_ce/`: stores DNN weight files under different SNR and CP settings

## 6. Execution Environment

This project was executed in a local virtual environment with the following settings:

- Virtual environment: `.venv`
- Python: `3.11`
- TensorFlow: `2.13.0`
- Usage mode: `tensorflow.compat.v1`

## 7. Experiment Procedure

### 7.1 Activate the Virtual Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 7.2 Generate the Channel Dataset

```powershell
python tools\generate_channel_dataset.py
```

This generates:

- `tools/channel_train.npy`
- `tools/channel_test.npy`

### 7.3 Train the DNN Channel Estimator

With CP:

```powershell
python main.py --ce-type dnn --mode train --cp-flag true --training-epochs 5 --snrs 5 10 15 20 25 30 35 40
```

Without CP:

```powershell
python main.py --ce-type dnn --mode train --cp-flag false --training-epochs 5 --snrs 5 10 15 20 25 30 35 40
```

### 7.4 Evaluate MSE

DNN with CP:

```powershell
python main.py --ce-type dnn --mode test --cp-flag true --num-trials 200 --snrs 5 10 15 20 25 30 35 40
```

LMMSE with CP:

```powershell
python main.py --ce-type mmse --mode test --cp-flag true --num-trials 200 --snrs 5 10 15 20 25 30 35 40
```

DNN without CP:

```powershell
python main.py --ce-type dnn --mode test --cp-flag false --num-trials 200 --snrs 5 10 15 20 25 30 35 40
```

LMMSE without CP:

```powershell
python main.py --ce-type mmse --mode test --cp-flag false --num-trials 200 --snrs 5 10 15 20 25 30 35 40
```

### 7.5 Plot the Final Figure

```powershell
python plot_results.py
```

This generates:

- `figure_2_9_reproduced.png`

## 8. Result Files

After evaluation, the following four result files are generated:

- `MSE_dnn_4QAM.mat`
- `MSE_mmse_4QAM.mat`
- `MSE_dnn_4QAM_CP_FREE.mat`
- `MSE_mmse_4QAM_CP_FREE.mat`

## 9. Result Analysis

The reproduced MSE curves are shown below.

![Reproduced Figure 2.9](./figure_2_9_reproduced.png)

The MSE curves obtained in this experiment show trends consistent with the problem requirements:

- Under the `with CP` setting, `LMMSE` improves steadily as SNR increases and gives the best performance.
- Under the `without CP` setting, both `DNN` and `LMMSE` experience performance degradation due to additional interference introduced by ISI.
- The CP-free `LMMSE` curve deteriorates at high SNR, indicating that the system becomes interference-limited rather than noise-limited.
- The DNN channel estimator works under both CP and CP-free settings, although its performance is more sensitive to the number of training epochs and the generated channel dataset.

These results satisfy the qualitative requirements of the homework:

- Reproduce the solid-line behavior when CP is used
- Reproduce the dashed-line behavior when CP is removed
