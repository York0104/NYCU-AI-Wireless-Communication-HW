# Q7 Extra Credit Report: DA-TCFNet

## 1. Proposed Architecture

To replace `CsiNet-LSTM`, I propose **DA-TCFNet (Doppler-Adaptive Temporal Convolutional Feedback Network)**. The design principle is simple: keep the UE lightweight and move temporal intelligence to the BS. Compared with `CsiNet-LSTM`, the proposed model still exploits temporal correlation, but it avoids recurrent processing on the terminal side and explicitly adapts to different Doppler conditions.

The end-to-end pipeline is:

$$
\mathbf{H}_t
\rightarrow
\text{angular-delay transform}
\rightarrow
\text{lightweight UE encoder}
\rightarrow
\mathbf{z}_t
\rightarrow
\text{quantization / feedback}
\rightarrow
\text{BS temporal fusion}
\rightarrow
\hat{\mathbf{H}}_t
$$

### 1.1 UE-side modules

The UE is responsible only for single-frame compression and a tiny mobility hint.

First, the current CSI `H_t` is transformed into the angular-delay domain. This is consistent with the CsiNet representation already used in Q7 and matches the sparse structure of massive MIMO CSI.

Next, the UE uses a lightweight encoder:

`complex mapping -> depthwise separable conv -> 1x1 conv -> squeeze-excitation -> FC bottleneck`

This encoder outputs the compressed latent vector `z_t`. Unlike `CsiNet-LSTM`, there is no LSTM, TCN, or Transformer on the UE. Therefore, UE-side FLOPs, memory footprint, and inference latency are all reduced.

Finally, the UE computes a small temporal correlation indicator

$$
\rho_t =
\frac{
\left|
\left\langle \mathbf{H}_t, \mathbf{H}_{t-1} \right\rangle
\right|
}{
\|\mathbf{H}_t\|_2 \, \|\mathbf{H}_{t-1}\|_2
}
$$

and feeds back a quantized version of it using only a few bits. This gives the BS a direct hint about whether the channel is slowly varying or rapidly changing.

### 1.2 BS-side modules

After receiving `z_t`, the BS first reconstructs a coarse CSI estimate:

$$
\tilde{\mathbf{H}}_t = D(\mathbf{z}_t)
$$

The BS also stores recent latent features in a temporal memory bank

$$
\mathcal{M}_t = \left\{ \mathbf{f}_{t-1}, \mathbf{f}_{t-2}, \dots, \mathbf{f}_{t-L} \right\}
$$

and processes them with a **dilated temporal convolution network (TCN)** instead of an LSTM. This is the key replacement of `CsiNet-LSTM`. The TCN offers a larger effective temporal receptive field and allows parallel computation, so it is more deployment-friendly at the BS.

To improve robustness under different mobility conditions, the BS uses a **Doppler-adaptive fusion gate**:

$$
\alpha_t = \sigma\!\left(\mathrm{MLP}\left([\mathbf{z}_t, \mathbf{f}_t, \rho_t]\right)\right)
$$

$$
\mathbf{z}_{\mathrm{fused}} =
\alpha_t \mathbf{z}_t + (1-\alpha_t)\mathbf{f}_t
$$

If Doppler is low, the gate relies more on temporal features because past CSI is still informative. If Doppler is high, the gate shifts its trust toward the current latent `z_t` and suppresses stale historical information. This directly targets the weakness of fixed temporal processing under fast fading.

Finally, a residual refinement decoder produces the final estimate:

$$
\hat{\mathbf{H}}_t =
\tilde{\mathbf{H}}_t + R\!\left(\tilde{\mathbf{H}}_t, \mathbf{z}_{\mathrm{fused}}\right)
$$

This residual path is important because it prevents the model from over-trusting history. Even under severe Doppler spread, the model can still fall back to current-frame reconstruction.

### 1.3 Why this architecture satisfies the three requirements

For **temporal correlation**, the BS-side TCN captures latent sequence evolution across multiple time steps. For **low UE computational overhead**, the UE only performs single-frame compression with a lightweight encoder and does not run any temporal network. For **Doppler robustness**, the adaptive fusion gate reduces the effect of stale history when temporal correlation drops quickly.

## 2. Training Strategy

The training strategy is designed to reuse the existing Q7 work instead of starting from scratch.

### 2.1 Offline pretraining

First, use the Q7 CsiNet model as the single-frame backbone. In particular, the mixed-dataset model from `MidTerm_Q7` is a good initialization because it already shows better cross-dataset generalization than the single-dataset baseline. This stage pretrains the UE encoder and BS decoder using standard reconstruction loss.

### 2.2 Temporal sequence construction

Since the Q7 dataset mainly contains static CSI snapshots, synthetic time-varying sequences can be created by

$$
\mathbf{H}_t =
\rho \mathbf{H}_{t-1} + \sqrt{1-\rho^2}\,\mathbf{H}_{\mathrm{random}}
$$

where `rho` controls temporal correlation. For example:

- `rho = 0.95`: low Doppler
- `rho = 0.80`: medium Doppler
- `rho = 0.50`: high Doppler

This is a practical way to emulate different Doppler spread conditions without needing a fully new data-collection pipeline.

### 2.3 BS-heavy temporal training

After pretraining, freeze the UE encoder and train only the BS-side modules:

- TCN temporal block
- Doppler-adaptive gate
- decoder refinement head

This matches the main design goal: all temporal intelligence stays at the BS, while the UE model remains simple and stable.

### 2.4 Quantization-aware training

Because real CSI feedback is transmitted as bits, not floating-point vectors, quantization-aware training should be used:

$$
\mathbf{z}_t
\rightarrow
\text{quantizer}
\rightarrow
\text{STE}
\rightarrow
\text{decoder}
$$

This reduces the mismatch between training and deployment and makes the proposal more realistic than a float-only latent feedback assumption.

### 2.5 Loss function

A suitable loss is

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{NMSE}}
+
\beta \mathcal{L}_{\mathrm{temp}}
+
\eta \mathcal{L}_{\mathrm{rate}}
$$

where

- `L_NMSE` measures reconstruction quality,
- `L_temp` preserves temporal evolution across adjacent CSI frames,
- `L_rate` penalizes quantization mismatch in the latent vector.

This objective encourages both accurate recovery and stable temporal behavior.

### 2.6 Online adaptation

I do **not** recommend online adaptation on the UE. If online update is used, it should be restricted to the BS, such as:

- updating gate parameters,
- updating a small adapter layer,
- refreshing normalization statistics.

This policy keeps the UE overhead low and avoids additional signaling or compatibility issues.

## 3. Ablation Studies

At least two ablations are required, but four are more convincing.

### 3.1 Ablation 1: Does temporal modeling help?

Compare:

- `CsiNet`
- `DA-TCFNet w/o TCN`
- `DA-TCFNet`

under `rho = 0.95`, `0.80`, and `0.50`.

This verifies whether the BS-side temporal module truly exploits temporal correlation. The expected trend is:

`CsiNet < DA-TCFNet w/o TCN < DA-TCFNet`

especially under low and medium Doppler.

### 3.2 Ablation 2: TCN versus LSTM

Compare:

- `CsiNet-LSTM`
- `DA-TCFNet` with BS-side TCN

using NMSE, parameter count, and inference latency.

This directly answers the question of whether the new architecture is a reasonable replacement for `CsiNet-LSTM`. The expected conclusion is that TCN gives comparable or better reconstruction quality while being more parallelizable and efficient at the BS.

### 3.3 Ablation 3: Does the Doppler-aware gate matter?

Compare:

- TCN without gate
- TCN with fixed fusion ratio
- full `DA-TCFNet`

under low, medium, high, and abrupt Doppler conditions.

The goal is to show that robustness at high Doppler does not come only from temporal fusion, but specifically from **adaptive** temporal fusion.

### 3.4 Ablation 4: BS-only online adaptation

Compare:

- offline-only model
- BN-statistics update
- small adapter tuning

under a domain-shifted test set. This validates whether a limited BS-side update can improve deployment robustness without modifying the UE.

## 4. Experimental Results

All results below use the final submission setting:

- `time_steps = 10`
- `encoded_dim = 128`
- `epochs = 100`
- `batch_size = 64`
- `rho = 0.95, 0.80, 0.50`
- `train / val / test limit = 100 / 50 / 50` per dataset

### 4.1 Proposed Model Result

The final `DA-TCFNet` test result is:

| Model | Time Steps | Encoded Dim | Epochs | NMSE (dB) | Cosine Similarity | Trainable Params | Inference Time / Sample (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `DA-TCFNet` | 10 | 128 | 100 | -21.5424 | 0.996646 | 1,471,708 | 0.00141381 |

The model converges stably over 100 epochs. The training loss drops from `0.1860` to `0.000623`, and the validation loss drops from `0.1376` to `0.000940`, which indicates that the proposed model can fit the synthetic temporal CSI reconstruction task without unstable behavior.

<img src="../figure/da_tcfnet_training_curve.png" alt="DA-TCFNet Training Curve" width="700">

*Figure: Training and validation loss curves of the proposed DA-TCFNet over 100 epochs.*

### 4.2 Three-Way Ablation

To make the comparison fair, the three models are trained and tested on the same generated sequence dataset:

- `CsiNet`
- `CsiNet-LSTM`
- `DA-TCFNet`

The final results are:

| Model | rho | NMSE (dB) | Cosine Similarity | Trainable Params | Inference Time / Sample (s) |
|---|---:|---:|---:|---:|---:|
| `CsiNet` | 0.50 | -15.0813 | 0.985575 | 1,318,114 | 0.00066901 |
| `CsiNet` | 0.80 | -27.2398 | 0.999071 | 1,318,114 | 0.00069716 |
| `CsiNet` | 0.95 | -35.5841 | 0.999874 | 1,318,114 | 0.00070372 |
| `DA-TCFNet` | 0.50 | -18.4157 | 0.993339 | 1,471,708 | 0.00148995 |
| `DA-TCFNet` | 0.80 | -27.1283 | 0.999123 | 1,471,708 | 0.00148664 |
| `DA-TCFNet` | 0.95 | -32.9461 | 0.999837 | 1,471,708 | 0.00167859 |
| `CsiNet-LSTM` | 0.50 | -16.0370 | 0.987763 | 1,482,722 | 0.00126966 |
| `CsiNet-LSTM` | 0.80 | -27.3505 | 0.999092 | 1,482,722 | 0.00126352 |
| `CsiNet-LSTM` | 0.95 | -35.3370 | 0.999873 | 1,482,722 | 0.00127245 |

<img src="../figure/ablation_nmse_grouped.png" alt="Ablation NMSE Grouped" width="700">

*Figure: Grouped NMSE comparison of CsiNet, CsiNet-LSTM, and DA-TCFNet under different temporal correlation settings.*

<img src="../figure/ablation_nmse_line.png" alt="NMSE versus Temporal Correlation" width="700">

*Figure: NMSE trend of the three models as the temporal correlation coefficient changes.*

<img src="../figure/ablation_latency.png" alt="Ablation Latency" width="650">

*Figure: Inference latency comparison among the three competing models.*

### 4.3 Result Interpretation

The main finding is that `DA-TCFNet` is strongest in the high-Doppler setting. At `rho = 0.50`, it achieves `-18.4157 dB`, which is better than both `CsiNet` and `CsiNet-LSTM`. This supports the idea that a Doppler-aware temporal fusion mechanism can be helpful when the channel changes more quickly.

At `rho = 0.80`, all three models are very close. The difference is small enough that the main conclusion is not about absolute dominance, but about comparable medium-Doppler performance.

At `rho = 0.95`, `CsiNet` and `CsiNet-LSTM` perform slightly better than `DA-TCFNet`. This suggests that when temporal correlation is already extremely strong, the current fusion strategy of `DA-TCFNet` may be too conservative and may not extract the full possible gain from historical CSI.

<img src="../figure/sequence_sample_counts.png" alt="Sequence Sample Counts" width="650">

*Figure: Number of generated temporal sequences in the train, validation, and test splits.*

<img src="../figure/sequence_rho_counts.png" alt="Sequence Rho Counts" width="650">

*Figure: Number of generated sequences under each temporal correlation setting (`rho = 0.95, 0.80, 0.50`).*

<img src="../figure/temporal_correlation_settings.png" alt="Temporal Correlation Settings" width="700">

*Figure: Illustration of how the three temporal correlation settings decay with time lag.*

<img src="../figure/reconstruction_comparison.png" alt="Reconstruction Comparison" width="750">

*Figure: Qualitative reconstruction comparison among Original CSI, CsiNet, CsiNet-LSTM, and DA-TCFNet at different Doppler conditions.*

### 4.4 Figure Support

The generated figures are stored in `MidTerm_Extra/figure/` and include:

- `da_tcfnet_training_curve.png`
- `ablation_nmse_grouped.png`
- `ablation_nmse_line.png`
- `ablation_latency.png`
- `sequence_sample_counts.png`
- `sequence_rho_counts.png`
- `temporal_correlation_settings.png`
- `reconstruction_comparison.png`

These figures support both the numerical ablation and the qualitative interpretation of the proposed architecture.

## 5. Expected Contributions

This proposal improves over `CsiNet-LSTM` in three ways. First, it still uses temporal correlation, but through a BS-side TCN rather than recurrent processing. Second, it reduces UE complexity by keeping the UE strictly single-frame and lightweight. Third, it improves robustness under high Doppler through a Doppler-adaptive fusion gate that avoids over-reliance on stale CSI history.

In short, **DA-TCFNet is a more deployment-oriented temporal CSI feedback architecture**: UE-light, BS-heavy, Doppler-aware, and easy to validate with clear ablation studies built on top of the existing Q7 CsiNet pipeline.

## 6. How It Connects to Existing Q7 Results

The Q7 mandatory part already showed that a mixed-dataset CsiNet significantly improves cross-dataset NMSE over a single-dataset model. That result suggests two important lessons:

1. CSI feedback models are sensitive to channel distribution shift.
2. Better robustness comes from explicitly modeling variability instead of training on one narrow setting.

DA-TCFNet extends this idea from **spatial/domain diversity** to **temporal/mobility diversity**. The mixed-dataset Q7 model can therefore serve as the pretrained single-frame backbone, while the extra-credit part adds temporal sequence modeling and Doppler-aware fusion on top of it.
