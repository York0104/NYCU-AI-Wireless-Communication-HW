# DA-TCFNet Architecture

## Motivation

`CsiNet-LSTM` uses temporal correlation, but its recurrent processing is sequential and not ideal when we want low UE overhead and better robustness under fast channel variation. The proposed **DA-TCFNet** moves temporal intelligence to the BS and keeps the UE lightweight.

## End-to-End Pipeline

`H_t -> angular-delay transform -> lightweight UE encoder -> z_t -> quantization/feedback -> BS temporal fusion -> decoder -> H_hat_t`

## UE Side

The UE only handles current-frame processing.

### 1. Angular-delay preprocessing

The input CSI is mapped to the angular-delay domain before neural compression. This follows the CsiNet-style representation already used in Q7 and leverages the sparse structure of massive MIMO channels.

### 2. Lightweight encoder

The encoder replaces the standard heavier CsiNet feature extractor with a lightweight block:

`complex input mapping -> depthwise separable conv -> pointwise conv -> squeeze-excitation -> FC bottleneck`

Design goal:

- keep the encoder single-frame only,
- avoid LSTM or Transformer blocks on the UE,
- reduce FLOPs, memory, and inference latency.

### 3. Doppler indicator

The UE additionally feeds back a tiny mobility indicator, for example a quantized temporal correlation score

`rho_t = |<H_t, H_(t-1)>| / (||H_t||_2 ||H_(t-1)||_2)`

This can be quantized into a few bits and sent with negligible overhead.

## BS Side

The BS is responsible for all temporal modeling and refinement.

### 1. Coarse decoder

The BS first reconstructs a coarse CSI estimate from the current codeword:

`H_t_tilde = D(z_t)`

### 2. Temporal latent memory

The BS stores recent latent vectors or decoder-side features:

`M_t = {f_(t-1), f_(t-2), ..., f_(t-L)}`

### 3. Dilated temporal convolution module

Instead of an LSTM, the BS uses a **dilated temporal convolution network (TCN)** over the latent sequence. This gives:

- parallel processing across time,
- longer effective receptive field than a shallow recurrent block,
- lower inference latency for batch processing at the BS.

### 4. Doppler-adaptive fusion gate

The BS predicts a fusion weight `alpha_t` using the current latent, temporal feature, and Doppler indicator:

`alpha_t = sigmoid(MLP([z_t, f_t, rho_t]))`

Fusion:

`z_fused = alpha_t * z_t + (1 - alpha_t) * f_t`

Behavior:

- low Doppler: rely more on temporal feature `f_t`
- high Doppler: rely more on current codeword `z_t`

This directly addresses the failure mode of stale history under fast fading.

### 5. Residual refinement decoder

The final CSI estimate is produced by a residual decoder:

`H_hat_t = H_t_tilde + R(H_t_tilde, z_fused)`

This guarantees a safe fallback path: even if temporal information is unreliable, the current-frame reconstruction path still exists.

## Why This Replaces CsiNet-LSTM Well

- It still exploits temporal correlation, but without recurrent UE-side computation.
- It makes the architecture BS-heavy, which is more practical for deployment.
- It adapts the temporal reliance according to Doppler conditions instead of using a fixed recurrent memory behavior.
