# Training Strategy

## Stage 1: Offline pretraining from Q7 baseline

Use the existing Q7 CsiNet setup to pretrain a strong single-frame encoder-decoder pair on COST2100 data. This gives a stable initialization and preserves continuity with the mandatory part.

Recommended source:

- `MidTerm_Q7` mixed-dataset model as the initial backbone

Reason:

- the mixed model already improves cross-domain generalization,
- it is a better starting point than a single-scenario model.

## Stage 2: Sequence construction for temporal learning

Because the Q7 dataset is mainly static snapshot CSI, construct synthetic time-varying sequences:

`H_t = rho * H_(t-1) + sqrt(1 - rho^2) * H_random`

Suggested settings:

- `rho = 0.95`: low Doppler
- `rho = 0.80`: medium Doppler
- `rho = 0.50`: high Doppler

This gives a controlled way to test temporal correlation and Doppler robustness.

## Stage 3: Freeze UE encoder, train BS temporal module

Freeze:

- UE encoder

Train:

- BS decoder
- TCN temporal fusion block
- Doppler-adaptive gate
- small refinement head

This keeps UE deployment simple and matches the requirement of low UE-side computational overhead.

## Loss Design

Use a weighted objective:

`L = L_NMSE + beta * L_temp + eta * L_rate`

### Reconstruction loss

`L_NMSE = ||H_t - H_hat_t||_2^2 / ||H_t||_2^2`

Main target: accurate CSI reconstruction.

### Temporal consistency loss

`L_temp = ||(H_hat_t - H_hat_(t-1)) - (H_t - H_(t-1))||_2^2`

This encourages the model to preserve temporal evolution rather than only single-frame fidelity.

### Rate-aware loss

`L_rate = ||z_t - Q(z_t)||_2^2`

This reduces mismatch between floating-point training and quantized feedback deployment.

## Quantization-Aware Training

Insert quantization in the latent feedback path during training:

`z_t -> quantizer -> straight-through estimator -> decoder`

Why it matters:

- CSI feedback is transmitted as bits, not float tensors.
- A proposal that ignores quantization may look strong in simulation but degrade after real feedback encoding.

## Doppler Augmentation

Training batches should mix sequences from different temporal correlations and include abrupt mobility changes. The goal is to teach the fusion gate when history is useful and when it should be suppressed.

## Online Adaptation

Recommended policy: **BS-only online adaptation**.

Keep fixed:

- UE encoder
- main backbone of the decoder

Adapt online:

- gate MLP
- small adapter layer
- normalization statistics

Why:

- avoids UE-side model update overhead,
- lowers signaling and compatibility burden,
- lets the BS react to deployment-specific mobility and scattering changes.
