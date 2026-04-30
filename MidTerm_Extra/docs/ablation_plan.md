# Ablation Plan

## Metrics

Use the following evaluation metrics:

- NMSE (dB)
- correlation coefficient `rho`
- UE FLOPs
- UE parameter count
- UE inference latency
- total feedback bits

## Ablation 1: Temporal module effectiveness

### Goal

Verify that BS-side temporal modeling truly improves over single-frame CSI compression.

### Models

- `CsiNet`: no temporal modeling
- `DA-TCFNet w/o TCN`: current-frame path only
- `DA-TCFNet`: full model with TCN and gate

### Expected result

`DA-TCFNet` should outperform single-frame CsiNet under low and medium Doppler, showing that temporal correlation is being used effectively.

## Ablation 2: TCN versus LSTM

### Goal

Show that the proposed temporal block is a reasonable replacement for `CsiNet-LSTM`.

### Models

- `CsiNet` single-frame baseline
- `CsiNet-LSTM`
- `BS-side TCN`

### Compare

- NMSE
- parameter count
- inference time

### Expected result

The TCN should match or slightly exceed LSTM reconstruction quality while reducing latency through parallel temporal convolution.

## Ablation 3: Doppler-aware gate effectiveness

### Goal

Verify that high-Doppler robustness comes from the adaptive gate rather than from temporal fusion alone.

### Models

- `TCN without gate`
- `TCN with fixed fusion weight`
- `DA-TCFNet full model`

### Test conditions

- `rho = 0.95`
- `rho = 0.80`
- `rho = 0.50`
- abrupt mobility change

### Expected result

The full model should be most stable at `rho = 0.50` because it can shift weight back to the current latent vector when historical CSI becomes stale.

## Ablation 4: BS-only online adaptation

### Goal

Test whether limited BS-side adaptation helps under domain shift without changing the UE model.

### Models

- offline only
- BN-statistics update only
- adapter tuning

### Expected result

Small BS-side adaptation should recover part of the performance loss under deployment shift, while keeping the UE architecture unchanged.
