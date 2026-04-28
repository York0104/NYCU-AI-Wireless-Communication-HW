# Micro-Doppler Shift — Rotary-Wing UAV Sub-6 GHz

Implementation of the channel model, estimation algorithm, and interactive
animated system model from:

> **Hou, H.-A., Wang, L.-C., & Lin, H.-P.** (2021).
> *Micro-Doppler Shift and Its Estimation in Rotary-Wing UAV Sub-6 GHz Communications.*
> IEEE Wireless Communications Letters, 10(10), 2185–2189.
> https://doi.org/10.1109/LWC.2021.3095898
---

## Repository Structure

```
Micro-Doppler-Shift_Rotary-Wing-UAV_Sub6G/
├── system-model.html               # Interactive animated system model (open in browser)
├── microdoppler_channel.py         # Core physics simulation + analytical plots
├── microdoppler_estimation.py      # L-FMCW estimation pipeline + RMSE bound
├── README.md                       # This file
└── [generated outputs]
    ├── microdoppler_results.png
    └── microdoppler_estimation_results.png
```
---

## Quick Start

### 1. Interactive Animation (no dependencies)

```bash
# Simply open in any modern browser
xdg-open system-model.html        # Linux
open system-model.html            # macOS
```

The animation shows:
- Quadcopter UAV with rotating propellers (top-down view)
- Direct path (blue dashed) and reflection path (orange, active only in zone)
- Propeller detail inset with reflection zone (±θ_rz sectors)
- Real-time f_D(t) waveform with cursor
- Live parameter readouts
- Interactive sliders for f_m, f_c, D_p, d_ant, N_b, animation speed

### 2. Python — Channel Simulation

```bash
pip install numpy scipy matplotlib
python microdoppler_channel.py
# → microdoppler_results.png  (~5 s)
```

Produces a 4-panel dark-theme figure:
- Panel 1: Blade angle θ_e(t) with reflection zone markers  [eq. 1]
- Panel 2: Micro-Doppler f_D,r(t)                           [eq. 5]
- Panel 3: |f_D|_max vs carrier frequency f_c
- Panel 4: |f_D|_max vs motor RPM

### 3. Python — Estimation Algorithm

```bash
python microdoppler_estimation.py
# → microdoppler_estimation_results.png  (~15 s)
```

Produces a 3-panel figure:
- Panel 1: True vs estimated f_D,r(t) via L-FMCW + phase-differential
- Panel 2: Reflection path power |â_r|²
- Panel 3: RMS Doppler error bound vs SNR for multiple parameter configs  [eq. 20]

---

## Key Parameters (Table I & II, Hou et al. 2021)

| Symbol | Parameter                  | Default    | Unit |
|--------|----------------------------|------------|------|
| D_p    | Propeller diameter         | 254        | mm   |
| f_m    | Motor revolutions          | 4620       | RPM  |
| N_b    | Blades per propeller       | 2          | —    |
| f_c    | Carrier frequency          | 2.5 / 38   | GHz  |
| d_UE   | UE-to-antenna distance     | 1000       | m    |
| d_ant  | Hub-to-antenna distance    | 200        | mm   |
| RL     | Return loss of propellers  | 5, 15      | dB   |
| f_s    | Sampling frequency         | 30, 60, 480| kSPS |
| SNR    | Signal-to-noise ratio      | 40         | dB   |
| N_avg  | Averaging half-window      | 4          | —    |

---

## Key Equations

**Blade angle**  [eq. 1]:
```
θ_e(t) = 2π f_m t + θ_p(0) + 2mπ/N_b − π/2
```

**Reflection zone** [eq. 4]:
```
θ_rz = arcsin(D_p / (4 d_ant))
     = arcsin(0.254 / (4 × 0.200))  ≈  0.32 rad  ≈  18.4°
```

**Instantaneous micro-Doppler** [eq. 5]:
```
f_D,r(t) = −4π f_m d_ant f_c sin(2θ_e(t)) / c
```

**Maximum |f_D|** (at θ_e = ±π/4):
```
|f_D|_max = 4π f_m d_ant f_c / c
           = 4π × 77 × 0.2 × 2.5×10⁹ / 3×10⁸  ≈  963 Hz
```

**Blade cycle period**:
```
T_blade = 1 / (f_m × N_b)  =  1 / (77 × 2)  ≈  6.494 ms
```

**Reflection duration per window**:
```
Δt_refl = 2 θ_rz T_blade / π  ≈  1.323 ms
```

**Doppler estimation** [eq. 15]:
```
f̂_D,r[n] = f_s · Arg(â_r[n] / â_r[n−1]) / (2π)
```

**Smoothing** [eq. 16]:
```
f̄_D,r[n] = (1/(2N_avg+1)) Σ_{m=−N_avg}^{N_avg} f̂_D,r[n+m]
```

**RMS error bound** [eq. 20]:
```
10 log E(|ε_{f_D}|²) < 20 log(f_s / π) + RL − SNR
```
At (f_s=60kSPS, SNR=40dB, RL=5dB): bound → **18.83 Hz** (1.95% of |f_D|_max).

---

## Dependencies

| Package    | Version tested | Purpose                       |
|------------|---------------|-------------------------------|
| numpy      | ≥ 1.24        | Array maths                   |
| scipy      | ≥ 1.10        | Signal processing              |
| matplotlib | ≥ 3.7         | Dark-theme plotting (Agg)     |

No GPU required. HTML animation requires no Python at all.

---

## Citation

```bibtex
@article{hou2021microdoppler,
  author  = {Hou, Hsin-An and Wang, Li-Chun and Lin, Hsin-Piao},
  title   = {Micro-{D}oppler Shift and Its Estimation in Rotary-Wing
             {UAV} Sub-6 {GHz} Communications},
  journal = {IEEE Wireless Communications Letters},
  volume  = {10},
  number  = {10},
  pages   = {2185--2189},
  year    = {2021},
  doi     = {10.1109/LWC.2021.3095898}
}
```
