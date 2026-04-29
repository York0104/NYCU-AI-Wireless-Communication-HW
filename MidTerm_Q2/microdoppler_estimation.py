"""
Micro-Doppler Estimation via L-FMCW Sounding Waveform
======================================================

Implements the estimation algorithm from:

    Hou, H.-A., Wang, L.-C., & Lin, H.-P. (2021).
    "Micro-Doppler Shift and Its Estimation in Rotary-Wing UAV
    Sub-6 GHz Communications."
    IEEE Wireless Communications Letters, 10(10), 2185–2189.
    https://doi.org/10.1109/LWC.2021.3095898

Algorithm steps (Section III)
------------------------------
  1. Generate L-FMCW sounding waveform x[n]           (Sec. III-A)
  2. Correlate received y[n] with x[n] to sync         [eq. 11–13]
  3. Isolate reflection path response â_r[n] e^{jΔθ_r} [eq. 14]
  4. Estimate Doppler via phase differential            [eq. 15]
  5. Smooth with 2N_avg+1 window                        [eq. 16]

Runtime: ~5–15 s depending on fs and sim duration.

Author : Timothius Victorio Yasin
Advisor: Prof. Li-Chun Wang, NYCU
"""
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from microdoppler_channel import (
    DEFAULT_PARAMS, blade_angle, reflection_zone_angle,
    microdoppler_freq, max_microdoppler, compute_doppler_waveform,
    _wrap_pi, _apply_dark_style, COLORS, C_LIGHT,
)
from typing import Dict

# ── L-FMCW Sounding Waveform ────────────────────────────────────────────────

def generate_lfmcw(N: int, fs: float, B: float = 10e3) -> np.ndarray:
    """Linear FM Continuous Wave (L-FMCW) sounding waveform.

    Chosen because it has the best combined time- and frequency-domain
    resolution for micro-Doppler detection (Sec. III-A, Hou et al. 2021).

    Parameters
    ----------
    N  : number of samples
    fs : sampling frequency  [Hz]
    B  : chirp bandwidth     [Hz]  (default 10 kHz)

    Returns
    -------
    x  : complex baseband L-FMCW  shape (N,)
    """
    t = np.arange(N) / fs
    T = N / fs  # chirp duration
    # Instantaneous frequency sweeps linearly from −B/2 to +B/2
    phase = 2 * np.pi * (-B / 2 * t + B / (2 * T) * t ** 2)
    return np.exp(1j * phase) / np.sqrt(N)


# ── Synchronisation (matched filter) ────────────────────────────────────────

def synchronise(y: np.ndarray, x: np.ndarray) -> tuple[int, float, float]:
    """Estimate direct-path delay, amplitude, and phase via matched filter.  [eq. 11–13]

    R_yx[m] = Σ y[n+m] x*[n]   →   n̂_d = argmax |R_yx[m]|

    Returns
    -------
    nd    : estimated delay (samples)
    ad    : estimated direct-path amplitude
    theta_d: estimated direct-path phase  [rad]
    """
    Nx = len(x)
    max_range = len(y) - Nx
    Ryx = np.array([np.dot(y[m:m + Nx], x.conj()) for m in range(max(1, max_range))])
    nd = int(np.argmax(np.abs(Ryx)))
    ad = np.abs(Ryx[nd])
    theta_d = float(np.angle(Ryx[nd]))
    return nd, ad, theta_d


# ── Reflection path isolation ────────────────────────────────────────────────

def isolate_reflection(y: np.ndarray, x: np.ndarray,
                        nd: int, ad: float, theta_d: float) -> np.ndarray:
    """Remove direct-path estimate to expose reflection path.  [eq. 14]

    â_r[n] e^{jΔθ̂_r[n]} = (y[n] − â_d x[n−n̂_d] e^{jθ̂_d})
                             ÷ (x[n−n̂_d] e^{jθ̂_d})

    The sounding waveform x is transmitted continuously (periodic), so
    x[n − nd] is indexed modulo Nx (tile behaviour).
    """
    Nx  = len(x)
    N   = len(y)
    out = np.zeros(N, dtype=complex)
    ej  = np.exp(1j * theta_d)
    for n in range(nd, N):
        xn  = x[(n - nd) % Nx] * ej         # periodic sounding waveform
        out[n] = (y[n] - ad * xn) / (xn + 1e-30)
    return out


# ── Phase-differential Doppler estimator ────────────────────────────────────

def estimate_doppler(ar_resp: np.ndarray, fs: float,
                     Pthd: float = 0.01, Navg: int = 4) -> np.ndarray:
    """Estimate Doppler frequency via phase differential.  [eq. 15–16]

    f̂_D,r[n] = f_s · Arg(â_r[n] / â_r[n−1]) / (2π)        [eq. 15]

    Then smooth over 2*Navg + 1 samples:
    f̄_D,r[n] = (1/(2N_avg+1)) Σ f̂_D,r[n+m]               [eq. 16]

    Parameters
    ----------
    ar_resp : reflection path complex response  â_r e^{jΔθ_r}
    fs      : sampling frequency  [Hz]
    Pthd    : power threshold; only estimate when |â_r|² > Pthd
    Navg    : half-window for smoothing

    Returns
    -------
    fD_est  : estimated micro-Doppler  [Hz],  shape = (N,)
    """
    N       = len(ar_resp)
    power   = np.abs(ar_resp) ** 2
    fD_raw  = np.zeros(N)

    for n in range(1, N):
        if power[n] > Pthd and power[n - 1] > Pthd:
            ratio = ar_resp[n] * np.conj(ar_resp[n - 1])
            fD_raw[n] = fs * np.angle(ratio) / (2 * np.pi)

    # Smoothing window  [eq. 16]
    kernel   = np.ones(2 * Navg + 1) / (2 * Navg + 1)
    fD_smooth = np.convolve(fD_raw, kernel, mode="same")

    return fD_smooth


# ── RMS error bound ──────────────────────────────────────────────────────────

def rms_error_bound_dB(p: Dict) -> float:
    """Theoretical upper bound on RMS Doppler estimation error.  [eq. 20]

    10 log E(|ε_{f_D}|²) < 20 log(f_s/π) + RL − SNR

    Returns bound in dB (linear: 10^(bound/10) Hz²).
    """
    fs  = p["fs"]
    RL  = p["RL"]
    SNR = p["SNR"]
    return 20 * np.log10(fs / np.pi) + RL - SNR


# ── Full estimation pipeline ─────────────────────────────────────────────────

def run_estimation(p: Dict, rng: np.random.Generator,
                   n_periods: float = 4.0) -> Dict[str, np.ndarray]:
    """Run the estimation pipeline over n_periods of the blade cycle.

    Pipeline:
      1. Generate baseband received signal  y = y_direct + y_reflect + w
      2. Subtract known direct-path estimate  [eq. 14 applied to baseband model]
      3. Phase-differential Doppler estimation  [eq. 15–16]

    Note: A full L-FMCW RF implementation adds the L-FMCW matched filter
    step between (1) and (2) (see synchronise() + isolate_reflection() above),
    giving the 18.83 Hz RMSE reported in Table II. This baseband version
    demonstrates eqs. 14–16 directly.

    Returns dict with 't', 'fD_true', 'fD_est', 'in_zone', 'power'.
    """
    from microdoppler_channel import generate_received_signal

    fs = p["fs"]
    T_blade = 1.0 / (p["fm"] * p["Nb"])
    N = int(fs * n_periods * T_blade)
    t = np.arange(N) / fs

    # Generate baseband received signal  [eq. 7]
    sig = generate_received_signal(t, p, rng)
    y   = sig["y"]

    # Known direct-path parameters (perfect sync at baseband)
    sigma2_s   = 1.0
    ad         = np.sqrt(sigma2_s)
    theta_d    = 2.0 * np.pi * p["fc"] * p["dUE"] / C_LIGHT

    # Subtract direct path → reflection channel response  [eq. 14]
    ar_resp = y - ad * np.exp(1j * theta_d)

    # Power threshold: reflection power = σ²_s × 10^{−RL/10}
    sigma2_r = sigma2_s * 10 ** (-p["RL"] / 10)
    Pthd     = sigma2_r * 0.1   # 10% of expected reflection power

    # Phase-differential Doppler estimation  [eq. 15–16]
    fD_est = estimate_doppler(ar_resp, fs, Pthd=Pthd, Navg=p["Navg"])

    return {
        "t":       t,
        "fD_true": sig["fD_true"],
        "fD_est":  fD_est,
        "in_zone": sig["in_zone"],
        "power":   np.abs(ar_resp) ** 2,
    }


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_estimation_result(res: Dict, p: Dict, ax_top: plt.Axes, ax_bot: plt.Axes) -> None:
    """Two-panel plot: true vs estimated f_D, and reflection path power."""
    t_ms = res["t"] * 1000

    # Top: true (orange) vs estimated (green)
    ax_top.plot(t_ms, res["fD_true"], color=COLORS["active"],  lw=2.0,  alpha=0.9,
                label="True f_D,r  [eq. 5]")
    ax_top.plot(t_ms, res["fD_est"],  color="#3fb950",          lw=1.2,  alpha=0.85,
                label="Estimated f̂_D,r  [eq. 15–16]", ls="--")
    ax_top.axhline(0, color="#3d444d", lw=0.8)

    fmax = max_microdoppler(p)
    ax_top.axhline( fmax, color=COLORS["active"], lw=0.7, ls=":", alpha=0.5)
    ax_top.axhline(-fmax, color=COLORS["active"], lw=0.7, ls=":", alpha=0.5)

    ax_top.set_ylabel("f_D  [Hz]")
    ax_top.set_title("True vs Estimated Micro-Doppler  [eq. 5, 15–16]")
    ax_top.legend(fontsize=8, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    _apply_dark_style(ax_top)

    # Bottom: reflection path power
    ax_bot.semilogy(t_ms, res["power"] + 1e-20, color="#58a6ff", lw=1.2)
    ax_bot.set_xlabel("Time [ms]")
    ax_bot.set_ylabel("|â_r|²  [log]")
    ax_bot.set_title("Reflection Path Power  (non-zero in active windows)")
    _apply_dark_style(ax_bot)

    for ax in [ax_top, ax_bot]:
        ax.set_xlim(t_ms[0], t_ms[-1])


def plot_rmse_vs_snr(p_base: Dict, ax: plt.Axes) -> None:
    """Plot RMS Doppler error vs SNR (analytical bound) — matches paper Fig. 5 style."""
    snr_range = np.arange(0, 45, 1)  # dB

    configs = [
        {"RL": 5,  "fs": 60e3,  "fc": 2.5e9,  "label": "RL=5dB, fs=60k, fc=2.5GHz",  "ls": "-",  "c": "#58a6ff"},
        {"RL": 15, "fs": 60e3,  "fc": 2.5e9,  "label": "RL=15dB, fs=60k, fc=2.5GHz", "ls": "--", "c": "#f0883e"},
        {"RL": 5,  "fs": 480e3, "fc": 2.5e9,  "label": "RL=5dB, fs=480k, fc=2.5GHz", "ls": "-.", "c": "#3fb950"},
        {"RL": 5,  "fs": 60e3,  "fc": 38e9,   "label": "RL=5dB, fs=60k, fc=38GHz",   "ls": ":",  "c": "#e3b341"},
    ]

    for cfg in configs:
        p = {**p_base, "RL": cfg["RL"], "fs": cfg["fs"], "fc": cfg["fc"]}
        # Theoretical bound  [eq. 20]
        bound_dB = [20 * np.log10(p["fs"] / np.pi) + p["RL"] - snr for snr in snr_range]
        bound_hz = [np.sqrt(10 ** (b / 10)) for b in bound_dB]
        ax.semilogy(snr_range, bound_hz, lw=1.8, ls=cfg["ls"], color=cfg["c"],
                    label=cfg["label"])

    ax.set_xlabel("SNR  [dB]")
    ax.set_ylabel("RMS error  ε_{f_D}  [Hz]")
    ax.set_title("RMS Doppler Error Bound  [eq. 20]")
    ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    _apply_dark_style(ax)
    ax.set_xlim(0, 44)
    ax.set_ylim(1e-1, 1e5)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(seed=0)
    p   = DEFAULT_PARAMS.copy()

    p["SNR"] = 10.0
    p["fs"] = 60e3
    p["RL"] = 5.0

    print("=" * 62)
    print("  Micro-Doppler Estimation  —  Hou, Wang & Lin (2021)")
    print("=" * 62)

    bound_dB = rms_error_bound_dB(p)
    bound_Hz = np.sqrt(10 ** (bound_dB / 10))



    print(f"  Theoretical RMS error bound [eq. 20]: {bound_Hz:.2f} Hz")

    # =========================
    # Q2 metrics
    # =========================
    Gamma = 50.0   # serviceability threshold [Hz]

    # Use theoretical RMS error bound as sigma
    sigma = bound_Hz

    # One-sided Gaussian tail: P(epsilon > Gamma)
    p_fail = 0.5 * math.erfc(Gamma / (sigma * math.sqrt(2.0)))
    p_serv = 1.0 - p_fail

    # k-out-of-n resilience: n=10, k=7
    n_uav = 10
    k_req = 7

    R_sys = sum(
        math.comb(n_uav, i) * (p_serv ** i) * ((1 - p_serv) ** (n_uav - i))
        for i in range(k_req, n_uav + 1)
    )

    print("\n================ Q2 Derived Metrics ================")
    print(f"Sigma (RMS bound)          = {sigma:.2f} Hz")
    print(f"Gamma                      = {Gamma:.2f} Hz")
    print(f"P_fail = P(epsilon>Gamma)  = {p_fail:.6f} ({p_fail*100:.2f}%)")
    print(f"P_serv                     = {p_serv:.6f} ({p_serv*100:.2f}%)")
    print(f"R_sys = P(X>={k_req})      = {R_sys:.6f} ({R_sys*100:.2f}%)")
    print("====================================================")

    print(f"  Paper reports:                         18.83 Hz  (at 40dB SNR)")

    print("\n  Running L-FMCW estimation pipeline …", end=" ", flush=True)
    res = run_estimation(p, rng, n_periods=4.0)
    print("done.")

    # RMS error (only in active zones)
    mask = res["in_zone"]
    if mask.any():
        err = res["fD_true"][mask] - res["fD_est"][mask]
        rmse = np.sqrt(np.mean(err ** 2))
        print(f"  Simulated RMS error (in-zone): {rmse:.2f} Hz")

    # ── Figure ──────────────────────────────────────────────────────
    fig = plt.figure(figsize=(13, 8), facecolor="#0d1117")
    fig.suptitle(
        "Micro-Doppler Estimation via L-FMCW Sounding\n"
        "Hou, Wang & Lin — IEEE Wireless Comm. Letters 2021",
        color="#c9d1d9", fontsize=11, y=0.97
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.35,
                           left=0.08, right=0.96, top=0.91, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, :])   # top: spans both columns
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    plot_estimation_result(res, p, ax1, ax2)
    plot_rmse_vs_snr(p, ax3)

    out = "microdoppler_estimation_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  → Saved: {out}")


if __name__ == "__main__":
    main()
