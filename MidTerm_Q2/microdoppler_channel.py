"""
Micro-Doppler Shift Simulation for Rotary-Wing UAV Sub-6 GHz Communications
=============================================================================

Reproduces the channel model and analytical results from:

    Hou, H.-A., Wang, L.-C., & Lin, H.-P. (2021).
    "Micro-Doppler Shift and Its Estimation in Rotary-Wing UAV
    Sub-6 GHz Communications."
    IEEE Wireless Communications Letters, 10(10), 2185–2189.
    https://doi.org/10.1109/LWC.2021.3095898

Key equations implemented
--------------------------
  [eq. 1]  θ_e(t)   = 2π f_m t + θ_p(0) + 2mπ/N_b − π/2
  [eq. 4]  θ_rz     = arcsin(D_p / (4 d_ant))
  [eq. 5]  f_D,r(t) = −4π f_m d_ant f_c sin(2θ_e(t)) / c
  [eq. 8]  E|a_r|²  = σ_s² · 10^{(RL+δ)/10}

Output
------
  microdoppler_results.png  —  four-panel figure matching style of paper Figs 4 & 5

"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import chirp
from typing import Dict, Any

# ── Physical constants ──────────────────────────────────────────────────────
C_LIGHT: float = 3e8          # speed of light [m/s]

# ── Default parameters (Table I & II, Hou et al. 2021) ─────────────────────
DEFAULT_PARAMS: Dict[str, Any] = {
    "Dp":    0.254,          # propeller diameter            [m]
    "fm":    4620 / 60,      # motor revolution              [rev/s]  (= 4620 rpm)
    "Nb":    2,              # number of blades per propeller
    "fc":    2.5e9,          # carrier frequency             [Hz]
    "dUE":   1000.0,         # UE-to-antenna distance        [m]
    "dant":  0.200,          # hub-to-antenna distance       [m]
    "RL":    5.0,            # return loss (there is the second config: 15 dB)[dB]
    "fs":    60e3,           # sampling frequency            [samples/s]
    "SNR":   40.0,           # signal-to-noise ratio         [dB]
    "theta_p0": 0.0,         # initial blade angle           [rad]
    "Navg":  4,              # averaging half-window for estimation [eq. 16]
}

# ── Core physics functions ──────────────────────────────────────────────────

def blade_angle(t: np.ndarray, m: int, p: Dict) -> np.ndarray:
    """Blade phase angle for blade index m at time t.  [eq. 1]

    θ_e(t) = 2π f_m t + θ_p(0) + 2mπ/N_b − π/2

    Parameters
    ----------
    t   : time array [s]
    m   : blade index  0 … N_b − 1
    p   : parameter dict

    Returns
    -------
    θ_e : ndarray, same shape as t  [rad]
    """
    return (2 * np.pi * p["fm"] * t
            + p["theta_p0"]
            + 2 * m * np.pi / p["Nb"]
            - np.pi / 2)


def reflection_zone_angle(p: Dict) -> float:
    """Half-angle of the reflection zone.  [eq. 4]

    θ_rz = arcsin(D_p / (4 d_ant))
    """
    arg = np.clip(p["Dp"] / (4.0 * p["dant"]), -1.0, 1.0)
    return float(np.arcsin(arg))


def blade_period(p: Dict) -> float:
    """Time between successive blade reflections = 1 / (f_m * N_b)  [s]"""
    return 1.0 / (p["fm"] * p["Nb"])


def reflection_duration(p: Dict) -> float:
    """Duration of a single reflection event.  [s]

    Derived from the reflection zone arc length:
        Δt_refl = 2 θ_rz · T_blade / π
    """
    return 2.0 * reflection_zone_angle(p) * blade_period(p) / np.pi


def microdoppler_freq(theta_e: np.ndarray, p: Dict) -> np.ndarray:
    """Instantaneous micro-Doppler Doppler frequency.  [eq. 5]

    f_D,r(t) = −4π f_m d_ant f_c sin(2θ_e(t)) / c
    """
    return (-4.0 * np.pi * p["fm"] * p["dant"] * p["fc"]
            * np.sin(2.0 * theta_e) / C_LIGHT)


def max_microdoppler(p: Dict) -> float:
    """Maximum |f_D| observed within the reflection zone.

    The blade is only active when |θ_e| < θ_rz, so the maximum occurs at
    the zone boundary θ_e = ±θ_rz, NOT at the global maximum (θ_e = π/4):

        |f_D|_max = 4π f_m d_ant f_c |sin(2 θ_rz)| / c

    With default params: ≈ 963 Hz  (paper Fig. 4b, Table II).
    """
    thrz = reflection_zone_angle(p)
    return 4.0 * np.pi * p["fm"] * p["dant"] * p["fc"] * abs(np.sin(2 * thrz)) / C_LIGHT


def in_reflection_zone(theta_e: np.ndarray, p: Dict) -> np.ndarray:
    """Boolean mask: True where blade angle is inside the reflection zone.

    Reflection occurs for each blade m when its angle θ_e,m satisfies
    |θ_e,m| < θ_rz  (after wrapping to (−π, π]).
    """
    thrz = reflection_zone_angle(p)
    # Collect mask across all blades
    mask = np.zeros(theta_e.shape, dtype=bool)
    for m in range(p["Nb"]):
        te_m = blade_angle(
            # recover t from θ_e0: t = (θ_e + π/2 − θ_p0) / (2π f_m)
            # We pass raw θ_e for blade 0 and add blade offset
            t=np.zeros_like(theta_e),   # dummy (we work with pre-computed angles)
            m=0, p=p
        )
        # Offset for blade m: θ_e,m = θ_e,0 + 2mπ/N_b
        te_wrapped = _wrap_pi(theta_e + 2 * m * np.pi / p["Nb"])
        mask |= (np.abs(te_wrapped) < thrz)
    return mask


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    """Wrap angle array to (−π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


# ── Signal generation ───────────────────────────────────────────────────────

def generate_received_signal(t: np.ndarray, p: Dict,
                              rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Generate the full received signal with direct + reflection paths.  [eq. 7]

    y[n] = a_d[n] x[n−n_d] e^{jθ_d[n]}
         + a_r[n] x[n−n_d] e^{j(θ_d[n] + Δθ_r[n])}
         + w[n]

    Parameters
    ----------
    t   : time array [s]  (uniform, step = 1/fs)
    p   : parameter dict
    rng : numpy random Generator for reproducibility

    Returns
    -------
    dict with keys:
        "y"          : complex received signal
        "theta_e"    : blade angle for blade 0     [rad]
        "in_zone"    : reflection-active boolean mask
        "fD_true"    : true instantaneous f_D,r   [Hz]  (0 outside zone)
        "thetaD_true": true phase increment Δθ_r  [rad] (0 outside zone)
    """
    fs   = p["fs"]
    fc   = p["fc"]
    dUE  = p["dUE"]
    dant = p["dant"]
    RL   = p["RL"]
    SNR  = p["SNR"]

    # Blade 0 angle
    theta_e = blade_angle(t, 0, p)

    # Reflection-zone mask (any blade active)
    thrz  = reflection_zone_angle(p)
    in_z  = np.zeros(len(t), dtype=bool)
    te_active = np.zeros(len(t))
    for m in range(p["Nb"]):
        te_m = _wrap_pi(blade_angle(t, m, p))
        active_m = np.abs(te_m) < thrz
        in_z |= active_m
        te_active = np.where(active_m, te_m, te_active)

    # True instantaneous Doppler   [eq. 5]
    fD_true = np.where(in_z, microdoppler_freq(te_active, p), 0.0)

    # True phase increment Δθ_r[n] = 2π f_D,r[n] / f_s   [eq. 9]
    # delta_theta_r already contains 1/f_s; cumsum gives cumulative Δθ_r[n]  [eq. 10]
    delta_theta_r = np.where(in_z, 2.0 * np.pi * fD_true / fs, 0.0)
    theta_r_cumul = np.cumsum(delta_theta_r)   # cumulative phase — do NOT divide by fs again

    # Signal power
    sigma2_s = 1.0                           # direct-path power (normalised)
    sigma2_r = sigma2_s * 10 ** (-(RL) / 10)  # reflection power  [eq. 8]  (simplified δ=0)

    # AWGN variance from SNR: SNR = σ²_s / σ²_w
    sigma2_w = sigma2_s / 10 ** (SNR / 10)

    # Baseband model: carrier is removed; only path phase (static) + Doppler remains.
    # theta_d = 2π fc dUE / c  is the static direct-path phase [constant over observation]
    ad = np.sqrt(sigma2_s)
    theta_d_static = 2.0 * np.pi * fc * dUE / C_LIGHT  # static LOS phase

    # Reflection path component
    ar = np.sqrt(sigma2_r)

    # Compose baseband received signal  [eq. 7]
    x_direct  = ad * np.exp(1j * theta_d_static) * np.ones(len(t))
    x_reflect = ar * np.exp(1j * (theta_d_static + theta_r_cumul))
    noise = ((rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t)))
             * np.sqrt(sigma2_w / 2))

    y = x_direct + x_reflect + noise

    return {
        "y":           y,
        "theta_e":     theta_e,
        "in_zone":     in_z,
        "fD_true":     fD_true,
        "thetaD_true": delta_theta_r,
    }


# ── Analysis helpers ────────────────────────────────────────────────────────

def compute_doppler_waveform(p: Dict, n_periods: float = 2.5
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the analytical f_D,r(t) over n_periods of the blade cycle.

    Returns
    -------
    t       : time array [s]
    fD      : micro-Doppler [Hz]  (0 outside reflection zones)
    in_zone : boolean mask
    """
    T_blade = blade_period(p)
    t = np.linspace(0, n_periods * T_blade, int(p["fs"] * n_periods * T_blade))

    thrz = reflection_zone_angle(p)
    fD    = np.zeros(len(t))
    in_z  = np.zeros(len(t), dtype=bool)
    te_active = np.zeros(len(t))

    for m in range(p["Nb"]):
        te_m    = _wrap_pi(blade_angle(t, m, p))
        active_m = np.abs(te_m) < thrz
        in_z    |= active_m
        te_active = np.where(active_m, te_m, te_active)

    fD = np.where(in_z, microdoppler_freq(te_active, p), 0.0)
    return t, fD, in_z


# ── Plotting ─────────────────────────────────────────────────────────────────

COLORS = {
    "active":   "#f0883e",
    "inactive": "#388bfd",
    "zone":     "rgba(240,136,62,0.12)",
    "grid":     "#2d3748",
    "bg":       "#0d1117",
    "text":     "#8b949e",
}

def _apply_dark_style(ax):
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.xaxis.label.set_color("#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.title.set_color("#c9d1d9")
    for spine in ax.spines.values():
        spine.set_edgecolor("#21262d")
    ax.grid(True, color="#161b22", linewidth=0.8, linestyle="--")


def plot_doppler_waveform(p: Dict, ax: plt.Axes) -> None:
    """Plot f_D,r(t) — matching paper Fig. 4(b)."""
    t, fD, in_z = compute_doppler_waveform(p, n_periods=2.5)
    t_ms = t * 1000  # convert to ms

    # Inactive segments (f_D = 0)
    ax.plot(t_ms, np.where(~in_z, fD, np.nan), color=COLORS["inactive"],
            lw=1.5, label="f_D,r = 0  (outside zone)")

    # Active segments
    ax.plot(t_ms, np.where(in_z, fD, np.nan), color=COLORS["active"],
            lw=2.2, label="f_D,r  (reflection active)")

    # Shade reflection windows
    thrz = reflection_zone_angle(p)
    in_transition = np.diff(in_z.astype(int), prepend=0)
    starts = t_ms[in_transition ==  1]
    ends   = t_ms[in_transition == -1] if in_transition[-1] != 1 else np.append(t_ms[in_transition == -1], t_ms[-1])
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.08, color=COLORS["active"], linewidth=0)

    # Max/min lines
    fD_max = max_microdoppler(p)
    ax.axhline( fD_max, color=COLORS["active"], lw=0.7, ls=":", alpha=0.6,
                label=f"|f_D|_max = {fD_max:.1f} Hz")
    ax.axhline(-fD_max, color=COLORS["active"], lw=0.7, ls=":", alpha=0.6)
    ax.axhline(0,        color="#3d444d", lw=0.8, ls="-")

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("f_D,r  [Hz]")
    ax.set_title(r"Micro-Doppler  $f_{D,r}(t)$  —  [eq. 5]")
    ax.legend(fontsize=7, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    _apply_dark_style(ax)
    ax.set_xlim(0, t_ms[-1])


def plot_blade_angle(p: Dict, ax: plt.Axes) -> None:
    """Plot blade angle θ_e(t) — matching paper Fig. 4(a) (phase proxy)."""
    T_blade = blade_period(p)
    t = np.linspace(0, 2.5 * T_blade, int(p["fs"] * 2.5 * T_blade))
    t_ms = t * 1000
    thrz = reflection_zone_angle(p)

    # Blade 0 angle (wrapped to (−π, π])
    te0 = _wrap_pi(blade_angle(t, 0, p))
    in_z = np.abs(te0) < thrz
    if p["Nb"] > 1:
        te1  = _wrap_pi(blade_angle(t, 1, p))
        in_z |= (np.abs(te1) < thrz)

    ax.plot(t_ms, te0, color=COLORS["inactive"], lw=1.5, label="θ_e,0  (blade 0)")
    ax.axhline( thrz, color=COLORS["active"], lw=1, ls="--", alpha=0.7,
                label=f"±θ_rz = ±{np.degrees(thrz):.1f}°")
    ax.axhline(-thrz, color=COLORS["active"], lw=1, ls="--", alpha=0.7)
    ax.axhline(0, color="#3d444d", lw=0.8)

    # Shade active windows
    in_transition = np.diff(in_z.astype(int), prepend=0)
    starts = t_ms[in_transition ==  1]
    ends_idx = np.where(in_transition == -1)[0]
    ends = t_ms[ends_idx] if len(ends_idx) else np.array([t_ms[-1]])
    for s, e in zip(starts, ends):
        ax.axvspan(s, e, alpha=0.08, color=COLORS["active"], linewidth=0)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("θ_e  [rad]")
    ax.set_title(r"Blade angle  $\theta_e(t)$  —  [eq. 1]")
    ax.legend(fontsize=7, loc="upper right",
              facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    _apply_dark_style(ax)
    ax.set_xlim(0, t_ms[-1])


def plot_fD_vs_fc(p_base: Dict, ax: plt.Axes) -> None:
    """Plot max |f_D| vs carrier frequency for different D_p values."""
    fc_range = np.linspace(0.5e9, 6e9, 300)
    Dp_vals  = [0.15, 0.254, 0.40]  # metres

    for Dp in Dp_vals:
        p = {**p_base, "Dp": Dp}
        fDmax = [4 * np.pi * p["fm"] * p["dant"] * fc / C_LIGHT for fc in fc_range]
        ax.plot(fc_range / 1e9, fDmax, lw=2,
                label=f"D_p = {Dp*1000:.0f} mm")

    ax.set_xlabel("Carrier frequency f_c  [GHz]")
    ax.set_ylabel("|f_D|_max  [Hz]")
    ax.set_title(r"Max Micro-Doppler vs $f_c$")
    ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    _apply_dark_style(ax)


def plot_fD_vs_rpm(p_base: Dict, ax: plt.Axes) -> None:
    """Plot max |f_D| vs motor RPM for different carrier frequencies."""
    rpm_range = np.arange(500, 9000, 50)
    fc_vals   = [1.0e9, 2.5e9, 5.8e9]

    for fc in fc_vals:
        p = {**p_base, "fc": fc}
        fDmax = [4 * np.pi * (rpm/60) * p["dant"] * p["Dp"] / 4 * fc / C_LIGHT
                 for rpm in rpm_range]
        # Correct: |f_D|_max = 4π fm dant fc / c  (not Dp/4, that's θ_rz)
        fDmax = [4 * np.pi * (rpm/60) * p["dant"] * fc / C_LIGHT
                 for rpm in rpm_range]
        ax.plot(rpm_range, fDmax, lw=2, label=f"f_c = {fc/1e9:.1f} GHz")

    ax.axvline(4620, color="#6e7681", lw=1, ls=":", label="Default 4620 rpm")
    ax.set_xlabel("Motor speed f_m  [RPM]")
    ax.set_ylabel("|f_D|_max  [Hz]")
    ax.set_title(r"Max Micro-Doppler vs RPM")
    ax.legend(fontsize=7, facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9")
    _apply_dark_style(ax)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    rng = np.random.default_rng(seed=42)
    p   = DEFAULT_PARAMS.copy()

    print("=" * 62)
    print("  Micro-Doppler Channel  —  Hou, Wang & Lin (2021)")
    print("=" * 62)
    print(f"  Motor speed   f_m  = {p['fm']*60:.0f} rpm  ({p['fm']:.3f} rev/s)")
    print(f"  Carrier freq  f_c  = {p['fc']/1e9:.2f} GHz")
    print(f"  Prop. diam.   D_p  = {p['Dp']*1000:.1f} mm")
    print(f"  Ant. offset   d_ant= {p['dant']*1000:.1f} mm")
    print(f"  Num. blades   N_b  = {p['Nb']}")
    print("-" * 62)

    thrz = reflection_zone_angle(p)
    T_bl = blade_period(p)
    t_rf = reflection_duration(p)
    fmax = max_microdoppler(p)

    print(f"  θ_rz   = {np.degrees(thrz):.2f}°  = {thrz:.4f} rad      [eq. 4]")
    print(f"  Period = {T_bl*1000:.3f} ms  (60 / (fm·Nb))")
    print(f"  Refl.  = {t_rf*1000:.3f} ms  per reflection window")
    print(f"  |f_D|_max = {fmax:.2f} Hz  (at θ_e = ±π/4)    [eq. 5]")
    print("=" * 62)

    # ── Figure: 4-panel matching paper style ───────────────────────
    fig = plt.figure(figsize=(13, 9), facecolor="#0d1117")
    fig.suptitle(
        "Micro-Doppler Effect in Rotary-Wing UAV Sub-6 GHz Communications\n"
        "Hou, Wang & Lin — IEEE Wireless Comm. Letters 2021",
        color="#c9d1d9", fontsize=11, y=0.97
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35,
                           left=0.08, right=0.96, top=0.91, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_blade_angle(p, ax1)           # Panel 1: θ_e(t)  → Fig 4(a) analogue
    plot_doppler_waveform(p, ax2)      # Panel 2: f_D(t)  → Fig 4(b) analogue
    plot_fD_vs_fc(p, ax3)              # Panel 3: |f_D|_max vs f_c
    plot_fD_vs_rpm(p, ax4)             # Panel 4: |f_D|_max vs RPM

    out = "microdoppler_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    print(f"\n  → Saved: {out}")

    # ── Quick numerical check ───────────────────────────────────────
    print("\n  Validation (default params vs paper Table II / Fig 4):")
    print(f"    Expected |f_D|_max ≈ 963 Hz  →  Got {fmax:.2f} Hz")
    print(f"    Expected period    ≈ 6.494 ms →  Got {T_bl*1000:.3f} ms")
    print(f"    Expected refl. dur ≈ 1.323 ms →  Got {t_rf*1000:.3f} ms")


if __name__ == "__main__":
    main()
