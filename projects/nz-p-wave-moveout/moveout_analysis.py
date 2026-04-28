"""
moveout_analysis.py
-------------------
Loads tohoku_nz_array.npz (three-component ZRT data from GeoNet), picks the
direct P on the vertical and the Moho P-to-S conversion (Ps) on the radial,
fits a linear moveout curve to each, and produces two figures.

Figures
-------
record_section.png   – three-panel ZRT record section with P and Ps markers
                       and fitted moveout lines overlaid
moveout_fit.png      – P and Ps arrival times vs distance, linear fits,
                       and the τ_Ps delay that constrains crustal thickness

Run after download_data.py:
    python moveout_analysis.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ── load ──────────────────────────────────────────────────────────────────────

DATA_FILE = Path("tohoku_nz_array.npz")

data          = np.load(str(DATA_FILE), allow_pickle=True)
Z             = data["Z"]            # (N, T)
R             = data["R"]
T_comp        = data["T"]
distances     = data["distances"]    # km
times_rel     = data["times_rel"]    # s relative to predicted P  (0 = P)
stations      = data["stations"]
sampling_rate = float(data["sampling_rate"])
evt_mag       = float(data["evt_mag"])
evt_dep       = float(data["evt_dep"])

N, npts = Z.shape
dt      = 1.0 / sampling_rate

print(f"Loaded {N} stations  |  {npts} samples @ {sampling_rate:.0f} sps  "
      f"|  window {times_rel[0]:.0f} to {times_rel[-1]:.0f} s")

# Sort north-to-south (ascending distance for Japan source ≈ northward wave)
order     = np.argsort(distances)
Z         = Z[order]
R         = R[order]
T_comp    = T_comp[order]
distances = distances[order]
stations  = stations[order]


# ── arrival picking ───────────────────────────────────────────────────────────

def sample(t_sec: float) -> int:
    """Convert a time (s relative to P) to a sample index."""
    return int(round((t_sec - times_rel[0]) * sampling_rate))


# P window: search for first large peak on Z, 0–8 s after predicted P
P_SEARCH_START =  0.0
P_SEARCH_END   =  8.0
i_ps, i_pe = sample(P_SEARCH_START), sample(P_SEARCH_END)

# Ps window: search on radial, 2–10 s AFTER the picked P
PS_OFFSET_START =  2.0
PS_OFFSET_END   = 10.0

p_picks  = np.empty(N)   # s relative to predicted P
ps_picks = np.empty(N)   # s relative to predicted P
ps_valid = np.ones(N, dtype=bool)

for i in range(N):
    # P: maximum absolute amplitude on Z in search window
    seg = Z[i, i_ps:i_pe]
    p_picks[i] = times_rel[i_ps + np.argmax(np.abs(seg))]

    # Ps: maximum amplitude on radial in window after picked P
    j0 = sample(p_picks[i] + PS_OFFSET_START)
    j1 = sample(p_picks[i] + PS_OFFSET_END)
    j0 = max(0, j0);  j1 = min(npts - 1, j1)
    if j1 <= j0:
        ps_valid[i] = False
        ps_picks[i] = np.nan
        continue
    seg_r = R[i, j0:j1]
    ps_picks[i] = times_rel[j0 + np.argmax(np.abs(seg_r))]


# ── moveout fitting ───────────────────────────────────────────────────────────

def linear(dist_km, a, v):
    return a + dist_km / v


# Fit P moveout
popt_p, pcov_p = curve_fit(linear, distances, p_picks, p0=[0.0, 8.0])
a_p, v_p       = popt_p
v_p_err        = np.sqrt(pcov_p[1, 1])

# Fit Ps moveout (same slope expected; intercept offset by τ_Ps)
mask = ps_valid
popt_ps, pcov_ps = curve_fit(linear, distances[mask], ps_picks[mask], p0=[5.0, 8.0])
a_ps, v_ps       = popt_ps
v_ps_err         = np.sqrt(pcov_ps[1, 1])

# τ_Ps at the median distance = time difference between the two lines
d_mid    = np.median(distances)
tau_ps   = linear(d_mid, *popt_ps) - linear(d_mid, *popt_p)

# Crustal thickness estimate: τ_Ps ≈ H(√(1/Vs²-p²) − √(1/Vp²-p²))
# For typical NZ crust (Vp=6.4, Vs=3.7, p≈0.068 s/km at ~75°):
VP_CRUST  = 6.4
VS_CRUST  = 3.7
P_RAY     = 0.068   # s/km — typical at 75°
eta_S = np.sqrt(max(1/VS_CRUST**2 - P_RAY**2, 0))
eta_P = np.sqrt(max(1/VP_CRUST**2 - P_RAY**2, 0))
H_est = tau_ps / (eta_S - eta_P) if (eta_S - eta_P) > 0 else np.nan

print(f"\n── P  moveout  ──  v = {v_p:.2f} ± {v_p_err:.2f} km/s")
print(f"── Ps moveout  ──  v = {v_ps:.2f} ± {v_ps_err:.2f} km/s")
print(f"── τ_Ps (median distance) = {tau_ps:.2f} s")
print(f"── Estimated crustal thickness H ≈ {H_est:.0f} km")


# ── figure 1: 3-panel record section ─────────────────────────────────────────

SCALE   = 60.0   # km — visual half-amplitude of each normalised trace
d_range = np.linspace(distances.min(), distances.max(), 300)

fig, axes = plt.subplots(1, 3, figsize=(16, 11), sharey=True)
titles = ["Vertical (Z)", "Radial (R)  ← toward event", "Transverse (T)"]
comps  = [Z, R, T_comp]
cols   = ["#2c3e50", "#c0392b", "#16a085"]

for ax, comp, title, col in zip(axes, comps, titles, cols):
    for i in range(N):
        d     = distances[i]
        trace = comp[i] * SCALE
        ax.plot(times_rel, trace + d, color=col, linewidth=0.6, alpha=0.75)

    # P picks (all panels)
    ax.scatter(p_picks, distances,
               marker="|", s=80, linewidths=1.5,
               color="steelblue", zorder=4, label="P pick")

    # Ps picks (radial panel only)
    if title.startswith("Radial"):
        ax.scatter(ps_picks[mask], distances[mask],
                   marker="|", s=80, linewidths=1.5,
                   color="gold", zorder=4, label="Ps pick")

    # Fitted moveout lines
    ax.plot(linear(d_range, *popt_p), d_range,
            color="steelblue", linewidth=1.8, linestyle="--",
            label=f"P fit  v={v_p:.1f} km/s")
    if title.startswith("Radial"):
        ax.plot(linear(d_range, *popt_ps), d_range,
                color="gold", linewidth=1.8, linestyle="--",
                label=f"Ps fit  τ={tau_ps:.1f} s")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time relative to predicted P (s)", fontsize=10)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(times_rel[0], times_rel[-1])

axes[0].set_ylabel("Epicentral distance (km)", fontsize=11)
axes[0].invert_yaxis()
axes[0].set_ylim(distances.max() + 150, distances.min() - 150)

fig.suptitle(
    f"Three-Component Record Section — 2011 Tohoku M{evt_mag}  "
    f"(depth {evt_dep:.0f} km)\n"
    f"GeoNet broadband, NZ  ·  {N} stations  ·  0.5–2 Hz bandpass",
    fontsize=12, y=1.01,
)
fig.tight_layout()
fig.savefig("record_section.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved → record_section.png")


# ── figure 2: moveout fit comparison ─────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10),
                                 gridspec_kw={"height_ratios": [3, 1]})

# Top: P and Ps picks with linear fits
ax1.scatter(distances, p_picks,
            color="steelblue", s=55, zorder=3, edgecolors="white",
            linewidths=0.5, label="P picks (Z component)")
ax1.scatter(distances[mask], ps_picks[mask],
            color="#e67e22", s=55, zorder=3, edgecolors="white",
            linewidths=0.5, label="Ps picks (R component)")

ax1.plot(d_range, linear(d_range, *popt_p),
         color="steelblue", linewidth=2,
         label=f"P linear fit   v = {v_p:.2f} ± {v_p_err:.2f} km/s")
ax1.plot(d_range, linear(d_range, *popt_ps),
         color="#e67e22", linewidth=2,
         label=f"Ps linear fit  v = {v_ps:.2f} ± {v_ps_err:.2f} km/s")

# Annotate τ_Ps
ax1.annotate(
    "",
    xy  =(d_mid, linear(d_mid, *popt_ps)),
    xytext=(d_mid, linear(d_mid, *popt_p)),
    arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5),
)
ax1.text(d_mid + 60, (linear(d_mid, *popt_p) + linear(d_mid, *popt_ps)) / 2,
         f"τ_Ps = {tau_ps:.1f} s\n→ H ≈ {H_est:.0f} km",
         fontsize=10, color="gray", va="center")

ax1.set_ylabel("Arrival time relative to predicted P (s)", fontsize=11)
ax1.set_title(
    f"P and Ps Moveout — 2011 Tohoku M{evt_mag}\n"
    f"τ_Ps = {tau_ps:.1f} s  →  crustal thickness H ≈ {H_est:.0f} km",
    fontsize=12,
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Bottom: τ_Ps per station (should be roughly flat)
tau_per_sta = ps_picks[mask] - p_picks[mask]
ax2.scatter(distances[mask], tau_per_sta,
            color="#e67e22", s=40, zorder=3, edgecolors="white", linewidths=0.4)
ax2.axhline(np.median(tau_per_sta), color="gray", linewidth=1.5, linestyle="--",
            label=f"Median τ_Ps = {np.median(tau_per_sta):.1f} s")
ax2.set_xlabel("Epicentral distance (km)", fontsize=11)
ax2.set_ylabel("τ_Ps  (s)", fontsize=11)
ax2.set_title("P-to-Ps Delay Per Station", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("moveout_fit.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved → moveout_fit.png")
