"""
moveout_analysis.py
-------------------
Loads the downloaded dataset (kaikoura_nz_array.npz), picks the P-wave
arrival at each station, fits a linear moveout curve, and produces two figures:

  record_section.png   – seismograms sorted by distance with the fitted
                         moveout line overlaid
  moveout_fit.png      – scatter of (distance, arrival time) with linear fit,
                         residuals, and derived Pn velocity

Run after download_data.py:
    python moveout_analysis.py

Outputs
-------
record_section.png
moveout_fit.png
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# ── load data ─────────────────────────────────────────────────────────────────

DATA_FILE = Path("kaikoura_nz_array.npz")

data          = np.load(str(DATA_FILE), allow_pickle=True)
waveforms     = data["waveforms"]          # (N, T)
distances     = data["distances"]          # km
stations      = data["stations"]           # "NET.STA"
times_rel     = data["times_rel"]          # seconds relative to predicted P
sampling_rate = float(data["sampling_rate"])
evt_mag       = float(data["evt_mag"])
evt_dep       = float(data["evt_dep"])

N, T = waveforms.shape
dt   = 1.0 / sampling_rate

print(f"Loaded {N} stations, {T} samples each ({times_rel[-1]:.0f} s window)")

# Sort by distance for clean record section display
order      = np.argsort(distances)
waveforms  = waveforms[order]
distances  = distances[order]
stations   = stations[order]


# ── arrival time picking ───────────────────────────────────────────────────────
# The waveforms are aligned so that time 0 = predicted P onset (from iasp91).
# We refine by finding the maximum absolute amplitude in the first 8 seconds
# after the predicted onset, which catches slight prediction errors.

PICK_WINDOW_START =  0.0   # seconds after predicted P
PICK_WINDOW_END   =  8.0   # seconds after predicted P

i_start = int((PICK_WINDOW_START - times_rel[0]) * sampling_rate)
i_end   = int((PICK_WINDOW_END   - times_rel[0]) * sampling_rate)
i_start = max(0, i_start)
i_end   = min(T - 1, i_end)

picked_times = np.empty(N)   # seconds relative to event origin
picked_rel   = np.empty(N)   # seconds relative to predicted P (for plotting)

# Use predicted P time (stored as origin time) to convert picks to absolute s
# The predicted P at each station is already baked into the alignment, so
# t_abs = EVT_TIME + p_tt is not directly available here.  Instead we fit
# using times_rel (relative to predicted P) converted to absolute by adding
# a per-station offset.  For the linear fit we just use times_rel picks and
# then add a common baseline later.

for i in range(N):
    segment = waveforms[i, i_start:i_end]
    idx_rel = np.argmax(np.abs(segment))
    picked_rel[i] = times_rel[i_start + idx_rel]

# ── linear moveout fit ────────────────────────────────────────────────────────
# Model: t_rel(d) = a + d / v_apparent
# where t_rel is the pick time relative to the predicted P,
# d is distance in km, and v_apparent is apparent velocity in km/s.
# A non-zero intercept 'a' absorbs any systematic offset in the theoretical P.

def linear_moveout(dist_km, a, v_app):
    return a + dist_km / v_app


p0 = [0.0, 8.0]   # initial guess: 0 s offset, 8 km/s apparent velocity
popt, pcov = curve_fit(linear_moveout, distances, picked_rel, p0=p0)
perr = np.sqrt(np.diag(pcov))

a_fit, v_fit = popt
a_err, v_err = perr

residuals = picked_rel - linear_moveout(distances, *popt)
rmse      = np.sqrt(np.mean(residuals ** 2))

print(f"\n── Moveout fit results ──────────────────────────────")
print(f"  Apparent Pn velocity : {v_fit:.2f} ± {v_err:.2f} km/s")
print(f"  Time offset (a)      : {a_fit:.2f} ± {a_err:.2f} s")
print(f"  RMSE of residuals    : {rmse:.2f} s")
print(f"  N stations used      : {N}")


# ── figure 1: record section ──────────────────────────────────────────────────

TRACE_SCALE = 80.0   # km — visual half-amplitude of each normalised trace

fig, ax = plt.subplots(figsize=(10, 12))

for i in range(N):
    d     = distances[i]
    trace = waveforms[i] * TRACE_SCALE
    ax.plot(times_rel, trace + d,
            color="k", linewidth=0.5, alpha=0.7)
    ax.plot(picked_rel[i], d, "r|", markersize=8, markeredgewidth=1.5)

# Fitted moveout line
d_range = np.linspace(distances.min(), distances.max(), 200)
ax.plot(linear_moveout(d_range, a_fit, v_fit), d_range,
        color="#e74c3c", linewidth=2.0, linestyle="--",
        label=f"Linear fit  v = {v_fit:.2f} km/s")

ax.axvline(0, color="steelblue", linewidth=1.0, linestyle=":",
           label="Predicted P (iasp91)")

ax.set_xlabel("Time relative to predicted P arrival (s)", fontsize=12)
ax.set_ylabel("Epicentral distance (km)", fontsize=12)
ax.set_title(
    f"P-Wave Record Section — 2016 Kaikoura M{evt_mag} Earthquake\n"
    f"GeoNet broadband network (NZ)  ·  {N} stations  ·  depth {evt_dep:.0f} km",
    fontsize=12,
)
ax.legend(fontsize=10, loc="upper left")
ax.set_xlim(times_rel[0], times_rel[-1])
ax.set_ylim(distances.min() - 100, distances.max() + 100)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, linewidth=0.5)

fig.tight_layout()
fig.savefig("record_section.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("\nSaved → record_section.png")


# ── figure 2: moveout fit + residuals ─────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                                 gridspec_kw={"height_ratios": [3, 1]})

# Top panel: observed picks + fitted line
ax1.scatter(distances, picked_rel,
            s=60, zorder=3, color="steelblue", edgecolors="white",
            linewidths=0.5, label="Observed P picks")
ax1.plot(d_range, linear_moveout(d_range, a_fit, v_fit),
         color="#e74c3c", linewidth=2.0,
         label=f"Linear fit:  t = {a_fit:.2f} + d / {v_fit:.2f}")

ax1.set_ylabel("P pick time relative to predicted P (s)", fontsize=11)
ax1.set_title(
    f"P-Wave Moveout Fit — 2016 Kaikoura M{evt_mag}\n"
    f"Apparent Pn velocity = {v_fit:.2f} ± {v_err:.2f} km/s",
    fontsize=12,
)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotate velocity result
ax1.annotate(
    f"v$_{{\\mathrm{{app}}}}$ = {v_fit:.2f} ± {v_err:.2f} km/s\n"
    f"RMSE = {rmse:.2f} s  ·  N = {N}",
    xy=(0.97, 0.05), xycoords="axes fraction",
    ha="right", va="bottom", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8),
)

# Bottom panel: residuals
ax2.axhline(0, color="k", linewidth=0.8)
ax2.bar(distances, residuals, width=20, color="steelblue",
        alpha=0.7, edgecolor="white")
ax2.set_xlabel("Epicentral distance (km)", fontsize=11)
ax2.set_ylabel("Residual (s)", fontsize=11)
ax2.set_title("Fit Residuals", fontsize=11)
ax2.grid(True, alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig("moveout_fit.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved → moveout_fit.png")
