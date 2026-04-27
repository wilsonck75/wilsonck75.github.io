"""
03_hk_stacking.py
-----------------
Estimates crustal thickness (H) and Vp/Vs ratio (κ) at each seismic
station using the H-κ stacking method of Zhu & Kanamori (2000).

The method stacks receiver function amplitudes at predicted arrival times
for three phases — Ps, PpPs, PpSs+PsPs — over a 2-D grid of (H, κ) values.
The grid maximum gives the best-fit crustal parameters.

Uncertainty is estimated via a bootstrap resampling of the RF stack.

Output:
    data/hk_results.csv    -- columns: network, station, lat, lon, H, kappa,
                               H_std, kappa_std, n_rfs
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
RF_DIR   = DATA_DIR / "receiver_functions"

# H-κ search grid
H_MIN    = 20.0     # km
H_MAX    = 80.0     # km
H_STEP   =  0.5     # km
K_MIN    =  1.55    # Vp/Vs
K_MAX    =  2.05
K_STEP   =  0.01

# Reference Vp for the bulk crust (km/s)
VP       = 6.4

# Phase weights  w1*Ps + w2*PpPs - w3*(PpSs+PsPs)
WEIGHTS  = (0.7, 0.2, 0.1)

# Bootstrap parameters
N_BOOT   = 200   # bootstrap replicates for uncertainty


# ── core stacking ──────────────────────────────────────────────────────────────

def predicted_times(H: float, k: float, p: float) -> tuple[float, float, float]:
    """
    Predicted arrival times relative to direct P for three multiples.

    Parameters
    ----------
    H : crustal thickness (km)
    k : Vp/Vs ratio
    p : ray parameter (s/km)

    Returns
    -------
    t_Ps, t_PpPs, t_PpSs  (seconds)
    """
    Vs     = VP / k
    eta_P  = np.sqrt(max(1.0 / VP**2 - p**2, 0.0))
    eta_S  = np.sqrt(max(1.0 / Vs**2 - p**2, 0.0))

    t_Ps   = H * (eta_S - eta_P)
    t_PpPs = H * (eta_S + eta_P)
    t_PpSs = 2.0 * H * eta_S
    return t_Ps, t_PpPs, t_PpSs


def rf_amplitude_at(rf: np.ndarray, t: float, dt: float) -> float:
    """Linear-interpolated RF amplitude at time t (seconds relative to P)."""
    PRE_P = 10.0   # seconds before P in the stored RF (set in script 02)
    idx   = (t + PRE_P) / dt
    i0    = int(idx)
    frac  = idx - i0
    n     = len(rf)
    if i0 < 0 or i0 >= n - 1:
        return 0.0
    return (1.0 - frac) * rf[i0] + frac * rf[i0 + 1]


def hk_stack(rfs: np.ndarray, ray_params: np.ndarray,
             times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the H-κ stack over the search grid.

    Parameters
    ----------
    rfs        : (N, T) receiver function array
    ray_params : (N,)  ray parameters (s/km)
    times      : (T,)  time axis (s relative to P)

    Returns
    -------
    stack : (nH, nK) stacking amplitude array
    H_arr : crustal thickness grid (km)
    k_arr : Vp/Vs grid
    """
    dt    = times[1] - times[0]
    H_arr = np.arange(H_MIN, H_MAX + H_STEP, H_STEP)
    k_arr = np.arange(K_MIN, K_MAX + K_STEP, K_STEP)
    stack = np.zeros((len(H_arr), len(k_arr)))
    w1, w2, w3 = WEIGHTS

    for iH, H in enumerate(H_arr):
        for ik, k in enumerate(k_arr):
            s = 0.0
            for rf, p in zip(rfs, ray_params):
                t_Ps, t_PpPs, t_PpSs = predicted_times(H, k, p)
                s += (w1 * rf_amplitude_at(rf, t_Ps,   dt)
                    + w2 * rf_amplitude_at(rf, t_PpPs, dt)
                    - w3 * rf_amplitude_at(rf, t_PpSs, dt))
            stack[iH, ik] = s / len(rfs)

    return stack, H_arr, k_arr


def find_best(stack: np.ndarray, H_arr: np.ndarray,
              k_arr: np.ndarray) -> tuple[float, float]:
    iH, ik = np.unravel_index(np.argmax(stack), stack.shape)
    return float(H_arr[iH]), float(k_arr[ik])


def bootstrap_uncertainty(rfs: np.ndarray, ray_params: np.ndarray,
                           times: np.ndarray) -> tuple[float, float]:
    """Bootstrap standard deviations for H and κ."""
    n      = len(rfs)
    H_boot = np.empty(N_BOOT)
    k_boot = np.empty(N_BOOT)

    for b in range(N_BOOT):
        idx    = np.random.randint(0, n, size=n)
        st, H_arr, k_arr = hk_stack(rfs[idx], ray_params[idx], times)
        H_boot[b], k_boot[b] = find_best(st, H_arr, k_arr)

    return float(H_boot.std()), float(k_boot.std())


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)

    rf_files = sorted(RF_DIR.glob("*.npz"))
    if not rf_files:
        log.error("No receiver function files found in %s", RF_DIR)
        return

    log.info("Running H-κ stacking on %d stations …", len(rf_files))
    records = []

    for rf_file in rf_files:
        stem  = rf_file.stem          # "NET.STA"
        parts = stem.split(".")
        if len(parts) != 2:
            continue
        net_code, sta_code = parts

        data       = np.load(str(rf_file))
        rfs        = data["rfs"]
        times      = data["times"]
        ray_params = data["ray_params"]
        sta_lat    = float(data["sta_lat"])
        sta_lon    = float(data["sta_lon"])
        n_rfs      = len(rfs)

        log.info("H-κ stack: %s.%s  (%d RFs) …", net_code, sta_code, n_rfs)

        stack, H_arr, k_arr = hk_stack(rfs, ray_params, times)
        H_best, k_best      = find_best(stack, H_arr, k_arr)
        H_std, k_std        = bootstrap_uncertainty(rfs, ray_params, times)

        records.append(dict(
            network=net_code,
            station=sta_code,
            lat=sta_lat,
            lon=sta_lon,
            H=round(H_best, 1),
            kappa=round(k_best, 3),
            H_std=round(H_std, 1),
            kappa_std=round(k_std, 3),
            n_rfs=n_rfs,
        ))

        log.info("  → H = %.1f ± %.1f km,  κ = %.3f ± %.3f",
                 H_best, H_std, k_best, k_std)

    results = pd.DataFrame(records)
    out_csv = DATA_DIR / "hk_results.csv"
    results.to_csv(str(out_csv), index=False)
    log.info("Saved %d station results → %s", len(results), out_csv)

    # Summary statistics
    log.info("\n── Global Crustal Thickness Summary ──")
    log.info("  Stations processed : %d", len(results))
    log.info("  Mean H             : %.1f km", results["H"].mean())
    log.info("  Std H              : %.1f km", results["H"].std())
    log.info("  Min / Max H        : %.1f / %.1f km",
             results["H"].min(), results["H"].max())
    log.info("  Mean Vp/Vs (κ)    : %.3f", results["kappa"].mean())


if __name__ == "__main__":
    main()
