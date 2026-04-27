"""
04_global_map.py
----------------
Reads H-κ stacking results (data/hk_results.csv) and produces publication-
quality figures:

  1. global_crustal_thickness.png  — Robinson projection scatter map,
                                     stations coloured by H
  2. hk_histogram.png              — Distribution of H and κ across all stations
  3. hk_scatter.png                — H vs κ coloured by tectonic setting proxy
                                     (continental / oceanic / mountain)
  4. hk_stack_example.png          — H-κ stack surface for one station

Run after 03_hk_stacking.py.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = Path("data")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

CMAP_H = "RdYlBu_r"
VMIN_H = 15.0   # km
VMAX_H = 75.0   # km


# ── helpers ────────────────────────────────────────────────────────────────────

def tectonic_label(lat: float, lon: float, H: float) -> str:
    """Very rough proxy: thick crust → mountain belt, thin → oceanic/rift."""
    if H >= 50:
        return "Mountain belt"
    if H <= 25:
        return "Oceanic / thin crust"
    return "Continental platform"


# ── figure 1 — global map ─────────────────────────────────────────────────────

def plot_global_map(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(18, 9))
    ax  = fig.add_subplot(1, 1, 1,
                           projection=ccrs.Robinson(central_longitude=0))
    ax.set_global()

    # Background features
    ax.add_feature(cfeature.LAND,      facecolor="#e8e0d8", zorder=0)
    ax.add_feature(cfeature.OCEAN,     facecolor="#c9dff0", zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5,  edgecolor="#555", zorder=1)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.25, edgecolor="#888", zorder=1)
    ax.add_feature(cfeature.LAKES,     facecolor="#c9dff0", zorder=1)

    # Plate boundaries (110 m resolution)
    plate_bnd = cfeature.NaturalEarthFeature(
        category="physical", name="geographic_lines",
        scale="110m", facecolor="none"
    )

    # Scatter: station locations coloured by crustal thickness
    norm = mcolors.Normalize(vmin=VMIN_H, vmax=VMAX_H)
    cmap = plt.get_cmap(CMAP_H)

    sc = ax.scatter(
        df["lon"].values,
        df["lat"].values,
        c=df["H"].values,
        cmap=cmap,
        norm=norm,
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.3,
        transform=ccrs.PlateCarree(),
        zorder=3,
    )

    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal",
                         shrink=0.55, pad=0.04, aspect=40)
    cbar.set_label("Crustal Thickness  H  (km)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.4,
                 color="gray", alpha=0.5, linestyle="--")

    ax.set_title(
        "Global Crustal Thickness from Teleseismic Receiver Function Analysis\n"
        f"H–κ stacking (Zhu & Kanamori 2000)  ·  {len(df)} broadband stations"
        "  ·  Networks: IU, II, G, GE",
        fontsize=12, pad=10
    )

    out = OUTPUT_DIR / "global_crustal_thickness.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ── figure 2 — histograms ─────────────────────────────────────────────────────

def plot_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(df["H"], bins=40, color="#2b7bba", edgecolor="white", linewidth=0.5)
    axes[0].axvline(df["H"].mean(), color="red", lw=1.5,
                    label=f"Mean = {df['H'].mean():.1f} km")
    axes[0].axvline(df["H"].median(), color="orange", lw=1.5, linestyle="--",
                    label=f"Median = {df['H'].median():.1f} km")
    axes[0].set_xlabel("Crustal Thickness H (km)", fontsize=11)
    axes[0].set_ylabel("Number of stations", fontsize=11)
    axes[0].set_title("Distribution of Crustal Thickness", fontsize=12)
    axes[0].legend(fontsize=9)

    axes[1].hist(df["kappa"], bins=40, color="#e06c22", edgecolor="white", linewidth=0.5)
    axes[1].axvline(df["kappa"].mean(), color="blue", lw=1.5,
                    label=f"Mean = {df['kappa'].mean():.3f}")
    axes[1].set_xlabel("Vp/Vs ratio (κ)", fontsize=11)
    axes[1].set_ylabel("Number of stations", fontsize=11)
    axes[1].set_title("Distribution of Vp/Vs Ratio", fontsize=12)
    axes[1].legend(fontsize=9)

    fig.suptitle("Global H–κ Results Summary", fontsize=13, y=1.01)
    fig.tight_layout()

    out = OUTPUT_DIR / "hk_histogram.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ── figure 3 — H vs κ scatter ─────────────────────────────────────────────────

def plot_hk_scatter(df: pd.DataFrame) -> None:
    labels   = df.apply(lambda r: tectonic_label(r["lat"], r["lon"], r["H"]), axis=1)
    settings = labels.unique()
    colors   = {"Mountain belt": "#c0392b",
                "Continental platform": "#27ae60",
                "Oceanic / thin crust": "#2980b9"}

    fig, ax = plt.subplots(figsize=(8, 6))
    for setting in settings:
        mask = labels == setting
        ax.scatter(df.loc[mask, "kappa"], df.loc[mask, "H"],
                   c=colors.get(setting, "gray"),
                   label=setting, s=30, alpha=0.7, edgecolors="none")

    ax.set_xlabel("Vp/Vs ratio (κ)", fontsize=11)
    ax.set_ylabel("Crustal Thickness H (km)", fontsize=11)
    ax.set_title("Crustal Thickness vs. Vp/Vs Ratio\nby Tectonic Setting", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)

    out = OUTPUT_DIR / "hk_scatter.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ── figure 4 — example H-κ stack surface ─────────────────────────────────────

def plot_hk_surface_example(df: pd.DataFrame) -> None:
    """
    Re-compute and plot the H-κ surface for the station closest to the
    global mean H. Requires the .npz file from script 02.
    """
    from importlib import import_module
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    hk_mod = import_module("03_hk_stacking")

    target_H  = df["H"].mean()
    idx_best  = (df["H"] - target_H).abs().idxmin()
    row       = df.loc[idx_best]
    sta_key   = f"{row['network']}.{row['station']}"
    npz_path  = DATA_DIR / "receiver_functions" / f"{sta_key}.npz"

    if not npz_path.exists():
        log.warning("NPZ file not found for %s — skipping surface plot.", sta_key)
        return

    data       = np.load(str(npz_path))
    rfs        = data["rfs"]
    times      = data["times"]
    ray_params = data["ray_params"]

    stack, H_arr, k_arr = hk_mod.hk_stack(rfs, ray_params, times)

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(k_arr, H_arr, stack, cmap="hot_r", shading="auto")
    fig.colorbar(pcm, ax=ax, label="Stacking Amplitude")

    H_best, k_best = hk_mod.find_best(stack, H_arr, k_arr)
    ax.plot(k_best, H_best, "b*", markersize=14,
            label=f"H = {H_best:.1f} km,  κ = {k_best:.3f}")

    ax.set_xlabel("Vp/Vs ratio (κ)", fontsize=11)
    ax.set_ylabel("Crustal Thickness H (km)", fontsize=11)
    ax.set_title(f"H–κ Stack Surface  ·  Station {sta_key}\n"
                 f"({len(rfs)} receiver functions)", fontsize=12)
    ax.invert_yaxis()
    ax.legend(fontsize=10)

    out = OUTPUT_DIR / "hk_stack_example.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved → %s", out)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    csv_path = DATA_DIR / "hk_results.csv"
    if not csv_path.exists():
        log.error("Results file not found: %s\nRun 03_hk_stacking.py first.", csv_path)
        return

    df = pd.read_csv(str(csv_path))
    log.info("Loaded %d stations from %s", len(df), csv_path)

    plot_global_map(df)
    plot_histograms(df)
    plot_hk_scatter(df)
    plot_hk_surface_example(df)

    log.info("All figures saved to %s/", OUTPUT_DIR)


if __name__ == "__main__":
    main()
