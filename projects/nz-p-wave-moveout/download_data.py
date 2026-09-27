"""
download_data.py
----------------
Downloads three-component waveforms from the GeoNet broadband network (NZ)
for the 2011 Tohoku M9.0 earthquake and saves them to a single compressed
NumPy archive small enough to commit to a repository (target < 20 MB;
typical output is 2–5 MB for three components).

The wave arrives from the NNW, so it sweeps south across New Zealand,
giving ~60 seconds of clear P moveout across the array.  The P-to-S
conversion at the Moho (Ps) appears on the radial component ~3–5 s after P.

Run once:
    python download_data.py

Output: tohoku_nz_array.npz
----------------------------
Z, N, E    float32 (N, T)  – ZNE traces, normalised by Z-component peak
R, T       float32 (N, T)  – radial/transverse (rotated from N/E via back-azimuth)
L, Q       float32 (N, T)  – ray-aligned (L along ray, Q perp in vertical plane)
                             obtained by rotating Z/R using the TauP incidence angle
distances  float64 (N,)    – epicentral distance from event (km)
dist_deg   float64 (N,)    – same in degrees
baz        float64 (N,)    – back-azimuth from station to event (degrees)
inc_angle  float64 (N,)    – TauP P-wave incidence angle at surface (degrees from vertical)
stations   str     (N,)    – "NET.STA"
times_rel  float64 (T,)    – seconds relative to predicted P arrival (0 = P)
evt_*                      – event scalars (lat, lon, dep, time, mag)
sampling_rate              – samples per second stored
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.signal.rotate import rotate_ne_rt
from obspy.taup import TauPyModel

# ── event ─────────────────────────────────────────────────────────────────────
EVT_TIME = UTCDateTime("2011-03-11T05:46:24")
EVT_LAT  =  38.297
EVT_LON  = 142.373
EVT_DEP  =  29.0     # km
EVT_MAG  =   9.0

# ── settings ──────────────────────────────────────────────────────────────────
NETWORK   = "NZ"
CHANNELS  = "HHZ,HHN,HHE"   # all three components at 100 sps
DIST_MIN  = 6_000.0          # km  (~54°) — all NZ stations qualify
DIST_MAX  = 9_500.0          # km  (~85°)
PRE_P     =    5.0           # s before predicted P
POST_P    =   45.0           # s after  predicted P  (captures P + Ps + gap)
SAMP_OUT  =   25.0           # decimate to 25 sps (Nyquist 12.5 Hz; better Ps resolution)
BP_LOW    =    0.5           # Hz — lower corner lets long-period Ps come through
BP_HIGH   =    5.0           # Hz
OUT_FILE  = Path("tohoku_nz_array.npz")

taup = TauPyModel("iasp91")


def first_p_time(depth_km: float, dist_deg: float) -> Optional[Tuple[float, float]]:
    """Return (travel_time_s, incidence_angle_deg) for the first P/Pdiff arrival."""
    arrivals = taup.get_travel_times(depth_km, dist_deg,
                                      phase_list=["P", "Pdiff"])
    if not arrivals:
        return None
    arr = min(arrivals, key=lambda a: a.time)
    return arr.time, arr.incident_angle


def preprocess(tr, factor: int) -> None:
    tr.detrend("demean")
    tr.detrend("linear")
    tr.taper(max_percentage=0.05, type="cosine")
    tr.filter("bandpass", freqmin=BP_LOW, freqmax=BP_HIGH,
              corners=4, zerophase=True)
    if factor > 1:
        tr.decimate(factor, no_filter=True)


def main() -> None:
    client = Client("GEONET")

    print("Fetching station inventory …")
    inventory = client.get_stations(
        network=NETWORK, station="*", channel="HHZ",
        starttime=EVT_TIME - 60, endtime=EVT_TIME + 1800,
        level="channel",
    )

    npts_out = int((PRE_P + POST_P) * SAMP_OUT)
    Z_list, N_list, E_list = [], [], []
    R_list, T_list = [], []
    L_list, Q_list = [], []
    dist_km_list, dist_deg_list, baz_list, inc_list, sta_list = [], [], [], [], []

    for network in inventory:
        for station in network:
            dist_m, az_to_evt, _ = gps2dist_azimuth(
                station.latitude, station.longitude, EVT_LAT, EVT_LON
            )
            dist_km  = dist_m / 1000.0
            dist_deg = locations2degrees(
                station.latitude, station.longitude, EVT_LAT, EVT_LON
            )
            baz = az_to_evt   # direction from station toward earthquake

            if not (DIST_MIN <= dist_km <= DIST_MAX):
                continue

            result = first_p_time(EVT_DEP, dist_deg)
            if result is None:
                continue
            p_tt, inc_angle = result

            t_start = EVT_TIME + p_tt - PRE_P
            t_end   = EVT_TIME + p_tt + POST_P

            try:
                st = client.get_waveforms(
                    NETWORK, station.code, "*", CHANNELS,
                    t_start, t_end,
                )
            except FDSNNoDataException:
                continue
            except Exception as exc:
                print(f"  {station.code}: {exc}")
                continue

            trZ = st.select(component="Z")
            trN = st.select(component="N")
            trE = st.select(component="E")
            if not (trZ and trN and trE):
                continue

            trZ, trN, trE = trZ[0].copy(), trN[0].copy(), trE[0].copy()

            factor = int(round(trZ.stats.sampling_rate / SAMP_OUT))
            for tr in (trZ, trN, trE):
                preprocess(tr, factor)

            # Ensure equal length before rotation
            n = min(len(trZ.data), len(trN.data), len(trE.data), npts_out)
            z_raw = trZ.data[:n]
            n_raw = trN.data[:n]
            e_raw = trE.data[:n]
            r_raw, t_raw = rotate_ne_rt(n_raw, e_raw, baz)

            # Pad to exact output length if needed
            def pad(arr):
                if len(arr) < npts_out:
                    arr = np.pad(arr, (0, npts_out - len(arr)))
                return arr[:npts_out]

            z_raw, n_raw, e_raw = pad(z_raw), pad(n_raw), pad(e_raw)
            r_raw, t_raw = pad(r_raw), pad(t_raw)

            # Normalise all components by Z peak so relative amplitudes are preserved
            peak = np.max(np.abs(z_raw))
            if peak == 0:
                continue

            z = z_raw / peak
            n_out = n_raw / peak
            e_out = e_raw / peak
            r = r_raw / peak
            t = t_raw / peak

            # ZR → LQ rotation using TauP incidence angle (i from vertical)
            # L is along the ray (≈Z for steep arrivals), Q is in-plane perpendicular
            i_rad = np.radians(inc_angle)
            l_out =  z * np.cos(i_rad) + r * np.sin(i_rad)
            q_out = -z * np.sin(i_rad) + r * np.cos(i_rad)

            Z_list.append(z.astype("float32"))
            N_list.append(n_out.astype("float32"))
            E_list.append(e_out.astype("float32"))
            R_list.append(r.astype("float32"))
            T_list.append(t.astype("float32"))
            L_list.append(l_out.astype("float32"))
            Q_list.append(q_out.astype("float32"))
            dist_km_list.append(dist_km)
            dist_deg_list.append(dist_deg)
            baz_list.append(baz)
            inc_list.append(inc_angle)
            sta_list.append(f"{network.code}.{station.code}")
            print(f"  ✓  {network.code}.{station.code:6s}  "
                  f"{dist_km:6.0f} km  baz={baz:.0f}°")

    if not Z_list:
        print("No waveforms downloaded.")
        return

    times_rel = np.linspace(-PRE_P, POST_P, npts_out)

    np.savez_compressed(
        str(OUT_FILE),
        Z             = np.vstack(Z_list),
        N             = np.vstack(N_list),
        E             = np.vstack(E_list),
        R             = np.vstack(R_list),
        T             = np.vstack(T_list),
        L             = np.vstack(L_list),
        Q             = np.vstack(Q_list),
        distances     = np.array(dist_km_list, dtype="float64"),
        dist_deg      = np.array(dist_deg_list, dtype="float64"),
        baz           = np.array(baz_list,      dtype="float64"),
        inc_angle     = np.array(inc_list,      dtype="float64"),
        stations      = np.array(sta_list),
        times_rel     = times_rel,
        evt_lat       = EVT_LAT,
        evt_lon       = EVT_LON,
        evt_dep       = EVT_DEP,
        evt_time      = EVT_TIME.timestamp,
        evt_mag       = EVT_MAG,
        sampling_rate = SAMP_OUT,
    )

    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"\nSaved {len(Z_list)} stations → {OUT_FILE}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
