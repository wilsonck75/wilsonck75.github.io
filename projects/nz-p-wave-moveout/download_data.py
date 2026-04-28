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
Z, R, T    float32 (N, T)  – ZRT traces, normalised by Z-component peak
distances  float64 (N,)    – epicentral distance from event (km)
dist_deg   float64 (N,)    – same in degrees
baz        float64 (N,)    – back-azimuth from station to event (degrees)
stations   str     (N,)    – "NET.STA"
times_rel  float64 (T,)    – seconds relative to predicted P arrival (0 = P)
evt_*                      – event scalars (lat, lon, dep, time, mag)
sampling_rate              – samples per second stored
"""

from pathlib import Path

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
SAMP_OUT  =   10.0           # decimate to 10 sps (Nyquist 5 Hz; fine for Ps)
BP_LOW    =    0.5           # Hz — lower corner lets long-period Ps come through
BP_HIGH   =    2.0           # Hz
OUT_FILE  = Path("tohoku_nz_array.npz")

taup = TauPyModel("iasp91")


def first_p_time(depth_km: float, dist_deg: float) -> float | None:
    arrivals = taup.get_travel_times(depth_km, dist_deg,
                                      phase_list=["P", "Pdiff"])
    return min((a.time for a in arrivals), default=None)


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
    Z_list, R_list, T_list = [], [], []
    dist_km_list, dist_deg_list, baz_list, sta_list = [], [], [], []

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

            p_tt = first_p_time(EVT_DEP, dist_deg)
            if p_tt is None:
                continue

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
            z = trZ.data[:n]
            r, t = rotate_ne_rt(trN.data[:n], trE.data[:n], baz)

            # Pad to exact output length if needed
            def pad(arr):
                if len(arr) < npts_out:
                    arr = np.pad(arr, (0, npts_out - len(arr)))
                return arr[:npts_out]

            z, r, t = pad(z), pad(r), pad(t)

            # Normalise all three by Z peak so relative amplitudes are preserved
            peak = np.max(np.abs(z))
            if peak == 0:
                continue

            Z_list.append((z / peak).astype("float32"))
            R_list.append((r / peak).astype("float32"))
            T_list.append((t / peak).astype("float32"))
            dist_km_list.append(dist_km)
            dist_deg_list.append(dist_deg)
            baz_list.append(baz)
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
        R             = np.vstack(R_list),
        T             = np.vstack(T_list),
        distances     = np.array(dist_km_list, dtype="float64"),
        dist_deg      = np.array(dist_deg_list, dtype="float64"),
        baz           = np.array(baz_list,      dtype="float64"),
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
