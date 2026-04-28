"""
download_data.py
----------------
Downloads vertical-component waveforms from the GeoNet broadband network (NZ)
for the 2016 Kaikoura M7.8 earthquake and saves them to a single compressed
NumPy archive (kaikoura_nz_array.npz) that is small enough to commit to a
repository (target < 20 MB; typical output is 1–3 MB).

Run once:
    python download_data.py

Requires:  obspy, numpy  (see requirements.txt)

Output file layout (kaikoura_nz_array.npz)
-------------------------------------------
waveforms   float32 (N, T)  – band-pass filtered velocity traces, one per row
distances   float64 (N,)    – epicentral distance for each station (km)
stations    str     (N,)    – "NET.STA" labels
times_rel   float64 (T,)    – time axis relative to P arrival (s)
                               0 = predicted P onset
evt_lat     scalar  – event latitude  (°)
evt_lon     scalar  – event longitude (°)
evt_dep     scalar  – event depth (km)
evt_time    scalar  – event origin time (POSIX timestamp)
evt_mag     scalar  – magnitude
sampling_rate scalar – samples per second of stored waveforms
"""

from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

# ── event: 2016 Kaikoura M7.8 ─────────────────────────────────────────────────
EVT_TIME = UTCDateTime("2016-11-13T11:02:56")
EVT_LAT  = -42.737
EVT_LON  = 173.054
EVT_DEP  = 15.0     # km
EVT_MAG  = 7.8

# ── download settings ─────────────────────────────────────────────────────────
NETWORK   = "NZ"          # GeoNet national network
CHANNEL   = "HHZ"         # high-gain vertical, 100 sps
DIST_MIN  = 150.0         # km  — below this, strong-motion clips HH channels
DIST_MAX  = 2000.0        # km
PRE_P     = 10.0          # seconds before predicted P
POST_P    = 80.0          # seconds after  predicted P
SAMP_OUT  = 25.0          # decimate to this rate (sps) before saving
BP_LOW    = 1.0           # bandpass low  corner (Hz)
BP_HIGH   = 10.0          # bandpass high corner (Hz)
OUT_FILE  = Path("kaikoura_nz_array.npz")

taup = TauPyModel("iasp91")


def p_travel_time(depth_km: float, dist_deg: float) -> float | None:
    """Predicted P/Pn travel time (s) from iasp91, or None if no arrival."""
    arrivals = taup.get_travel_times(depth_km, dist_deg,
                                      phase_list=["P", "Pn", "Pg"])
    if not arrivals:
        return None
    return min(a.time for a in arrivals)


def main() -> None:
    client = Client("GEONET")

    print("Fetching station inventory …")
    inventory = client.get_stations(
        network=NETWORK,
        station="*",
        channel=CHANNEL,
        starttime=EVT_TIME - 60,
        endtime=EVT_TIME + 600,
        level="channel",
    )

    npts_out = int((PRE_P + POST_P) * SAMP_OUT)
    waveforms, distances, station_ids = [], [], []

    total_stations = sum(len(net) for net in inventory)
    print(f"Checking {total_stations} stations in inventory …")

    for network in inventory:
        for station in network:
            dist_m, _, _ = gps2dist_azimuth(
                EVT_LAT, EVT_LON, station.latitude, station.longitude
            )
            dist_km  = dist_m / 1000.0
            dist_deg = locations2degrees(
                EVT_LAT, EVT_LON, station.latitude, station.longitude
            )

            if not (DIST_MIN <= dist_km <= DIST_MAX):
                continue

            p_tt = p_travel_time(EVT_DEP, dist_deg)
            if p_tt is None:
                continue

            t_start = EVT_TIME + p_tt - PRE_P
            t_end   = EVT_TIME + p_tt + POST_P

            try:
                st = client.get_waveforms(
                    network=NETWORK,
                    station=station.code,
                    location="*",
                    channel=CHANNEL,
                    starttime=t_start,
                    endtime=t_end,
                )
            except FDSNNoDataException:
                continue
            except Exception as exc:
                print(f"  {station.code}: {exc}")
                continue

            if not st:
                continue

            tr = st.select(component="Z")
            if not tr:
                tr = st[0:1]
            tr = tr[0].copy()

            # ── preprocess ────────────────────────────────────────────────────
            tr.detrend("demean")
            tr.detrend("linear")
            tr.taper(max_percentage=0.05, type="cosine")
            tr.filter("bandpass",
                      freqmin=BP_LOW, freqmax=BP_HIGH,
                      corners=4, zerophase=True)

            # Decimate to SAMP_OUT sps
            factor = int(round(tr.stats.sampling_rate / SAMP_OUT))
            if factor > 1:
                tr.decimate(factor, no_filter=True)

            # Normalise to peak amplitude so all traces share the same scale
            peak = np.max(np.abs(tr.data))
            if peak == 0:
                continue
            tr.data = tr.data.astype("float32") / peak

            # Trim / pad to exact output length
            data = tr.data[:npts_out]
            if len(data) < npts_out:
                data = np.pad(data, (0, npts_out - len(data)))

            waveforms.append(data)
            distances.append(dist_km)
            station_ids.append(f"{network.code}.{station.code}")
            print(f"  ✓  {network.code}.{station.code:6s}  {dist_km:6.0f} km")

    if not waveforms:
        print("No waveforms downloaded.  Check network connectivity and event parameters.")
        return

    times_rel = np.linspace(-PRE_P, POST_P, npts_out)

    np.savez_compressed(
        str(OUT_FILE),
        waveforms     = np.vstack(waveforms),
        distances     = np.array(distances, dtype="float64"),
        stations      = np.array(station_ids),
        times_rel     = times_rel,
        evt_lat       = EVT_LAT,
        evt_lon       = EVT_LON,
        evt_dep       = EVT_DEP,
        evt_time      = EVT_TIME.timestamp,
        evt_mag       = EVT_MAG,
        sampling_rate = SAMP_OUT,
    )

    size_mb = OUT_FILE.stat().st_size / 1e6
    print(f"\nSaved {len(waveforms)} stations → {OUT_FILE}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
