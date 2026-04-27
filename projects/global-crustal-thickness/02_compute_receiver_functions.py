"""
02_compute_receiver_functions.py
---------------------------------
Preprocesses downloaded waveforms and computes P-wave receiver functions
using iterative time-domain deconvolution (Ligorria & Ammon, 1999).

Pipeline per station-event:
  1. Load ZNE traces, attach response
  2. Detrend, demean, cosine taper
  3. Remove instrument response → velocity
  4. Bandpass filter 0.05 – 2.0 Hz
  5. Rotate ZNE → ZRT using event back-azimuth
  6. Iterative deconvolution: R / Z → radial RF
  7. Quality control (SNR, max amplitude)
  8. Save RF time series + metadata to data/receiver_functions/

Output per station:
    data/receiver_functions/NET.STA.npz
        rfs       : (N_events, N_samples)  receiver function traces
        times     : (N_samples,)           time axis relative to P (s)
        ray_params: (N_events,)            ray parameter (s/km)
        baz       : (N_events,)            back-azimuth (degrees)
        distances : (N_events,)            epicentral distance (degrees)
        sta_lat   : scalar
        sta_lon   : scalar
"""

import logging
from pathlib import Path

import numpy as np
from obspy import read, read_inventory, read_events, UTCDateTime
from obspy.geodetics import gps2dist_azimuth, locations2degrees
from obspy.taup import TauPyModel

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
WAV_DIR  = DATA_DIR / "waveforms"
RF_DIR   = DATA_DIR / "receiver_functions"

# Preprocessing parameters
FREQMIN      = 0.05     # Hz
FREQMAX      = 2.0      # Hz
GAUSS_PARAM  = 2.5      # Gaussian width factor (controls RF frequency content)
MAX_ITER     = 400      # iterative deconvolution iterations
CONV_TOL     = 0.001    # convergence criterion (fractional residual power)
SNR_MIN      = 3.0      # minimum signal-to-noise on vertical component
RF_MAX_AMP   = 1.5      # reject RFs with peak amplitude above this threshold
PRE_P        = 10.0     # seconds before P in output RF
POST_P       = 100.0    # seconds after P in output RF
SAMP_RATE    = 10.0     # target sample rate after resampling (Hz)

taup = TauPyModel(model="iasp91")


# ── signal processing ──────────────────────────────────────────────────────────

def preprocess(st, pre_filt=(0.01, 0.05, 10.0, 20.0)):
    """Detrend, taper, remove response, bandpass."""
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="cosine")
    st.remove_response(output="VEL", pre_filt=pre_filt)
    st.filter("bandpass", freqmin=FREQMIN, freqmax=FREQMAX, corners=4, zerophase=True)
    st.resample(SAMP_RATE)
    return st


def snr(trace, signal_window=(0.0, 30.0), noise_window=(-30.0, -1.0)):
    """Signal-to-noise ratio around the P onset (time zero = P arrival)."""
    dt = trace.stats.delta
    n  = len(trace.data)
    # The trace starts PRE_P seconds before P, so P is at index PRE_P/dt
    p_idx = int(PRE_P / dt)

    sig_start = p_idx + int(signal_window[0] / dt)
    sig_end   = p_idx + int(signal_window[1] / dt)
    noi_start = p_idx + int(noise_window[0] / dt)
    noi_end   = p_idx + int(noise_window[1] / dt)

    sig_start = max(0, sig_start); sig_end   = min(n, sig_end)
    noi_start = max(0, noi_start); noi_end   = min(n, noi_end)

    sig_rms = np.sqrt(np.mean(trace.data[sig_start:sig_end] ** 2))
    noi_rms = np.sqrt(np.mean(trace.data[noi_start:noi_end] ** 2))

    return sig_rms / noi_rms if noi_rms > 0 else 0.0


def iterative_deconvolution(radial: np.ndarray, vertical: np.ndarray,
                             dt: float) -> np.ndarray:
    """
    Time-domain iterative deconvolution (Ligorria & Ammon, 1999).

    Computes the radial receiver function by deconvolving the vertical (P)
    from the radial (SV) component in the time domain.

    Parameters
    ----------
    radial   : radial component, aligned on P arrival
    vertical : vertical component (source wavelet estimate)
    dt       : sample interval (s)

    Returns
    -------
    rf : receiver function (same length as input arrays)
    """
    npts = len(radial)
    nfft = 2 ** int(np.ceil(np.log2(npts)))

    # Gaussian filter in the frequency domain
    freq  = np.fft.rfftfreq(nfft, d=dt)
    gauss = np.exp(-(freq ** 2) / (4.0 * GAUSS_PARAM ** 2))

    R_fft = np.fft.rfft(radial,   n=nfft)
    V_fft = np.fft.rfft(vertical, n=nfft)

    R_g = np.fft.irfft(R_fft * gauss, n=nfft)[:npts]
    V_g = np.fft.irfft(V_fft * gauss, n=nfft)[:npts]

    # Denominator power for normalisation
    power = np.sum(V_g ** 2)
    if power == 0:
        return np.zeros(npts)

    spikes   = np.zeros(npts)
    residual = R_g.copy()
    r0_power = np.sum(residual ** 2)

    for _ in range(MAX_ITER):
        # Cross-correlate residual with source
        cc     = np.correlate(residual, V_g, mode="full")
        lag    = np.argmax(np.abs(cc)) - (npts - 1)
        amp    = cc[lag + npts - 1] / power

        # Accumulate spike
        if 0 <= lag < npts:
            spikes[lag] += amp

        # Subtract contribution from residual
        if lag >= 0:
            residual[:npts - lag] -= amp * V_g[lag:]
        else:
            residual[-lag:] -= amp * V_g[:npts + lag]

        # Convergence check
        if np.sum(residual ** 2) / r0_power < CONV_TOL:
            break

    # Apply Gaussian to accumulated spikes
    S_fft = np.fft.rfft(spikes, n=nfft)
    rf    = np.fft.irfft(S_fft * gauss, n=nfft)[:npts]
    return rf


# ── per-station processing ─────────────────────────────────────────────────────

def process_station(sta_dir: Path, inventory, catalog) -> dict | None:
    """
    Process all events for one station. Returns a dict of arrays or None if
    fewer than 5 usable receiver functions are found.
    """
    parts = sta_dir.name.split(".")
    if len(parts) != 2:
        return None
    net_code, sta_code = parts

    # Look up station coordinates
    try:
        net_inv = inventory.select(network=net_code, station=sta_code)
        sta     = net_inv[0][0]
    except (IndexError, Exception):
        log.warning("Station %s.%s not in inventory, skipping.", net_code, sta_code)
        return None

    sta_lat = sta.latitude
    sta_lon = sta.longitude

    rfs, ray_params, bazs, dists = [], [], [], []

    event_dirs = sorted(sta_dir.iterdir())
    for ev_dir in event_dirs:
        if not ev_dir.is_dir():
            continue

        mseed_files = list(ev_dir.glob("*.mseed"))
        if not mseed_files:
            continue

        try:
            st = read(str(ev_dir / "*.mseed"))
        except Exception as exc:
            log.debug("Could not read %s: %s", ev_dir, exc)
            continue

        # Match to catalog event
        ev_time_str = ev_dir.name                     # "YYYY-MM-DDTHH:MM:SS"
        ev_time     = UTCDateTime(ev_time_str)
        matching    = [e for e in catalog
                       if abs((e.preferred_origin() or e.origins[0]).time - ev_time) < 5]
        if not matching:
            continue
        event  = matching[0]
        origin = event.preferred_origin() or event.origins[0]
        ev_lat = origin.latitude
        ev_lon = origin.longitude
        ev_dep = origin.depth / 1000.0

        dist_m, az, baz = gps2dist_azimuth(sta_lat, sta_lon, ev_lat, ev_lon)
        dist_deg = locations2degrees(sta_lat, sta_lon, ev_lat, ev_lon)

        # Predicted P arrival time
        arrivals = taup.get_travel_times(source_depth_in_km=ev_dep,
                                         distance_in_degree=dist_deg,
                                         phase_list=["P"])
        if not arrivals:
            continue
        p_tt       = arrivals[0].time
        ray_param  = arrivals[0].ray_param_sec_degree / 111.195   # s/deg → s/km

        p_abs_time = origin.time + p_tt

        # Trim to consistent window around P
        try:
            st.trim(starttime=p_abs_time - PRE_P,
                    endtime=p_abs_time + POST_P,
                    pad=True, fill_value=0.0)
        except Exception:
            continue

        # Attach inventory for response removal
        st.attach_response(inventory)

        # Preprocess
        try:
            st = preprocess(st)
        except Exception as exc:
            log.debug("Preprocessing failed %s / %s: %s", sta_dir.name, ev_dir.name, exc)
            continue

        # Need Z, N, E
        if not all(st.select(component=c) for c in ("Z", "N", "E")):
            continue

        trZ = st.select(component="Z")[0]
        trN = st.select(component="N")[0]
        trE = st.select(component="E")[0]

        # Check SNR on vertical
        if snr(trZ) < SNR_MIN:
            continue

        # Rotate NE → RT
        from obspy.signal.rotate import rotate_ne_rt
        r_data, t_data = rotate_ne_rt(trN.data, trE.data, baz)

        # Ensure equal lengths
        npts = min(len(trZ.data), len(r_data))
        z_data = trZ.data[:npts]
        r_data = r_data[:npts]

        dt = trZ.stats.delta

        # Compute receiver function
        rf = iterative_deconvolution(r_data, z_data, dt)

        # Quality check: reject unstable RFs
        if np.max(np.abs(rf)) > RF_MAX_AMP:
            continue

        rfs.append(rf)
        ray_params.append(ray_param)
        bazs.append(baz)
        dists.append(dist_deg)

    if len(rfs) < 5:
        log.info("Only %d RFs for %s.%s — skipping.", len(rfs), net_code, sta_code)
        return None

    npts_out = int((PRE_P + POST_P) * SAMP_RATE) + 1
    rfs_arr  = np.vstack([r[:npts_out] if len(r) >= npts_out
                           else np.pad(r, (0, npts_out - len(r)))
                           for r in rfs])

    times = np.linspace(-PRE_P, POST_P, npts_out)

    log.info("%s.%s: %d receiver functions computed.", net_code, sta_code, len(rfs))
    return dict(rfs=rfs_arr,
                times=times,
                ray_params=np.array(ray_params),
                baz=np.array(bazs),
                distances=np.array(dists),
                sta_lat=sta_lat,
                sta_lon=sta_lon)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    RF_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading inventory and catalog …")
    inventory = read_inventory(str(DATA_DIR / "inventory.xml"))
    catalog   = read_events(str(DATA_DIR / "catalog.xml"))

    sta_dirs = sorted(d for d in WAV_DIR.iterdir() if d.is_dir())
    log.info("Processing %d station directories …", len(sta_dirs))

    for sta_dir in sta_dirs:
        result = process_station(sta_dir, inventory, catalog)
        if result is None:
            continue
        outfile = RF_DIR / f"{sta_dir.name}.npz"
        np.savez(str(outfile), **result)
        log.info("Saved → %s", outfile)

    log.info("Receiver function computation complete.")


if __name__ == "__main__":
    main()
