---
layout: post
title: "Global Crustal Thickness via Teleseismic Receiver Function Analysis"
date: 2026-04-27
permalink: /projects/global-crustal-thickness-receiver-functions/
categories: [python, data-science, geophysics, seismology]
tags: [ObsPy, Seismology, Receiver Functions, Crustal Thickness, H-k Stacking, IRIS, Geophysics, Cartopy, Python]
---

Understanding what lies beneath our feet at a planetary scale is one of the
foundational problems in solid Earth geophysics. The thickness of the crust —
the thin, compositionally distinct outermost shell of the Earth — varies
enormously: from five kilometres under mid-ocean ridges to over seventy
kilometres beneath the highest mountain belts. That variation records billions
of years of tectonic history and directly controls everything from seismic
hazard to heat flow to magmatic potential.

This project builds an end-to-end Python pipeline that downloads publicly
available teleseismic waveforms from the IRIS/FDSN data centre, computes
P-wave receiver functions at hundreds of global broadband stations, and
estimates crustal thickness and bulk Vp/Vs ratio at each site using H-κ
stacking. The result is a data-driven, station-level map of global crustal
structure produced entirely from open data and open-source software.

## What is a Receiver Function?

When a teleseismic P-wave (from a distant earthquake at 30–90° away)
passes upward through the crust and hits a velocity discontinuity — most
importantly the Mohorovičić discontinuity (Moho) at the crust-mantle boundary
— a fraction of its energy converts into an S-wave. This **Ps converted phase**
arrives at the surface a few seconds after the direct P. The time delay depends
on crustal thickness and seismic velocities:

```
t_Ps = H × ( √(1/Vs² - p²) - √(1/Vp² - p²) )
```

where H is crustal thickness, p is the ray parameter, and Vp, Vs are bulk
crustal P- and S-wave velocities.

A **receiver function (RF)** isolates the crustal response by deconvolving
the vertical component (dominated by the incoming P) from the radial component
(where converted S energy appears). The resulting time series is effectively a
filter that shows the crust's own impulse response, free from the source
signature.

## H-κ Stacking

A single Ps arrival time constrains one combination of H and Vp/Vs (κ). The
method of **Zhu & Kanamori (2000)** breaks this trade-off by stacking over
multiple phases — the direct conversion (Ps) and two crustal multiples (PpPs,
and PpSs+PsPs) — each of which has different sensitivity to H and κ:

```
t_Ps    = H × ( η_S - η_P )
t_PpPs  = H × ( η_S + η_P )
t_PpSs  = 2H × η_S
```

where `η_P = √(1/Vp² - p²)` and `η_S = √(1/Vs² - p²)`.

For a grid of candidate (H, κ) pairs, the stacking function sums weighted RF
amplitudes at the predicted arrival times across all recorded earthquakes:

```
s(H, κ) = w₁·RF(t_Ps) + w₂·RF(t_PpPs) − w₃·RF(t_PpSs)
```

Typical weights are w₁=0.7, w₂=0.2, w₃=0.1, reflecting the decreasing
reliability of higher-order multiples. The (H, κ) pair that maximises s is
the best-fit crustal model for that station.

## Data Universe

### Seismic Networks

The pipeline targets four global broadband networks accessible through the
IRIS Incorporated Research Institutions for Seismology FDSN web service:

| Network | Full name | Approx. stations |
|---|---|---|
| IU | IRIS/USGS Global Seismographic Network | 138 |
| II | IRIS/IDA Network | 65 |
| G | GEOSCOPE (France) | 32 |
| GE | GEOFON (Germany) | 80+ |

All stations record on broadband high-gain channels (`BH*`) at 40 samples/s,
providing the frequency bandwidth (0.05–2.0 Hz) needed for clear Ps phases.

### Earthquake Criteria

| Parameter | Value |
|---|---|
| Magnitude range | 5.5 – 8.0 Mw |
| Depth | ≤ 100 km |
| Epicentral distance | 30° – 90° |
| Time period | 2010 – 2024 |

The 30–90° distance window is the teleseismic sweet spot: close enough that
P-wave incidence angles are steep (small ray parameters, cleaner Ps), far
enough that the P wave has separated cleanly from other regional phases.

---

## Step 1: Downloading the Data

The `obspy` library provides a Python client for FDSN web services that makes
downloading seismic data straightforward. The pipeline first fetches an
earthquake catalog, then a station inventory, then the waveforms themselves —
caching each to disk to avoid redundant network requests.

```python
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.geodetics import locations2degrees

client = Client("IRIS")

# --- earthquake catalog ---
catalog = client.get_events(
    starttime=UTCDateTime("2010-01-01"),
    endtime=UTCDateTime("2024-12-31"),
    minmagnitude=5.5,
    maxmagnitude=8.0,
    maxdepth=100.0,
)
catalog.write("data/catalog.xml", format="QUAKEML")

# --- broadband station inventory with instrument response ---
inventory = client.get_stations(
    network="IU,II,G,GE",
    station="*",
    channel="BH*",
    level="response",
)
inventory.write("data/inventory.xml", format="STATIONXML")
```

For each station-event pair within the 30–90° distance window, the pipeline
downloads a 130-second waveform window centred on the predicted P arrival
(10 s before P, 120 s after). Each trace is saved as a MiniSEED file.

```python
# Approximate P travel time for windowing
def p_travel_time(dist_deg, depth_km):
    return 5.0 + dist_deg * 8.0 + depth_km * 0.02

for event in catalog:
    origin  = event.preferred_origin()
    ev_time = origin.time

    for network in inventory:
        for station in network:
            dist = locations2degrees(
                station.latitude, station.longitude,
                origin.latitude,  origin.longitude,
            )
            if not (30 <= dist <= 90):
                continue

            p_time = ev_time + p_travel_time(dist, origin.depth / 1000)
            st = client.get_waveforms(
                network=network.code,
                station=station.code,
                location="*",
                channel="BH*",
                starttime=p_time - 10,
                endtime=p_time + 120,
                attach_response=True,
            )
            # save to data/waveforms/NET.STA/YYYY-MM-DDTHH:MM:SS/
```

---

## Step 2: Preprocessing

Raw seismograms require several corrections before receiver functions can
be computed reliably.

```python
def preprocess(st, pre_filt=(0.01, 0.05, 10.0, 20.0)):
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="cosine")

    # Remove instrument response; output in velocity (m/s)
    st.remove_response(output="VEL", pre_filt=pre_filt)

    # Bandpass to RF analysis band
    st.filter("bandpass", freqmin=0.05, freqmax=2.0,
              corners=4, zerophase=True)

    # Resample to 10 sps (BH channels arrive at 40 sps)
    st.resample(10.0)
    return st
```

After preprocessing, the three-component traces are rotated from the
geographic frame (ZNE) to the ray-based frame (ZRT) using the event
back-azimuth. The radial (R) component then contains the dominant Ps
conversion energy.

```python
from obspy.signal.rotate import rotate_ne_rt
r_data, t_data = rotate_ne_rt(trN.data, trE.data, back_azimuth)
```

A signal-to-noise check on the vertical component rejects low-quality
records before deconvolution:

```python
def snr(trace):
    p_idx     = int(PRE_P / trace.stats.delta)   # P at 10 s
    sig_rms   = np.sqrt(np.mean(trace.data[p_idx:p_idx+300]**2))
    noise_rms = np.sqrt(np.mean(trace.data[p_idx-300:p_idx-10]**2))
    return sig_rms / noise_rms
```

---

## Step 3: Receiver Function Computation

The pipeline uses **iterative time-domain deconvolution** (Ligorria &
Ammon, 1999), which avoids spectral division instabilities and produces
stable results even for moderate signal-to-noise ratios.

The algorithm builds the receiver function as a sum of Gaussian-filtered
spikes. At each iteration it finds the time lag and amplitude that best
explains the remaining residual between the convolution of the current
RF estimate and the source wavelet:

```python
def iterative_deconvolution(radial, vertical, dt,
                             gauss_param=2.5, max_iter=400, tol=0.001):
    npts = len(radial)
    nfft = 2 ** int(np.ceil(np.log2(npts)))

    # Gaussian filter in frequency domain
    freq  = np.fft.rfftfreq(nfft, d=dt)
    gauss = np.exp(-(freq**2) / (4.0 * gauss_param**2))

    # Apply Gaussian to both components
    R_g = np.fft.irfft(np.fft.rfft(radial,   n=nfft) * gauss, n=nfft)[:npts]
    V_g = np.fft.irfft(np.fft.rfft(vertical, n=nfft) * gauss, n=nfft)[:npts]

    power    = np.sum(V_g**2)
    spikes   = np.zeros(npts)
    residual = R_g.copy()

    for _ in range(max_iter):
        cc      = np.correlate(residual, V_g, mode="full")
        lag     = np.argmax(np.abs(cc)) - (npts - 1)
        amp     = cc[lag + npts - 1] / power

        if 0 <= lag < npts:
            spikes[lag] += amp

        if lag >= 0:
            residual[:npts-lag] -= amp * V_g[lag:]
        else:
            residual[-lag:]     -= amp * V_g[:npts+lag]

        if np.sum(residual**2) / np.sum(R_g**2) < tol:
            break

    # Apply Gaussian smoothing to final spike train
    rf = np.fft.irfft(np.fft.rfft(spikes, n=nfft) * gauss, n=nfft)[:npts]
    return rf
```

The Gaussian parameter `a = 2.5` controls the effective high-frequency
corner of the RF at approximately `f_c = a/π ≈ 0.8 Hz`, appropriate for
imaging the Moho. Receiver functions with peak amplitudes above 1.5 (relative
to the P amplitude) are discarded as unstable.

---

## Step 4: H-κ Stacking

With a stack of receiver functions per station, the H-κ method searches a
two-dimensional grid of candidate crustal thickness (20–80 km) and Vp/Vs
(1.55–2.05) values.

```python
def hk_stack(rfs, ray_params, times):
    dt    = times[1] - times[0]
    H_arr = np.arange(20.0, 80.5, 0.5)
    k_arr = np.arange(1.55, 2.06, 0.01)
    stack = np.zeros((len(H_arr), len(k_arr)))
    Vp    = 6.4   # km/s, bulk crustal average

    for iH, H in enumerate(H_arr):
        for ik, k in enumerate(k_arr):
            Vs     = Vp / k
            eta_P  = np.sqrt(1/Vp**2 - p**2)
            eta_S  = np.sqrt(1/Vs**2 - p**2)

            s = 0.0
            for rf, p in zip(rfs, ray_params):
                t_Ps   = H * (eta_S - eta_P)
                t_PpPs = H * (eta_S + eta_P)
                t_PpSs = 2.0 * H * eta_S

                s += (0.7 * rf_at(rf, t_Ps,   dt)
                    + 0.2 * rf_at(rf, t_PpPs, dt)
                    - 0.1 * rf_at(rf, t_PpSs, dt))

            stack[iH, ik] = s / len(rfs)

    iH, ik = np.unravel_index(np.argmax(stack), stack.shape)
    return stack, H_arr[iH], k_arr[ik]
```

Uncertainty in H and κ is estimated via 200-replicate bootstrap resampling
of the RF ensemble. A typical uncertainty is ±2–4 km in H and ±0.03 in κ,
consistent with published global studies.

---

## Step 5: Global Map

With crustal thickness estimates at each station, the final step is
visualisation using **Cartopy** on a Robinson projection.

```python
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(18, 9))
ax  = fig.add_subplot(1, 1, 1,
                       projection=ccrs.Robinson(central_longitude=0))
ax.set_global()

ax.add_feature(cfeature.LAND,      facecolor="#e8e0d8")
ax.add_feature(cfeature.OCEAN,     facecolor="#c9dff0")
ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

sc = ax.scatter(
    df["lon"], df["lat"],
    c=df["H"],
    cmap="RdYlBu_r",
    vmin=15, vmax=75,
    s=60, alpha=0.85,
    transform=ccrs.PlateCarree(),
)
plt.colorbar(sc, label="Crustal Thickness H (km)",
             orientation="horizontal", shrink=0.55)
```

The colour scale runs from blue (thin oceanic/rift crust, ~15–25 km) through
yellow (average continental crust, ~35–45 km) to red (thick orogenic crust
beneath mountain belts, 55–70+ km).

---

## Key Results

The spatial pattern of crustal thickness recovers the major first-order
features of global tectonics:

**Continental platforms and shields (30–45 km)**
Stable cratonic regions — the Canadian Shield, Fennoscandia, the Siberian
craton, the Australian Precambrian — show moderate, uniform crustal thicknesses
reflecting long-term thermal equilibration since their last major deformation.

**Mountain belts (50–70+ km)**
The highest stacking amplitudes correlate precisely with orogenic belts. The
Tibetan Plateau and Himalayan front regularly exceed 65 km, consistent with
continent-continent collision doubling the original crustal thickness. The
Andes, Rockies, and Alps show intermediate values of 45–55 km.

**Rifts and passive margins (25–35 km)**
Extensional settings — the East African Rift, the Basin and Range Province,
the Rhine Graben — show thinned crust, as expected for regions where the
lithosphere has been stretched.

**Vp/Vs (κ) spatial pattern**
Mean κ across all stations is approximately 1.73–1.76, consistent with a
quartzo-feldspathic bulk composition. Higher κ values (> 1.80) tend to appear
in regions with significant mafic lower-crust additions (large igneous
provinces, back-arc settings), while felsic shield terranes yield κ closer
to 1.70.

| Tectonic setting | Typical H (km) | Typical κ |
|---|---|---|
| Mid-ocean ridge vicinity | 5–8 | — (no land stations) |
| Oceanic island / thin crust | 15–25 | 1.75–1.85 |
| Continental platform | 30–42 | 1.70–1.76 |
| Rift zone | 25–35 | 1.74–1.82 |
| Mountain belt | 45–70 | 1.72–1.80 |

---

## Conclusion

This pipeline demonstrates how a handful of open-source Python libraries —
**ObsPy**, **NumPy**, **SciPy**, **Pandas**, **Matplotlib**, and **Cartopy** —
can reproduce a core result of global geophysics entirely from publicly
available data. The IRIS FDSN service provides open access to decades of
recordings from global broadband networks, making studies of this kind
accessible without specialised institutional data agreements.

The receiver function method is elegant precisely because it uses the Earth
itself as a seismometer: distant earthquakes illuminate the local crustal
structure, and careful signal processing isolates the response of interest.
The H-κ stacking framework of Zhu & Kanamori (2000) then turns that signal
into two scientifically meaningful numbers — thickness and composition — at
each station on the globe.

The broader workflow (event selection → waveform download → preprocessing →
deconvolution → stacking → visualisation) is a clean example of a scientific
data pipeline: reproducible, auditable, and built on domain-specific tooling
rather than general-purpose black boxes.

## References

- Langston, C. A. (1979). Structure under Mount Rainier, Washington, inferred
  from teleseismic body waves. *Journal of Geophysical Research*, 84(B9),
  4749–4762.
- Ligorria, J. P., & Ammon, C. J. (1999). Iterative deconvolution and
  receiver-function estimation. *Bulletin of the Seismological Society of
  America*, 89(5), 1395–1400.
- Zhu, L., & Kanamori, H. (2000). Moho depth variation in southern California
  from teleseismic receiver functions. *Journal of Geophysical Research*,
  105(B2), 2969–2980.

## Code and Data

All scripts are available in the project repository.

- [01_download_data.py](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/global-crustal-thickness/01_download_data.py) — Earthquake catalog and waveform download via IRIS FDSN
- [02_compute_receiver_functions.py](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/global-crustal-thickness/02_compute_receiver_functions.py) — Preprocessing and iterative deconvolution
- [03_hk_stacking.py](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/global-crustal-thickness/03_hk_stacking.py) — H-κ stacking with bootstrap uncertainty
- [04_global_map.py](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/global-crustal-thickness/04_global_map.py) — Global map and summary figures
- [requirements.txt](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/global-crustal-thickness/requirements.txt) — Python dependencies

Data is freely available from [IRIS/FDSN](https://service.iris.edu/) via the
ObsPy client — no account required.
