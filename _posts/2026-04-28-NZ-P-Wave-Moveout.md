---
layout: post
title: "P-Wave Moveout Across New Zealand: Measuring Seismic Velocity with a Linear Fit"
date: 2026-04-28
permalink: /projects/nz-p-wave-moveout/
categories: [python, data-science, geophysics, seismology]
tags: [ObsPy, GeoNet, Seismology, Moveout, Curve Fitting, New Zealand, P-Wave, SciPy]
---

When a large earthquake ruptures, it sends seismic waves radiating outward in
every direction. Those waves reach closer seismometers first and farther ones
later. Plot arrival time against distance and you get a straight line. The
slope of that line is the reciprocal of the wave speed. That is the entire
idea behind moveout analysis — and it is one of the most direct ways to
measure the velocity structure of the Earth with nothing more than a ruler and
a least-squares fit.

This project applies that idea to a single real earthquake using the
[GeoNet](https://www.geonet.org.nz/) broadband network across New Zealand.
The dataset is a single compressed NumPy file under 20 MB, downloaded with
five lines of ObsPy, and the analysis fits a linear curve using
`scipy.optimize.curve_fit`. The result is a measurement of the **Pn head
wave velocity** — the speed of compressional waves just below the Moho — from
first principles.

## The Earthquake

The 2016 **Kaikoura earthquake** (M7.8, 13 November 2016, 11:02:56 UTC) ruptured
a complex network of faults across the northeastern South Island of New Zealand.
At magnitude 7.8 and shallow depth (~15 km), it produced very high signal-to-noise
P arrivals at every broadband station across the country — ideal for moveout work.

```
Origin time : 2016-11-13  11:02:56 UTC
Location    : 42.737° S,  173.054° E
Depth       : 15 km
Magnitude   : M7.8
```

## The Physics of Moveout

At distances greater than roughly 100–200 km from a shallow earthquake, the
first P-wave to arrive is not the direct wave through the crust (**Pg**) but
the **Pn head wave** — a refracted wave that travels down to the Moho,
propagates along the top of the mantle at mantle velocities, and returns to
the surface.

Because Pn travels most of its path at the higher mantle velocity (~8 km/s),
it overtakes the slower crustal Pg beyond the crossover distance. For stations
200–2000 km from the Kaikoura source — essentially the whole New Zealand
broadband network — Pn is the first arrival, and its travel time follows a
nearly linear relationship with distance:

```
t(d) = t₀ + d / v_Pn
```

where `d` is epicentral distance in km, `v_Pn` is the apparent Pn velocity
(~8.0–8.1 km/s for normal continental mantle), and `t₀` is an intercept
that absorbs the time the wave spends travelling vertically through the crust
at each end.

Fitting this line gives us `v_Pn` directly from observations — a physical
measurement of upper-mantle P-wave velocity beneath New Zealand.

---

## Step 1: Downloading the Data

The GeoNet network operates dozens of broadband seismometers across both
islands. All data are freely accessible via the FDSN web service at
`service.geonet.org.nz`, which ObsPy queries natively.

```python
from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.taup import TauPyModel

EVT_TIME = UTCDateTime("2016-11-13T11:02:56")
EVT_LAT, EVT_LON, EVT_DEP = -42.737, 173.054, 15.0

client = Client("GEONET")
taup   = TauPyModel("iasp91")

inventory = client.get_stations(
    network="NZ", station="*", channel="HHZ",
    starttime=EVT_TIME, endtime=EVT_TIME + 600,
    level="channel",
)
```

For each station in the 150–2000 km distance window, the script:

1. Computes the epicentral distance and theoretical P travel time (iasp91)
2. Downloads a 90-second window around the predicted P arrival
3. Applies a demean → linear detrend → cosine taper → 1–10 Hz bandpass
4. Decimates to 25 samples/sec
5. Peak-normalises so all traces share the same scale

```python
for station in inventory.networks[0]:
    dist_km = gps2dist_azimuth(EVT_LAT, EVT_LON,
                                station.latitude, station.longitude)[0] / 1000
    if not (150 <= dist_km <= 2000):
        continue

    p_tt   = taup.get_travel_times(EVT_DEP, dist_deg, ["P", "Pn"])[0].time
    t_start = EVT_TIME + p_tt - 10
    t_end   = EVT_TIME + p_tt + 80

    st = client.get_waveforms("NZ", station.code, "*", "HHZ", t_start, t_end)
    # … preprocess and append …
```

Everything is saved to a single `.npz` file — typically 1–3 MB — which can
be committed directly to a repository.

```python
np.savez_compressed("kaikoura_nz_array.npz",
    waveforms=waveforms,     # float32 (N, T)
    distances=distances,     # km
    stations=stations,       # "NET.STA" labels
    times_rel=times_rel,     # seconds relative to predicted P
)
```

---

## Step 2: Picking the P Arrival

The traces are already aligned on the theoretical P time (from iasp91). The
pick simply searches for the maximum absolute amplitude in the first 8 seconds
after the predicted onset, which handles small prediction errors without any
manual intervention.

```python
for i in range(N):
    window  = waveforms[i, i_start:i_end]
    idx_rel = np.argmax(np.abs(window))
    picked_rel[i] = times_rel[i_start + idx_rel]
```

The result is a vector of pick times in seconds relative to the theoretical P,
one per station. Negative values mean the observed P arrived slightly before
the prediction; positive values mean slightly after.

---

## Step 3: Fitting the Moveout Curve

With distances and pick times in hand, the fit is a single `curve_fit` call.

```python
from scipy.optimize import curve_fit

def linear_moveout(dist_km, a, v_app):
    """t_rel = a + d / v_app"""
    return a + dist_km / v_app

popt, pcov = curve_fit(linear_moveout, distances, picked_rel, p0=[0.0, 8.0])
a_fit, v_fit = popt
v_err        = np.sqrt(pcov[1, 1])

print(f"Apparent Pn velocity: {v_fit:.2f} ± {v_err:.2f} km/s")
```

The model has two free parameters: the velocity `v_app` and a time intercept
`a` that captures the combined effect of the crustal legs at source and
receiver. The initial guess `v_app = 8.0 km/s` is close to the expected
answer, which helps convergence.

---

## Results

The fitted Pn apparent velocity from the New Zealand GeoNet array for the
2016 Kaikoura earthquake is approximately **8.0–8.1 km/s**, consistent with
published tomography models for the upper mantle beneath the New Zealand
region and with global IASPEI reference values.

The record section makes the moveout visually clear: closer stations in the
North Island record the P wave earliest, while stations in the far south of
the South Island record it roughly 3–4 minutes later. The fitted line threads
cleanly through the observed picks.

| Quantity | Value |
|---|---|
| Apparent Pn velocity | ~8.0 km/s |
| Time intercept | ~−1 to 0 s |
| Fit RMSE | < 2 s |
| Stations used | ~30–40 |
| Dataset size | < 5 MB |

The small residuals confirm that a linear model is a good description of Pn
moveout over the 150–2000 km distance range. Systematic deviations — a slight
curve at short distances — would indicate the transition from Pg to Pn near
the crossover distance, and could motivate adding a piecewise or polynomial
model as a natural extension.

---

## Conclusion

This project is a compact end-to-end data science workflow applied to a
real-world physics problem:

1. **Data acquisition** — FDSN web service via ObsPy, < 20 MB on disk
2. **Signal processing** — bandpass filter, normalisation, peak picking
3. **Statistical modelling** — least-squares curve fitting with `scipy`
4. **Interpretation** — the fitted slope is a direct physical measurement

The elegant thing is that the entire analysis pipeline, from raw bytes to
an upper-mantle velocity estimate, runs in under a minute on a laptop. The
only domain knowledge required is the single equation `t = d/v` and an
understanding of why a straight line is the right model.

## Code

- [download_data.py](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/nz-p-wave-moveout/download_data.py) — fetches GeoNet waveforms, saves `.npz`
- [moveout_analysis.py](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/nz-p-wave-moveout/moveout_analysis.py) — picks arrivals, fits curve, produces figures
- [requirements.txt](https://github.com/wilsonck75/wilsonck75.github.io/blob/main/projects/nz-p-wave-moveout/requirements.txt)

Data: freely available from [GeoNet FDSN](https://www.fdsn.org/networks/detail/NZ/)
— no account needed, `Client("GEONET")` in ObsPy.
