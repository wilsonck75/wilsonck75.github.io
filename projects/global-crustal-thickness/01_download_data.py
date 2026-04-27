"""
01_download_data.py
-------------------
Downloads teleseismic earthquake waveforms from IRIS/FDSN for receiver
function analysis. Targets global broadband networks (IU, II, G, GE) and
events in the epicentral distance window 30–90° where P-to-S conversions
at the Moho are cleanest.

Output directory structure:
    data/
      catalog.xml           -- ObsPy Catalog (QuakeML)
      inventory.xml         -- ObsPy Inventory (StationXML)
      waveforms/
        NET.STA/
          YYYY-MM-DDTHH:MM:SS/
            NET.STA.LOC.CHA.mseed
"""

import os
import logging
from pathlib import Path

from obspy import UTCDateTime
from obspy.clients.fdsn import Client
from obspy.clients.fdsn.header import FDSNNoDataException
from obspy.geodetics import locations2degrees

# ── configuration ──────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
WAV_DIR    = DATA_DIR / "waveforms"
START      = UTCDateTime("2010-01-01")
END        = UTCDateTime("2024-12-31")
MIN_MAG    = 5.5
MAX_MAG    = 8.0
MAX_DEPTH  = 100.0        # km  — deep events degrade P waveform quality
DIST_MIN   = 30.0         # degrees
DIST_MAX   = 90.0         # degrees
PRE_P      = 10.0         # seconds before predicted P arrival
POST_P     = 120.0        # seconds after P — captures Ps, PpPs, PpSs

NETWORKS   = "IU,II,G,GE"
CHANNELS   = "BH*"        # broadband high-gain
LEVEL      = "response"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────────────────

def p_travel_time(dist_deg: float, depth_km: float) -> float:
    """
    Rough iasp91-based estimate of P travel time (seconds).
    Good enough for windowing; ObsPy TauPy gives exact values when needed.
    """
    # Empirical linear fit valid for 30–90°
    return 5.0 + dist_deg * 8.0 + depth_km * 0.02


def event_id(event) -> str:
    origin = event.preferred_origin() or event.origins[0]
    return origin.time.strftime("%Y-%m-%dT%H:%M:%S")


# ── main download logic ─────────────────────────────────────────────────────────

def download_catalog(client: Client) -> object:
    catalog_path = DATA_DIR / "catalog.xml"
    if catalog_path.exists():
        log.info("Loading cached catalog from %s", catalog_path)
        from obspy import read_events
        return read_events(str(catalog_path))

    log.info("Fetching earthquake catalog from IRIS …")
    catalog = client.get_events(
        starttime=START,
        endtime=END,
        minmagnitude=MIN_MAG,
        maxmagnitude=MAX_MAG,
        maxdepth=MAX_DEPTH,
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.write(str(catalog_path), format="QUAKEML")
    log.info("Saved %d events → %s", len(catalog), catalog_path)
    return catalog


def download_inventory(client: Client) -> object:
    inv_path = DATA_DIR / "inventory.xml"
    if inv_path.exists():
        log.info("Loading cached inventory from %s", inv_path)
        from obspy import read_inventory
        return read_inventory(str(inv_path))

    log.info("Fetching station inventory …")
    inventory = client.get_stations(
        network=NETWORKS,
        station="*",
        channel=CHANNELS,
        level=LEVEL,
        starttime=START,
        endtime=END,
    )
    inventory.write(str(inv_path), format="STATIONXML")
    log.info("Saved inventory → %s", inv_path)
    return inventory


def download_waveforms(client: Client, catalog, inventory) -> None:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    saved_pairs = 0

    for event in catalog:
        origin = event.preferred_origin() or event.origins[0]
        ev_lat  = origin.latitude
        ev_lon  = origin.longitude
        ev_dep  = origin.depth / 1000.0   # m → km
        ev_time = origin.time

        for network in inventory:
            for station in network:
                sta_lat = station.latitude
                sta_lon = station.longitude

                dist = locations2degrees(ev_lat, ev_lon, sta_lat, sta_lon)
                if not (DIST_MIN <= dist <= DIST_MAX):
                    continue

                total_pairs += 1
                p_time = ev_time + p_travel_time(dist, ev_dep)
                t_start = p_time - PRE_P
                t_end   = p_time + POST_P

                out_dir = WAV_DIR / f"{network.code}.{station.code}" / event_id(event)
                if out_dir.exists() and any(out_dir.iterdir()):
                    saved_pairs += 1
                    continue

                try:
                    st = client.get_waveforms(
                        network=network.code,
                        station=station.code,
                        location="*",
                        channel=CHANNELS,
                        starttime=t_start,
                        endtime=t_end,
                        attach_response=True,
                    )
                except FDSNNoDataException:
                    continue
                except Exception as exc:
                    log.warning("Failed %s.%s / %s: %s",
                                network.code, station.code, event_id(event), exc)
                    continue

                if len(st) < 3:
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)
                for tr in st:
                    fname = f"{tr.id}.mseed"
                    tr.write(str(out_dir / fname), format="MSEED")

                saved_pairs += 1
                if saved_pairs % 100 == 0:
                    log.info("Saved %d / %d station-event pairs …",
                             saved_pairs, total_pairs)

    log.info("Download complete. Saved %d pairs (of %d in distance window).",
             saved_pairs, total_pairs)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    client = Client("IRIS")
    catalog   = download_catalog(client)
    inventory = download_inventory(client)
    download_waveforms(client, catalog, inventory)


if __name__ == "__main__":
    main()
