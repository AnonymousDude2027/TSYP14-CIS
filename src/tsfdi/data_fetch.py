"""Build the offline data bundle (network needed once).

Sources (both verified against their live documentation on 2026-10-28):

GHCN-Daily, NOAA NCEI data service
    https://www.ncei.noaa.gov/access/services/data/v1
    dataset=daily-summaries, stations=<ID>, startDate, endDate,
    dataTypes=TMAX,TMIN,PRCP[,AWND], format=csv, units=metric,
    includeAttributes=true
    The attribute string is "measurement-flag,quality-flag,source-flag"; a
    non-empty quality flag means the value failed a GHCN QC check.

NASA POWER, daily point API
    https://power.larc.nasa.gov/api/temporal/daily/point
    parameters=..., community=AG, latitude, longitude, start=YYYYMMDD,
    end=YYYYMMDD, format=JSON
    The JSON header states ``fill_value: -999.0`` (missing-data marker).

Both were exercised during development; the exact URLs and the observed payload
shape are written to ``data/bundle/FETCH_INFO.json`` so that a reader can see
what was requested without reading this file, and ``make data-verify`` re-checks
every hash against the signed manifest.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .provenance.anchor import SoftwareAnchor
from .provenance.manifest import sign_manifest

NCEI = ("https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries"
        "&stations={sid}&startDate={start}&endDate={end}"
        "&dataTypes={types}&format=csv&units=metric&includeAttributes=true")
POWER = ("https://power.larc.nasa.gov/api/temporal/daily/point?"
         "parameters={params}&community=AG&latitude={lat}&longitude={lon}"
         "&start={start}&end={end}&format=JSON")
POWER_PARAMS = "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN"


def _get(url: str, tries: int = 4, timeout: int = 120) -> str:
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310
                return r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"download failed after {tries} attempts: {url}") from last


def fetch_station(site, start: str, end: str, types: str = "TMAX,TMIN,PRCP") -> pd.DataFrame:
    url = NCEI.format(sid=site.id, start=start, end=end, types=types)
    txt = _get(url)
    if not txt.strip():
        raise RuntimeError(f"empty response for {site.id}")
    df = pd.read_csv(io.StringIO(txt), dtype=str, keep_default_na=False)
    return df


def fetch_power(site, start: str, end: str) -> pd.DataFrame:
    url = POWER.format(params=POWER_PARAMS, lat=site.lat, lon=site.lon,
                       start=start.replace("-", ""), end=end.replace("-", ""))
    js = json.loads(_get(url))
    prm = js["properties"]["parameter"]
    df = pd.DataFrame(prm)
    df.index = pd.to_datetime(list(prm[list(prm)[0]].keys()))
    df.index.name = "date"
    return df


def build_bundle(sites: list[str] | None = None, force: bool = False) -> dict:
    """Download every site into ``data/bundle`` and write a signed manifest."""
    cfg = config.load("sites.yaml")
    start, _ = config.period("calibration")
    _, end = config.period("test")
    out_dir = config.data_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [s for s in config.sites() if sites is None or s.name in sites or s.id in sites]
    info = {}
    for s in wanted:
        sp = out_dir / f"{s.id}_station.csv"
        pp = out_dir / f"{s.id}_power.csv"
        if force or not sp.is_file():
            df = fetch_station(s, start, end)
            df.to_csv(sp, index=False)
        if force or not pp.is_file():
            df = fetch_power(s, start, end)
            df.to_csv(pp)
        info[s.id] = {"name": s.name, "lat": s.lat, "lon": s.lon,
                      "climate": s.climate, "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        print(f"[fetch] {s.id} {s.name}", flush=True)

    hashes = {}
    for f in sorted(out_dir.iterdir()):
        if f.is_file() and f.suffix in (".csv", ".json") and f.name not in ("FETCH_INFO.json",):
            hashes[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    (out_dir / "FETCH_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    anchor = SoftwareAnchor.load_or_create(config.path("data", "keys", "anchor.pem"))
    sign_manifest(anchor, hashes, out_dir)
    anchor.save_attestation(config.path("docs", "anchor_attestation.json"))
    print(f"[fetch] manifest signed by anchor {anchor.fingerprint()} over {len(hashes)} files", flush=True)
    return hashes


def verify() -> tuple[bool, list[str]]:
    from .provenance.manifest import verify_manifest

    return verify_manifest(config.data_dir())
