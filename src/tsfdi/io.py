"""Data access: the offline bundle, quality flags, calendar alignment (SP1).

Nothing here touches the network.  ``scripts/fetch_data.py`` builds
``data/bundle``; this module only reads it and verifies the manifest.

Design rules enforced here (spec section 5):
  * values carrying a non-empty GHCN quality flag are treated as missing and are
    excluded from calibration;
  * missing values are *flagged*, never interpolated inside the detector;
  * the reference lag is estimated on the calibration period only.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# GHCN-Daily quality flags that indicate a failed check.  The third attribute
# field is the source flag ("S" = SOURCE, ...) and is not a quality failure.
_QC_OK = {"", " "}


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def manifest_path() -> Path:
    return config.data_dir() / "MANIFEST.sha256"


def manifest() -> dict[str, str]:
    mp = manifest_path()
    if not mp.is_file():
        return {}
    out = {}
    for line in mp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, name = line.split(None, 1)
        out[name.strip()] = digest
    return out


def verify_bundle(strict: bool = True) -> list[str]:
    """Check every bundled file against MANIFEST.sha256.

    Returns a list of problems (empty when the bundle is intact).  With
    ``strict=True`` any mismatch raises, because a poisoned reference defeats
    every comparison downstream (spec section 5.5).
    """
    man = manifest()
    problems: list[str] = []
    if not man:
        problems.append("MANIFEST.sha256 missing or empty")
    for name, digest in man.items():
        p = config.data_dir() / name
        if not p.is_file():
            problems.append(f"missing: {name}")
            continue
        got = sha256_file(p)
        if got != digest:
            problems.append(f"hash mismatch: {name} (expected {digest[:12]}..., got {got[:12]}...)")
    if strict and problems:
        raise BundleIntegrityError("; ".join(problems))
    return problems


class BundleIntegrityError(RuntimeError):
    """Raised when a bundled file does not match its pinned hash."""


# --------------------------------------------------------------------------- #
# Series loading
# --------------------------------------------------------------------------- #


@dataclass
class SeriesBundle:
    """One (site, variable) pair: sensor road, reference road, and their alignment."""

    site_id: str
    site_name: str
    variable: str
    reference_variable: str
    units: str
    kind: str
    dates: pd.DatetimeIndex
    sensor: np.ndarray            # float, NaN where missing or quality-flagged
    reference: np.ndarray         # float, NaN where POWER is missing (-999)
    sensor_ok: np.ndarray         # bool, True where the value is usable
    reference_ok: np.ndarray
    lag_days: int = 0
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.dates)

    def mask_between(self, start: str, end: str) -> np.ndarray:
        return np.asarray((self.dates >= pd.Timestamp(start)) & (self.dates <= pd.Timestamp(end)))

    def subset(self, mask: np.ndarray) -> "SeriesBundle":
        return SeriesBundle(
            site_id=self.site_id,
            site_name=self.site_name,
            variable=self.variable,
            reference_variable=self.reference_variable,
            units=self.units,
            kind=self.kind,
            dates=self.dates[mask],
            sensor=self.sensor[mask],
            reference=self.reference[mask],
            sensor_ok=self.sensor_ok[mask],
            reference_ok=self.reference_ok[mask],
            lag_days=self.lag_days,
            meta=dict(self.meta),
        )

    def with_sensor(self, values: np.ndarray, ok: np.ndarray | None = None) -> "SeriesBundle":
        """Return a copy whose sensor road is replaced (used by the injector)."""
        ok = (np.isfinite(values) & self.reference_ok) if ok is None else ok
        return SeriesBundle(
            site_id=self.site_id,
            site_name=self.site_name,
            variable=self.variable,
            reference_variable=self.reference_variable,
            units=self.units,
            kind=self.kind,
            dates=self.dates,
            sensor=np.asarray(values, dtype=float),
            reference=self.reference,
            sensor_ok=ok,
            reference_ok=self.reference_ok,
            lag_days=self.lag_days,
            meta=dict(self.meta),
        )


def _station_frame(site_id: str) -> pd.DataFrame:
    p = config.data_dir() / f"{site_id}_station.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"bundle file missing: {p.name}. Run `python -m tsfdi fetch` (needs network) "
            f"or restore the committed bundle."
        )
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"])
    return df.set_index("DATE").sort_index()


def _power_frame(site_id: str) -> pd.DataFrame:
    p = config.data_dir() / f"{site_id}_power.csv"
    if not p.is_file():
        raise FileNotFoundError(f"bundle file missing: {p.name}")
    df = pd.read_csv(p, index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df.replace(-999.0, np.nan).replace(-999, np.nan)


def _clean_sensor(df: pd.DataFrame, var: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (values, ok) with quality-flagged samples removed (spec 5.1)."""
    if var not in df.columns:
        n = len(df)
        return np.full(n, np.nan), np.zeros(n, dtype=bool)
    vals = pd.to_numeric(df[var].replace("", np.nan), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(vals)
    att_col = f"{var}_ATTRIBUTES"
    if att_col in df.columns:
        att = df[att_col].astype(str).fillna("")
        # GHCN attribute string: "measurement-flag,quality-flag,source-flag".
        # A non-empty quality flag means the value failed one of the GHCN-Daily
        # QC checks: treat it as missing (spec 5.1).
        qflag = att.str.split(",").str[1].fillna("").str.strip()
        ok &= qflag.eq("").to_numpy()
    return vals, ok


def load_pair(site_name_or_id: str, var: str, verify: bool = True) -> SeriesBundle:
    """Load a (sensor, reference) pair over the full configured period."""
    if verify:
        # Cheap integrity gate: the reference road is pinned by hash because a
        # poisoned reference defeats every comparison.
        verify_bundle(strict=True)
    s = config.site(site_name_or_id)
    vspec = config.variable_spec(var)
    start, _ = config.period("calibration")
    _, end = config.period("test")
    idx = pd.date_range(start, end, freq="D")

    st = _station_frame(s.id).reindex(idx)
    pw = _power_frame(s.id).reindex(idx)

    sensor, ok = _clean_sensor(st, vspec["sensor"])
    if vspec["reference"] in pw.columns:
        ref = pw[vspec["reference"]].to_numpy(dtype=float)
    else:
        ref = np.full(len(idx), np.nan)
    ref_ok = np.isfinite(ref)

    b = SeriesBundle(
        site_id=s.id,
        site_name=s.name,
        variable=var,
        reference_variable=vspec["reference"],
        units=vspec["units"],
        kind=vspec["kind"],
        dates=idx,
        sensor=sensor,
        reference=ref,
        sensor_ok=ok & np.isfinite(sensor),
        reference_ok=ref_ok,
        meta={"climate": s.climate, "elevation_m": s.elevation_m},
    )
    b.lag_days, b.meta["lag_corr"] = estimate_lag(b)
    b.reference = apply_lag(b.reference, b.lag_days)
    b.reference_ok = apply_lag(b.reference_ok.astype(float), b.lag_days) > 0.5
    return b


def apply_lag(x: np.ndarray, lag: int) -> np.ndarray:
    """Shift ``x`` by ``lag`` days (positive lag = reference leads the sensor)."""
    if lag == 0:
        return x.copy()
    out = np.full_like(x, np.nan, dtype=float)
    if lag > 0:
        out[lag:] = x[:-lag]
    else:
        out[:lag] = x[-lag:]
    return out


def estimate_lag(b: SeriesBundle, max_lag: int = 2) -> tuple[int, float]:
    """Calendar lag in {-max_lag..+max_lag} from the calibration cross-correlation.

    Stations and reanalysis products use slightly different day conventions
    (POWER is a daily mean over local solar time, GHCN observations are 24 h
    maxima with an unknown observation hour), so a one-day offset is expected.
    The lag maximising the correlation of the *anomalies* is used (spec 5.4).
    """
    cal = b.mask_between(*config.period("calibration"))
    s, r = b.sensor[cal], b.reference[cal]
    ok = np.isfinite(s) & np.isfinite(r)
    if ok.sum() < 100:
        return 0, float("nan")
    s = s[ok]
    r = r[ok]
    # remove the smooth annual cycle so the lag is not dominated by seasonality
    s = s - pd.Series(s).rolling(31, center=True, min_periods=10).mean().to_numpy()
    r = r - pd.Series(r).rolling(31, center=True, min_periods=10).mean().to_numpy()
    best, best_r = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        rr = np.roll(r, lag)
        if lag > 0:
            a, c = s[lag:], rr[lag:]
        elif lag < 0:
            a, c = s[:lag], rr[:lag]
        else:
            a, c = s, rr
        m = np.isfinite(a) & np.isfinite(c)
        if m.sum() < 100:
            continue
        val = float(np.corrcoef(a[m], c[m])[0, 1])
        if val > best_r:
            best, best_r = lag, val
    return int(best), float(best_r)


# --------------------------------------------------------------------------- #
# Provenance-side helper: the signed record stream
# --------------------------------------------------------------------------- #


def sensor_records(b: SeriesBundle, sensor_id: str | None = None):
    """Yield the canonical record dicts that L0 signs and verifies."""
    sid = sensor_id or b.site_id
    for i, (d, v) in enumerate(zip(b.dates, b.sensor)):
        if not np.isfinite(v):
            continue
        yield {
            "sensor_id": sid,
            "seq": i,
            "ts": d.strftime("%Y-%m-%dT00:00:00Z"),
            "variable": b.variable,
            "value": f"{v:.3f}",
        }


def library_versions() -> dict[str, str]:
    import platform

    import scipy
    import sklearn

    out = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "platform": platform.platform(),
    }
    try:
        import pywt

        out["pywavelets"] = pywt.__version__
    except Exception:  # pragma: no cover
        out["pywavelets"] = "absent"
    try:
        import ruptures

        out["ruptures"] = getattr(ruptures, "__version__", "1.1.10")
    except Exception:  # pragma: no cover
        out["ruptures"] = "absent"
    try:
        import cryptography

        out["cryptography"] = cryptography.__version__
    except Exception:  # pragma: no cover
        out["cryptography"] = "absent"
    return out


def dump_json(obj, name: str) -> Path:
    p = config.results_dir() / name
    p.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return p
