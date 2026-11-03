"""SP1 conditioning: units, calendar alignment, lag compensation, gap flags.

SP1 does *not* smooth, interpolate or impute.  A single sensor day is one
sample; a missing day is a hole with a flag, and every downstream stage must
cope with holes without inventing data (spec 5.4).  Interpolating inside the
detector would smear an injected anomaly across its neighbours and, worse, would
let a dropout masquerade as a valid observation.

What SP1 does do:

1. units are canonicalised (degC, mm) and physically impossible values are
   rejected as missing;
2. the reference is shifted by the lag estimated on the calibration period, so
   that day *t* of the sensor is compared with the reference *day* it actually
   describes;
3. a boolean ``valid`` mask is produced: sensor present, reference present, and
   the calendar day exists.  All later stages work on ``valid`` samples only;
4. a coarse robust trim (0.1 % symmetric) is recorded as a *calibration-only*
   diagnostic and is never applied at run time.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..io import SeriesBundle


def to_working(v: np.ndarray, kind: str) -> np.ndarray:
    """SP1 variance-stabilising map into the space where the residual is additive.

    Precipitation is the reason this exists.  Daily rain is zero-inflated and
    right-skewed: 70 % of days are zero, and a 12 mm day is 20 robust standard
    deviations away from the fitted expectation, so a Gaussian-innovation pipeline
    produces |z| ~ 20 on perfectly honest rainy days.  Measured on the real bundle,
    a raw PRCP residual gave 19 false alarms per sensor-year against a budget of 1.
    The square-root transform is the standard variance-stabilisation for
    non-negative, count-like accumulations (its variance is approximately constant
    for a Poisson-like process), and it is applied to both roads so the comparison
    stays like-for-like.  All reporting is still in millimetres: the transform is
    an internal coordinate, applied at SP1 and inverted for presentation.
    """
    v = np.asarray(v, dtype=float)
    if kind == "multiplicative":
        return np.sqrt(np.clip(v, 0.0, None))
    return v


@dataclass
class Conditioned:
    dates: np.ndarray          # datetime64[ns]
    y: np.ndarray              # sensor values (NaN where invalid)
    x: np.ndarray              # aligned reference values (NaN where unavailable)
    valid: np.ndarray          # bool
    t: np.ndarray              # day index, float (NaN where invalid)
    gap: np.ndarray            # bool, True where the sensor day is missing
    reference_gap: np.ndarray  # bool, True where the reference day is missing
    units: str
    variable: str
    kind: str
    site_id: str
    site_name: str
    trim_bounds: tuple[float, float] | None = None

    def __len__(self) -> int:
        return len(self.y)


class Conditioning:
    """SP1."""

    def __init__(self, variable: str | None = None):
        self.variable = variable

    def fit_trim(self, c: Conditioned, quantile: float = 0.001) -> Conditioned:
        """Record symmetric trim bounds from calibration data only."""
        q = config.load("detectors.yaml")["sp"]["trim_quantile"] if quantile is None else quantile
        y = c.y[c.valid]
        if y.size < 100:
            c.trim_bounds = None
            return c
        lo, hi = np.quantile(y, [q, 1 - q])
        c.trim_bounds = (float(lo), float(hi))
        return c

    def transform(self, b: SeriesBundle) -> Conditioned:
        y = np.asarray(b.sensor, dtype=float).copy()
        x = np.asarray(b.reference, dtype=float).copy()
        y = to_working(y, b.kind)
        x = to_working(x, b.kind)

        # --- physical sanity of the raw channel (a range guard, not a detector;
        # SP1 only removes values that cannot be a measurement at all).
        if b.kind == "multiplicative":
            y[y < 0] = np.nan                      # precipitation cannot be negative
            x[x < 0] = np.nan
        else:
            y[np.abs(y) > 70.0] = np.nan           # |T| > 70 degC is not an air temperature
            x[np.abs(x) > 70.0] = np.nan

        valid = np.isfinite(y) & np.isfinite(x)
        n = len(y)
        t = np.arange(n, dtype=float)
        t[~valid] = np.nan

        return Conditioned(
            dates=b.dates.to_numpy(),
            y=y,
            x=x,
            valid=valid,
            t=t,
            gap=~np.isfinite(y),
            reference_gap=~np.isfinite(x),
            units=b.units,
            variable=b.variable,
            kind=b.kind,
            site_id=b.site_id,
            site_name=b.site_name,
        )


def day_of_year(dates: np.ndarray) -> np.ndarray:
    """Fractional day index measured in tropical years since the epoch.

    ``phase = 2*pi*(t - t0)/365.2422`` is used for the harmonics instead of the
    calendar day-of-year so that leap years do not create a one-day phase jump.
    """
    d = dates.astype("datetime64[D]").astype(np.int64)
    return d - d[0]


def seasonal_phase(dates: np.ndarray, period_days: float = 365.2422) -> np.ndarray:
    d = dates.astype("datetime64[D]").astype(np.int64)
    return 2.0 * np.pi * (d % period_days) / period_days
