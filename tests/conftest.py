"""Fixtures: a synthetic (site, variable) bundle with known ground truth.

The tests never touch the network and never depend on the real bundle: they build
a synthetic series whose generating process is known exactly, wrap it in the same
``SeriesBundle`` the real loader returns, and push it through the real code path.
That is the "validate every module on synthetic data with known ground truth
before using it on real data" rule from section 17 of the specification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsfdi.io import SeriesBundle

N = 3653
START = "2015-01-01"


def synth_series(seed: int = 7, n: int = N, phi: float = 0.6, sigma0: float = 1.6,
                 amp: float = 12.0, beta: float = 0.9, gaps: float = 0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    x = 12 * np.sin(2 * np.pi * t / 365.2422 - 1.0) + rng.normal(0, 2.5, n)
    season = 1.0 + 0.35 * np.sin(2 * np.pi * t / 365.2422)
    e = np.zeros(n)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + rng.normal(0, sigma0 * season[i])
    y = 8.0 + beta * x + amp * np.sin(2 * np.pi * t / 365.2422) + 3 * np.sin(4 * np.pi * t / 365.2422) + e
    if gaps:
        mask = rng.random(n) < gaps
        y = np.where(mask, np.nan, y)
    return y, x


def make_bundle(y: np.ndarray, x: np.ndarray, variable: str = "TMAX",
                kind: str = "temperature") -> SeriesBundle:
    n = len(y)
    dates = pd.date_range(START, periods=n, freq="D")
    return SeriesBundle(
        site_id="SYN00000001", site_name="SYNTH", variable=variable,
        reference_variable="T2M_MAX", units="degC" if kind == "temperature" else "mm",
        kind=kind, dates=dates, sensor=np.asarray(y, dtype=float),
        reference=np.asarray(x, dtype=float), sensor_ok=np.isfinite(y),
        reference_ok=np.isfinite(x),
    )


@pytest.fixture(scope="session")
def synth_bundle() -> SeriesBundle:
    y, x = synth_series()
    return make_bundle(y, x)


@pytest.fixture(scope="session")
def synth_model(synth_bundle):
    """A calibrated model on the synthetic bundle (built once per session)."""
    from tsfdi.detect.engine import build_model

    return build_model("SYNTH", "TMAX", budget=1.0, rng=np.random.default_rng(11),
                       bundle=synth_bundle)
