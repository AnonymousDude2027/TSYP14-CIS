"""Ablations (specification section 12).

Each ablation removes exactly one component and re-runs the *same* calibrated
false-alarm procedure and the *same* injections.  The comparison is reported on
three axes because they tell different stories:

* detection rate at a fixed false-alarm budget -- does the component find more?
* detection delay -- does it find it sooner?  (Whitening's main benefit shows up
  here, not in the false-alarm rate: the block bootstrap already absorbs
  autocorrelation when it sets the threshold, so the naive claim "without
  whitening the false-alarm rate explodes" is *false* once thresholds are
  bootstrapped.  What collapses without whitening is power, because the CUSUM is
  fighting correlated noise.  Reporting only the false-alarm rate would hide
  this and the report says so explicitly.)
* false alarms per sensor-year -- does it stay honest?
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from ..attacks.families import Injector
from ..detect import statistics as st
from ..detect.engine import build_model
from ..io import load_pair
from .metrics import detection_summary

VARIANTS = {
    "full": {},
    "no_whitening": {"use_whitening": False},
    "no_reference": {"use_reference": False},
    "no_fingerprint": {"use_fingerprint": False},
    "gbm_expectation": {"expectation_kind": "gbm"},
}


def run_variant(name: str, sites: list[str], variables: tuple[str, ...] = ("TMAX",),
                magnitudes: tuple[float, ...] = (1.5, 3.0), n_per_cell: int = 5,
                family: str = "step") -> pd.DataFrame:
    kwargs = VARIANTS[name]
    rows = []
    for site in sites:
        for var in variables:
            b = load_pair(site, var)
            t0 = time.perf_counter()
            m = build_model(site, var, rng=np.random.default_rng(12345), **kwargs)
            build_s = time.perf_counter() - t0
            n = len(b.sensor)
            test_start = int(np.flatnonzero(b.dates >= np.datetime64("2020-01-01"))[0])
            inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, n, test_start=test_start)
            for mag in magnitudes:
                delays = []
                for rep in range(n_per_cell):
                    rng = np.random.default_rng(4242 + 31 * rep)
                    spec = inj.make_spec(family, rng, magnitude_sigma=mag, magnitude_percent=None)
                    res = inj.inject(b.sensor, spec)
                    import copy

                    from ..sp.conditioning import to_working

                    mm = copy.copy(m)
                    cc = copy.copy(m.conditioned)
                    cc.y = to_working(np.asarray(res.values, dtype=float), cc.kind)
                    cc.valid = np.isfinite(cc.y) & np.isfinite(cc.x)
                    pred = m.expectation.predict(cc)
                    r = cc.y - pred
                    r[~cc.valid] = np.nan
                    _, z = m.whitener.transform(cc, r)
                    mm.conditioned, mm.z, mm.residual = cc, z, r
                    al = mm.alarms()
                    mask = st.event_onsets_mask(al["exceed_any"], 14)
                    ev = np.flatnonzero(mask)
                    hits = ev[(ev >= spec.onset) & (ev <= spec.onset + spec.duration + 30)]
                    delays.append(int(hits[0] - spec.onset) if len(hits) else None)
                s = detection_summary(delays, n_per_cell, 10 ** 9)
                # false alarms on the clean test period for this variant
                al = m.alarms()
                mask = st.event_onsets_mask(al["exceed_any"], 14)
                testmask = ~m._cal  # noqa: SLF001
                years = max(testmask.sum() / 365.25, 1e-9)
                rows.append({
                    "variant": name, "site": site, "variable": var, "family": family,
                    "magnitude_sigma": mag, "build_seconds": build_s,
                    "sigma_residual": m.sigma_residual,
                    "fa_per_year": float(mask[testmask].sum() / years),
                    **{k: v for k, v in s.items()},
                })
    return pd.DataFrame(rows)
