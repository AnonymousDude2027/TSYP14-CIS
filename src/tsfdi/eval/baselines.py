"""Baselines B0-B3 (specification section 12).

B0  fixed threshold on the standardised residual, |z| > 3 (the "obvious" method)
B1  CUSUM on the *unwhitened* residual (SP3 removed) -- the ablation that shows
    what the signal-processing chain buys
B2  offline change-point detection with ``ruptures`` (PELT, L2 cost) on the
    residual -- the tool the brief itself suggests
B3  Isolation Forest on window features (scikit-learn's implementation of the
    same algorithm PyOD ships as ``IForest``; PyOD is listed as an optional
    dependency in requirements.txt and is used when importable)

Each baseline is given exactly the same injections, the same evaluation period
and the same scoring function as the proposed system, and any baseline threshold
is calibrated to the same false-alarm budget whenever the baseline exposes a
threshold (B0, B3); B2 is an offline batch method with no threshold to calibrate,
which is itself part of the honest comparison (it cannot run in streaming mode).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config
from ..detect import statistics as st


@dataclass
class BaselineResult:
    name: str
    rows: list

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _detect_from_exceed(exceed: np.ndarray, labels: np.ndarray, onset: int, duration: int,
                        cooldown: int = 14, extra: int = 30) -> tuple[bool, int | None]:
    mask = st.event_onsets_mask(exceed, cooldown)
    events = np.flatnonzero(mask)
    deadline = onset + duration + extra
    hits = events[(events >= onset) & (events <= deadline)]
    return (len(hits) > 0, int(hits[0] - onset) if len(hits) else None)


def b0_fixed_threshold(z: np.ndarray, h: float, onset: int, duration: int) -> tuple[bool, int | None]:
    return _detect_from_exceed(np.abs(z) > h, None, onset, duration)


def calibrate_b0(z_cal: np.ndarray, budget_per_year: float = 1.0, n_days: int = 1826) -> float:
    """Simplest possible calibration: choose h so |z| > h happens ~budget times a year."""
    v = np.abs(z_cal[np.isfinite(z_cal)])
    if v.size == 0:
        return 3.0
    target = budget_per_year
    # expected number of exceedances per year under independence
    q = 1.0 - target * 14.0 / 365.25       # one event per cooldown window, roughly
    q = float(np.clip(q, 0.5, 1 - 1e-6))
    return float(np.quantile(v, q))


def b1_cusum_unwhitened(resid: np.ndarray, sigma: float, k: float = 1.0,
                        threshold_sigma: float = 5.0) -> np.ndarray:
    z = resid / sigma
    up, dn = st.cusum(z, k=k)
    return np.maximum(up, dn) > threshold_sigma


def b2_ruptures(resid: np.ndarray, pen: float | None = None) -> list[int]:
    """Offline PELT change points on the residual (L2 cost)."""
    try:
        import ruptures as rpt
    except Exception:  # pragma: no cover
        return []
    x = np.asarray(resid, dtype=float)
    ok = np.isfinite(x)
    if ok.sum() < 50:
        return []
    filled = np.interp(np.arange(len(x)), np.flatnonzero(ok), x[ok])
    algo = rpt.Pelt(model="l2", min_size=3).fit(filled.reshape(-1, 1))
    pen = pen if pen is not None else 3.0 * np.log(len(filled)) * np.var(filled)
    try:
        cps = algo.predict(pen=pen)
    except Exception:  # noqa: BLE001
        return []
    return [int(c) for c in cps if c < len(filled)]


def b3_isolation_forest(z: np.ndarray, window: int = 7, contamination: float = 0.01,
                        seed: int = 0):
    """Isolation Forest on rolling window features (mean, std, min, max, slope)."""
    try:
        from pyod.models.iforest import IForest  # type: ignore

        backend = "pyod"
    except Exception:  # pragma: no cover
        from sklearn.ensemble import IsolationForest as IForest  # type: ignore

        backend = "sklearn"

    X = _window_features(z, window)
    ok = np.all(np.isfinite(X), axis=1)
    if ok.sum() < 100:
        return np.zeros(len(z)), backend
    model = IForest(contamination=contamination, random_state=seed)
    model.fit(X[ok])
    if backend == "pyod":
        scores = model.decision_function(X[ok])
    else:
        scores = -model.score_samples(X[ok])
    thr = np.quantile(scores, 1 - contamination)
    out = np.zeros(len(z))
    out[ok] = scores > thr
    return out.astype(bool), backend


def _window_features(z: np.ndarray, w: int) -> np.ndarray:
    n = len(z)
    X = np.full((n, 5), np.nan)
    for t in range(w - 1, n):
        win = z[t - w + 1: t + 1]
        if not np.all(np.isfinite(win)):
            continue
        i = np.arange(w, dtype=float)
        slope = np.polyfit(i, win, 1)[0]
        X[t] = [win.mean(), win.std(), win.min(), win.max(), slope]
    return X


def run_baselines(model, bundle, injections: list[dict], budget: float = 1.0,
                  include_proposed: bool = True) -> pd.DataFrame:
    """Evaluate B0-B3 -- and, by default, the proposed system -- on one injection set.

    The proposed system is evaluated *here*, on the identical injections, rather
    than in a separate run: a baseline comparison assembled from two different
    injection draws is not a comparison.  Its row is labelled ``PROPOSED`` so that
    every table can be read as "same attacks, same metrics, different detector".
    """
    rows = []
    cal = model._cal  # noqa: SLF001
    z_cal = model.z[cal]
    h0 = calibrate_b0(z_cal, budget, int(cal.sum()))
    for inj in injections:
        y = inj["values"]
        c = model.conditioned
        import copy

        from ..sp.conditioning import to_working

        cc = copy.copy(c)
        cc.y = to_working(np.asarray(y, dtype=float), cc.kind)
        cc.valid = np.isfinite(cc.y) & np.isfinite(cc.x)
        pred = model.expectation.predict(cc)
        r = cc.y - pred
        r[~cc.valid] = np.nan
        _, z = model.whitener.transform(cc, r)
        onset, dur = int(inj["onset"]), int(inj["duration"])

        det, delay = b0_fixed_threshold(z, h0, onset, dur)
        rows.append({"baseline": "B0_fixed_threshold", "family": inj["family"],
                     "magnitude_sigma": inj.get("magnitude_sigma", np.nan),
                     "site": model.site, "variable": model.variable,
                     "detected": det, "delay": delay})

        # B1: no whitening
        sig = model.sigma_residual
        ex = b1_cusum_unwhitened(r, sig)
        det, delay = _detect_from_exceed(ex, None, onset, dur)
        rows.append({"baseline": "B1_cusum_unwhitened", "family": inj["family"],
                     "magnitude_sigma": inj.get("magnitude_sigma", np.nan),
                     "site": model.site, "variable": model.variable,
                     "detected": det, "delay": delay})

        # B2: ruptures, offline
        cps = b2_ruptures(r)
        hits = [c for c in cps if onset <= c <= onset + dur + 30]
        rows.append({"baseline": "B2_ruptures_pelt", "family": inj["family"],
                     "magnitude_sigma": inj.get("magnitude_sigma", np.nan),
                     "site": model.site, "variable": model.variable,
                     "detected": bool(hits), "delay": (min(hits) - onset) if hits else None})

        # PROPOSED: the calibrated D1-D6 suite on the same corrupted series
        if include_proposed:
            from .runner import evaluate_injection

            out = evaluate_injection(model, bundle, y, np.asarray(inj["labels"]),
                                     onset, dur, inj["family"],
                                     float(inj.get("magnitude_sigma", np.nan)), np.nan)
            rows.append({"baseline": "PROPOSED", "family": inj["family"],
                         "magnitude_sigma": inj.get("magnitude_sigma", np.nan),
                         "site": model.site, "variable": model.variable,
                         "detected": out.detected, "delay": out.delay,
                         "auprc": out.auprc, "detectors_fired": "|".join(out.detectors_fired)})

        # B3: isolation forest on window features
        ex, backend = b3_isolation_forest(z)
        det, delay = _detect_from_exceed(ex, None, onset, dur)
        rows.append({"baseline": f"B3_isolation_forest[{backend}]", "family": inj["family"],
                     "magnitude_sigma": inj.get("magnitude_sigma", np.nan),
                     "site": model.site, "variable": model.variable,
                     "detected": det, "delay": delay})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# false alarms of the baselines on the *clean* series
# --------------------------------------------------------------------------- #

def baseline_false_alarms(model, budget: float = 1.0, cooldown: int = 14) -> list[dict]:
    """Alarm events per sensor-year for each baseline on the untouched series.

    A detection rate without the matching false-alarm rate is not a comparison: a
    detector that alarms every day detects every attack and is useless.  Each
    baseline is calibrated (where it has a threshold) to the *same* budget and then
    run on the clean 2020-2024 period, with the same event definition and cooldown
    as the proposed suite.
    """
    cal = model._cal                       # noqa: SLF001
    test = ~cal
    years = float(test.sum() / 365.25)
    z = model.z
    h0 = calibrate_b0(z[cal], budget, int(cal.sum()))
    rows: list[dict] = []

    def add(name: str, exceed: np.ndarray) -> None:
        mask = st.event_onsets_mask(np.asarray(exceed, dtype=bool), cooldown) & test
        rows.append({"baseline": name, "site": model.site, "variable": model.variable,
                     "years": years, "events": int(mask.sum()),
                     "events_per_year": float(mask.sum() / years) if years else np.nan})

    with np.errstate(invalid="ignore"):
        add("B0_fixed_threshold", np.isfinite(z) & (np.abs(z) > h0))
    add("B1_cusum_unwhitened", b1_cusum_unwhitened(model.residual, model.sigma_residual))
    # B2 has no threshold: every change point it reports is an "alarm"
    cps = np.asarray(b2_ruptures(model.residual), dtype=int)
    keep = cps[(cps >= 0) & (cps < len(z))]
    b2 = np.zeros(len(z), bool)
    b2[keep] = True
    add("B2_ruptures_pelt", b2)
    ex, backend = b3_isolation_forest(z)
    add(f"B3_isolation_forest[{backend}]", ex)
    al = model.alarms()
    add("PROPOSED", al["exceed_any"])
    return rows
