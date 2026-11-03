"""Adaptive adversaries A1, A2 and the compromised signer A3 (spec section 9).

A0  the standard profile: fixed magnitude, no knowledge of the detector.
A1  reference-aware.  The attacker knows that the operator compares the sensor
    with a public reanalysis, so it replaces the reading with the *expected*
    value: A1a is the pure expectation, A1b adds white noise at the matched
    variance, A1c adds AR-matched noise (the attacker has reverse engineered the
    fingerprint and reproduces its autocorrelation).  The bias an A1 attacker
    achieves is bounded by the residual scale, but it is invisible to any level
    test -- D6 exists for this family.
A2  detector-aware.  The white-box variant runs the whole detector suite
    incrementally and picks, every day, the largest bias increment that keeps
    every statistic below (1 - margin) x its threshold.  The black-box variant
    only observes alarms and bisects a constant bias rate.
A3  compromised signer: values are false but every signature is valid, so L0
    passes and only L1-L5 can help (tested in tests/test_provenance.py and
    measured in the evasion study).

The greedy A2 is the metric the specification calls Urbina-style: what damage can
a detector-aware attacker do *at a fixed false-alarm budget*.  Reporting detection
rate against A0 alone would overstate the system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..detect import statistics as st


@dataclass
class AdaptiveResult:
    values: np.ndarray
    delta: np.ndarray
    labels: np.ndarray
    cumulative_bias: float
    detected: bool
    detection_day: int | None
    level: str
    meta: dict


def _z_of(model, y_modified: np.ndarray) -> np.ndarray:
    """Recompute z for a modified sensor road with the *fitted* SP2/SP3 parameters."""
    import copy

    from ..sp.conditioning import to_working

    c = copy.copy(model.conditioned)
    c.y = to_working(np.array(y_modified, dtype=float), c.kind)
    c.valid = np.isfinite(c.y) & np.isfinite(c.x)
    pred = model.expectation.predict(c)
    r = c.y - pred
    r[~c.valid] = np.nan
    _, z = model.whitener.transform(c, r)
    return z


# --------------------------------------------------------------------------- #
# A1: reference-aware substitution
# --------------------------------------------------------------------------- #


def a1_reference_aware(model, noise: str = "none", rng: np.random.Generator | None = None,
                       start: int | None = None, duration: int = 180) -> AdaptiveResult:
    """Replace the sensor by the SP2 expectation (+ matched noise)."""
    rng = np.random.default_rng(config.SEED) if rng is None else rng
    c = model.conditioned
    n = len(c.y)
    start = n // 2 if start is None else int(start)
    stop = min(n, start + duration)
    pred = model.expectation.predict(c)
    out = np.array(c.y, dtype=float)
    delta = np.zeros(n)
    new = pred[start:stop].copy()
    if noise == "white":
        sig = np.nanstd(model.residual[np.isfinite(model.residual)])
        new = new + rng.normal(0, sig, len(new))
    elif noise == "ar1":
        phi = float(np.atleast_1d(model.whitener.phi)[0]) if model.whitener.p else 0.0
        sig_e = float(model.whitener.sigma_e)
        e = np.zeros(len(new))
        for i in range(len(new)):
            e[i] = phi * (e[i - 1] if i else 0.0) + rng.normal(0, sig_e)
        new = new + e
    delta[start:stop] = new - c.y[start:stop]
    out[start:stop] = new
    labels = np.zeros(n, bool)
    labels[start:stop] = True
    # run the calibrated suite on the result
    z = _z_of(model, out)
    exceed, _ = _exceed_any(model, z)
    mask = st.event_onsets_mask(exceed, 14)
    hits = np.flatnonzero(mask[start:]) + start
    return AdaptiveResult(out, delta, labels, float(np.nansum(np.abs(delta))),
                          bool(len(hits)), int(hits[0]) if len(hits) else None,
                          f"A1{'' if noise == 'none' else noise}",
                          {"noise": noise})


def _exceed_any(model, z: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run every calibrated detector on a given z road."""
    import copy

    m = copy.copy(model)
    m.z = z
    al = m.alarms()
    return al["exceed_any"], al["exceed"]


# --------------------------------------------------------------------------- #
# A2: detector-aware greedy (white box)
# --------------------------------------------------------------------------- #


class _OnlineState:
    """Thin wrapper over :class:`detect.online.OnlineSuite` (used by A2).

    ``feasible`` asks the question the white-box attacker asks: would this
    candidate value push any statistic above (1 - margin) x its threshold?
    """

    def __init__(self, model, thresholds: dict, safety: float):
        from ..detect.online import OnlineSuite

        self.model = model
        self.h = thresholds
        self.safety = safety
        self.suite = OnlineSuite.from_config()

    def clone(self):
        s = _OnlineState.__new__(_OnlineState)
        s.model, s.h, s.safety = self.model, self.h, self.safety
        s.suite = self.suite.clone()
        return s

    def commit(self, z: float) -> None:
        self.suite.step(z)

    def stats_with(self, z_new: float) -> dict[str, float]:
        trial = self.suite.clone()
        return trial.step(z_new)

    def feasible(self, z_new: float) -> bool:
        s = self.stats_with(z_new)
        for k, v in s.items():
            h = self.h.get(k, np.inf)
            if not np.isfinite(h) or not np.isfinite(v):
                continue
            if v > (1.0 - self.safety) * h:
                return False
        return True


def a2_greedy(model, rng: np.random.Generator | None = None, start: int | None = None,
              duration: int = 240, safety: float | None = None,
              increments: list[float] | None = None) -> AdaptiveResult:
    """White-box greedy: largest daily bias increment that keeps every statistic quiet."""
    rng = np.random.default_rng(config.SEED) if rng is None else rng
    acfg = config.load("attacks.yaml")["adversary_levels"]["A2"]
    safety = float(acfg["safety_margin"]) if safety is None else float(safety)
    grid = np.array(acfg["increment_grid_sigma"] if increments is None else increments, dtype=float)
    grid = np.sort(grid)

    c = model.conditioned
    n = len(c.y)
    start = n // 2 if start is None else int(start)
    stop = min(n, start + duration)
    sigma = float(model.sigma_residual)

    y = np.array(c.y, dtype=float)
    delta = np.zeros(n)
    state = _OnlineState(model, model.thresholds, safety)
    # warm-up: replay the clean history through the state
    z_clean = model.z
    for i in range(max(0, start)):
        state.commit(z_clean[i])

    sign = int(rng.choice([-1, 1]))
    for t in range(start, stop):
        if not np.isfinite(c.y[t]):
            state.commit(z_clean[t])
            continue
        best = 0.0
        for g in grid[::-1]:                       # largest first
            cand = c.y[t] + sign * g * sigma
            y[t] = cand
            z_try = _z_single(model, c, y, t)
            if np.isfinite(z_try) and state.feasible(z_try):
                best = g
                break
            y[t] = c.y[t]
        if best > 0:
            y[t] = c.y[t] + sign * best * sigma
            delta[t] = sign * best * sigma
        z_now = _z_single(model, c, y, t)
        state.commit(z_now)

    labels = np.zeros(n, bool)
    labels[start:stop] = np.abs(delta[start:stop]) > 0
    z = _z_of(model, y)
    exceed, _ = _exceed_any(model, z)
    mask = st.event_onsets_mask(exceed, 14) & ~_clean_onset_mask(model)
    hits = np.flatnonzero(mask[start: stop + 30]) + start   # attributable alarms only
    return AdaptiveResult(y, delta, labels, float(np.nansum(np.abs(delta))),
                          bool(len(hits)), int(hits[0]) if len(hits) else None,
                          "A2", {"safety": safety, "sign": sign,
                                 "mean_increment_sigma": float(np.mean(np.abs(delta[start:stop])) / sigma)})


def _z_single(model, c, y_mod: np.ndarray, t: int) -> float:
    """z_t for a single modified value (cheap: only formulas for day t are needed)."""
    x = c.x
    if not np.isfinite(x[t]):
        return np.nan
    coef = model.expectation.coef
    if coef is None:
        return np.nan
    # rebuild the design row for day t exactly as SP2 does
    from ..sp.expectation import harmonic_design

    h = harmonic_design(c.dates[t: t + 1], model.expectation.n_harm, n_days=np.array([c.t[t]]))[0]
    pred = coef[0] + (coef[1] * x[t] if model.expectation.use_reference else 0.0) + h @ coef[2 if model.expectation.use_reference else 1:]
    r_t = y_mod[t] - pred
    # innovations need the AR history of residuals; recompute locally from y_mod
    p = model.whitener.p
    if p > 0:
        hist = []
        for k in range(1, p + 1):
            i = t - k
            if i < 0 or not np.isfinite(y_mod[i]) or not np.isfinite(x[i]):
                return np.nan
            hh = harmonic_design(c.dates[i: i + 1], model.expectation.n_harm, n_days=np.array([c.t[i]]))[0]
            pi = coef[0] + (coef[1] * x[i] if model.expectation.use_reference else 0.0) + hh @ coef[2 if model.expectation.use_reference else 1:]
            hist.append(y_mod[i] - pi)
        e = r_t - float(np.atleast_1d(model.whitener.phi) @ np.asarray(hist))
    else:
        e = r_t
    s = model.whitener.scale(c)[t]
    return float(e / s) if s and np.isfinite(s) else np.nan


def _clean_onset_mask(model) -> np.ndarray:
    """Alarm events of the *unmodified* series (used to subtract pre-existing alarms).

    Without this, an adaptive search credits a 0.02-sigma injection with the false
    alarm that would have fired anyway: the black-box bisection then collapses to
    zero on any site whose 240-day window happens to contain a natural event, and
    the resulting "the adversary can achieve nothing" conclusion is an artefact.
    """
    al = model.alarms()
    return st.event_onsets_mask(al["exceed_any"], 14)


# --------------------------------------------------------------------------- #
# A2 black box
# --------------------------------------------------------------------------- #


def a2_blackbox(model, rng: np.random.Generator | None = None, start: int | None = None,
                duration: int = 240, iters: int = 16) -> AdaptiveResult:
    """Only alarms are observed: bisect a constant bias rate until an alarm fires."""
    rng = np.random.default_rng(config.SEED) if rng is None else rng
    c = model.conditioned
    n = len(c.y)
    start = n // 2 if start is None else int(start)
    stop = min(n, start + duration)
    sign = int(rng.choice([-1, 1]))
    sigma = float(model.sigma_residual)

    clean = _clean_onset_mask(model)
    lo, hi = 0.0, 6.0
    best_y, best_delta, best_hits = None, None, None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        y = np.array(c.y, dtype=float)
        y[start:stop] = y[start:stop] + sign * mid * sigma
        z = _z_of(model, y)
        exceed, _ = _exceed_any(model, z)
        mask = st.event_onsets_mask(exceed, 14) & ~clean
        # Only an alarm *attributable to the injection* counts: an event that
        # starts inside the attack window or within the 30-day detection deadline.
        # (The first version scanned everything after ``start``, so an unrelated
        # false alarm months later always counted as a detection and the
        # bisection collapsed to zero damage on every site.)
        hits = np.flatnonzero(mask[start: stop + 30]) + start
        if len(hits) == 0:
            best_y, best_delta, best_hits = y, y - c.y, hits
            lo = mid
        else:
            hi = mid
        if hi - lo < 0.01:          # 0.01 sigma resolution (was 0.05: too coarse to
            break                   # distinguish "no margin" from "0.05 sigma of margin")
    if best_y is None:                     # even the smallest step alarms: no damage
        best_y = np.array(c.y, dtype=float)
        best_delta = np.zeros(n)
        best_hits = np.array([], int)
    labels = np.zeros(n, bool)
    labels[start:stop] = True
    return AdaptiveResult(best_y, best_delta, labels, float(np.nansum(np.abs(best_delta))),
                          bool(len(best_hits)), int(best_hits[0]) if len(best_hits) else None,
                          "A2-blackbox", {"achieved_sigma": float(abs(lo))})
