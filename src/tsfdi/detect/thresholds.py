"""False-alarm-calibrated thresholds by block bootstrap.

Motivation (spec section 8).  A threshold chosen to catch a 12 % step with "the
best accuracy" is not a security control: it silently trades false alarms
against detection and the trade is invisible to the reader.  Here every
threshold is chosen *only* from a false-alarm budget, on the calibration period,
before the detector ever sees test data.  Detection performance is then a
*consequence* that is measured, not a target that is fitted.

Procedure.

1. The calibration series of a (site, variable) pair is resampled in contiguous
   blocks of B = 30 days (a circular block bootstrap).  Blocks preserve the
   short-range dependence that SP3 has not removed, so the synthetic years look
   like the real ones; 30 days covers the synoptic timescale that dominates
   weather anomalies.
2. For each replicate the detector statistic is recomputed end to end.  Note
   that the statistic is recomputed, not merely permuted: the CUSUM state, the
   rolling windows and the matched filter all see a different trajectory, which
   is exactly the source of variability that matters.
3. For a candidate threshold h the number of alarm *events* (onsets separated by
   the cooldown, see ``statistics.event_onsets``) is counted per replicate and
   converted to events per sensor-year.
4. The threshold is the smallest h whose rate does not exceed the budget
   (monotone in h, so a bisection converges).

Budget allocation across the suite is Bonferroni by default: with K statistics
in the suite each gets budget/K, so the *system-level* false-alarm rate honours
the requested budget even though the statistics are correlated (Bonferroni is
conservative when they are, which is the safe direction).

Cost.  One calibration per (site, variable, detector).  Measured cost is in
``results/tables/table_efficiency.csv``; the check ``make verify`` asserts that
the thresholds are reproducible from the committed seed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from .statistics import event_onsets


def block_bootstrap_indices(n: int, n_rep: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Circular block bootstrap: (n_rep, n) matrix of indices into a length-n series."""
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_rep, n_blocks))
    offs = np.arange(block)[None, None, :]
    idx = (starts[:, :, None] + offs) % n
    return idx.reshape(n_rep, n_blocks * block)[:, :n]


def _events_rate(exceed: np.ndarray, cooldown: int, n_days: int) -> float:
    """Mean alarm events per sensor-year for a (T, R) boolean exceedance array."""
    counts = event_onsets(exceed, cooldown)
    counts = np.atleast_1d(counts)
    return float(counts.sum() * 365.25 / max(n_days * len(counts), 1))


@dataclass
class ThresholdResult:
    detector: str
    budget: float
    threshold: float
    achieved_rate: float
    all_thresholds: dict[float, tuple[float, float]] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "detector": self.detector,
            "budget_per_year": self.budget,
            "threshold": self.threshold,
            "achieved_rate_per_year": self.achieved_rate,
        }


def calibrate_threshold(
    stat_replicates: np.ndarray,
    budget: float,
    n_days: int,
    cooldown: int,
    bisection_steps: int = 14,
    n_events_floor: float = 0.0,
) -> tuple[float, float]:
    """Smallest threshold whose bootstrap false-alarm rate is <= ``budget``.

    ``stat_replicates`` is (T, R).  NaN entries are never alarms.
    """
    S = np.asarray(stat_replicates, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    finite = S[np.isfinite(S)]
    if finite.size == 0:
        return float("inf"), 0.0
    hi = float(np.max(finite)) * 1.0001 + 1e-9
    # The search is restricted to thresholds above the *median* of the null
    # statistic.  Below the median the detector is in alarm on more than half of
    # all days, which is not an operating point, and the event counter loses
    # monotonicity there: with the Schmitt-trigger rule of
    # statistics.event_onsets a permanently-alarmed detector counts as ONE event,
    # so a vanishing threshold would look *better* than a sensible one and the
    # bisection could converge to it.  (This is exactly the degenerate case that
    # tests/test_detectors.py::test_calibrated_threshold_meets_the_budget_on_synthetic_null
    # pins down.)
    lo = float(np.median(finite))
    if not np.isfinite(lo) or lo >= hi:
        lo = float(np.min(finite)) * 0.9999 - 1e-9
    S = np.where(np.isfinite(S), S, -np.inf)

    def rate(h: float) -> float:
        return _events_rate(S > h, cooldown, n_days)

    if rate(lo) <= budget:
        return lo, rate(lo)
    if rate(hi) > budget:
        # Even the highest value observed in the bootstrap violates the budget.
        # This happens when the null statistic is so clustered that a single
        # excursion spans the whole replicate; the caller records the achieved
        # rate so that the failure is visible in the tables instead of being
        # silently rounded away.
        return hi, rate(hi)
    for _ in range(bisection_steps):
        mid = 0.5 * (lo + hi)
        if rate(mid) <= budget:
            hi = mid
        else:
            lo = mid
    return float(hi), float(rate(hi))


def calibrate_suite(
    stat_builder,
    detector_names: list[str],
    n_cal: int,
    budgets: list[float],
    rng: np.random.Generator,
    n_boot: dict[str, int] | None = None,
    split: str = "bonferroni",
) -> dict[str, dict]:
    """Calibrate every detector of a suite on the calibration period.

    ``stat_builder(idx)`` must return a dict ``name -> (T, R)`` array for the
    resampled series described by the index matrix ``idx`` (shape (R, n_cal)).
    """
    cfg = config.load("detectors.yaml")["calibration"]
    block = int(cfg["block_length_days"])
    cooldown = int(cfg["cooldown_days"])
    n_boot = n_boot or {}
    k = len(detector_names) if split == "bonferroni" else 1
    out: dict[str, dict] = {}
    for name in detector_names:
        R = int(n_boot.get(name, cfg["n_bootstrap"]))
        idx = block_bootstrap_indices(n_cal, R, block, rng)
        stats = stat_builder(idx, name)                    # (n_cal, R)
        res = {}
        for budget in budgets:
            per_det = budget / k
            h, achieved = calibrate_threshold(stats, per_det, n_cal, cooldown,
                                              bisection_steps=int(cfg["bisection_steps"]))
            res[float(budget)] = (h, achieved, per_det)
        out[name] = res
    return out


def pick_threshold(calib: dict, detector: str, budget: float) -> float:
    """Threshold for ``detector`` at the operating budget."""
    return float(calib[detector][float(budget)][0])
