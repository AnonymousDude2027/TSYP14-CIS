"""Evaluation metrics, defined explicitly.

The brief asks for detection quality, false alarms, latency, memory and security
overhead; the specification forbids point-adjusted F1 and demands per-attack
reporting with medians, interquartile ranges and bootstrap confidence intervals.
This module implements exactly those, with the definitions written down, because
"detection rate" without a definition is not a number anyone can check.

Definitions used everywhere in this repository
----------------------------------------------
*An injection* is one corrupted series produced by one attack spec at one site
and variable.

*Detection* of an injection is the existence of an alarm **event** (first day of
an excursion above the calibrated threshold, onsets separated by the cooldown) at
a day ``t`` with ``onset <= t <= onset + deadline``, where ``deadline`` is the
attack duration plus a 30-day grace.  Alarms before the onset are counted as
false alarms, not as detections, even if the same injection eventually fires.

*Detection delay* is ``t_alarm - onset`` in days, reported as a median with the
interquartile range over detected injections, together with the never-detected
rate (a detector that is fast when it works and silent otherwise must not be
allowed to look good).

*False alarms per sensor-year* are counted on clean runs: events of the whole
suite divided by the number of years fed through it.  They are also reported
restricted to natural-extreme windows, which is where a naive residual threshold
fails.

*AUPRC* is computed over **sample days, not events, and without point
adjustment**: every day of every injection is one instance, labelled positive if
it is inside the attack window and negative otherwise, scored by the maximum
standardised statistic.  Point adjustment inflates every published number and is
prohibited here.

*Minimum detectable magnitude* is the smallest magnitude in the grid whose
detection rate reaches 0.5 (and 0.9) at the operating false-alarm budget.

*Damage at a fixed false-alarm budget* (the Urbina-style metric) is the
cumulative absolute bias an adaptive attacker inserts before the first alarm, in
units of sensor-days x sigma.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# basic statistics
# --------------------------------------------------------------------------- #


def bootstrap_ci(x, statistic=np.median, n_boot: int = 2000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI of a statistic."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(0) if rng is None else rng
    point = float(statistic(x))
    if x.size == 1:
        return point, point, point
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    vals = np.array([statistic(x[i]) for i in idx])
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def iqr(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))


def detection_summary(delays: list[int | None], n_injections: int, deadline: int) -> dict:
    """Detection rate, delay median/IQR and never-detected rate."""
    det = [d for d in delays if d is not None and d <= deadline]
    nev = [d for d in delays if d is None or d > deadline]
    arr = np.array(det, dtype=float) if det else np.array([], dtype=float)
    med, lo, hi = bootstrap_ci(arr) if arr.size else (float("nan"),) * 3
    q1, q3 = iqr(arr) if arr.size else (float("nan"), float("nan"))
    return {
        "n_injections": int(n_injections),
        "n_detected": int(len(det)),
        "detection_rate": float(len(det) / n_injections) if n_injections else float("nan"),
        "never_detected_rate": float(len(nev) / n_injections) if n_injections else float("nan"),
        "delay_median": med, "delay_ci_lo": lo, "delay_ci_hi": hi,
        "delay_q1": q1, "delay_q3": q3,
        "delay_mean": float(arr.mean()) if arr.size else float("nan"),
    }


# --------------------------------------------------------------------------- #
# AUPRC without point adjustment
# --------------------------------------------------------------------------- #


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision), no adjustment."""
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    if y_true.sum() == 0 or y_true.size == 0:
        return float("nan")
    order = np.argsort(-score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / y_true.sum()
    # average precision: sum over positives of precision at that recall
    ap = float(np.sum(precision[y] * np.diff(np.concatenate([[0.0], recall[y]]))))
    return ap


def pr_curve(y_true: np.ndarray, score: np.ndarray, n_points: int = 200) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    ok = np.isfinite(score)
    y_true, score = y_true[ok], score[ok]
    if y_true.size == 0:
        return {"precision": [], "recall": []}
    thr = np.quantile(score, np.linspace(1, 0, n_points))
    prec, rec = [], []
    for t in np.unique(thr):
        pred = score >= t
        tp = np.sum(pred & y_true)
        fp = np.sum(pred & ~y_true)
        fn = np.sum(~pred & y_true)
        prec.append(float(tp / max(tp + fp, 1)))
        rec.append(float(tp / max(tp + fn, 1)))
    return {"precision": prec, "recall": rec}


# --------------------------------------------------------------------------- #
# confusion matrix
# --------------------------------------------------------------------------- #


def confusion_matrix(y_true: list[str], y_pred: list[str], classes: tuple[str, ...]) -> dict:
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(y_true, y_pred):
        if a in idx and b in idx:
            M[idx[a], idx[b]] += 1
    return {"classes": list(classes), "matrix": M.tolist(),
            "accuracy": float(np.trace(M) / max(M.sum(), 1))}
