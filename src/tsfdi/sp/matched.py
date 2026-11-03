"""SP5 matched filtering: replay detection by self-similarity.

A replay attack copies a block of *earlier* readings of the same sensor into
the present.  Two independent fingerprints of that act exist:

1. **Exact repetition.**  The injected values are bit-identical to the values
   the sensor produced weeks or months ago (up to the data quantisation, 0.1 degC
   / 0.1 mm).  That is a search for an exactly repeated subsequence -- extremely
   specific and essentially free of false alarms because weather does not repeat
   to the decile.

2. **Shape repetition.**  Even if the attacker rescales or noises the copied
   block (gain, additive noise, see ``attacks.families.replay``), the *shape* is
   preserved.  The normalised cross-correlation (NCC) of the incoming window
   against all earlier windows measures that, and is invariant to gain and
   offset by construction.

Both statistics are computed here.  The NCC search uses cumulative sums for the
sliding mean/variance and one FFT convolution for all lags at once, so the cost
is O(n log n) rather than O(n L) -- this is what makes an online, every-window
search affordable (measured latency in results/tables/table_efficiency.csv).

Statistical note.  Under the null (no replay) the maximum NCC over ~n candidate
lags is the maximum of n weakly dependent correlations, so its null distribution
is *not* the distribution of a single correlation.  We never use a parametric
NCC threshold: the threshold is obtained from the same block-bootstrap
false-alarm calibration as the other detectors (spec section 8).

A search window is only compared with history that ends at least
``min_gap`` days before the window starts, so a legitimate overlap of the window
with itself cannot manufacture a match.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve


def sliding_moments(s: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding sum, mean and (population) std of every length-``w`` window."""
    s = np.nan_to_num(np.asarray(s, dtype=float), nan=0.0)
    n = len(s)
    if n < w:
        return np.zeros(0), np.zeros(0), np.zeros(0)
    cs = np.concatenate([[0.0], np.cumsum(s)])
    cs2 = np.concatenate([[0.0], np.cumsum(s * s)])
    tot = cs[w:] - cs[:-w]
    tot2 = cs2[w:] - cs2[:-w]
    mean = tot / w
    var = np.maximum(tot2 / w - mean ** 2, 0.0)
    return tot, mean, np.sqrt(var)


def ncc_profile(series: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of ``window`` against every position of ``series``.

    ``out[j] = corr(window, series[j:j+L])``.  Positions whose window is
    constant (zero variance) get NaN, not a spurious 1.0.
    """
    L = len(window)
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < L or L < 2:
        return np.full(max(n - L + 1, 0), np.nan)
    w = np.asarray(window, dtype=float)
    wc = w - w.mean()
    wnorm = np.sqrt(np.sum(wc ** 2))
    if wnorm <= 0:
        return np.full(n - L + 1, np.nan)
    # dot product of the zero-mean window with every placement
    dots = fftconvolve(np.nan_to_num(series, nan=0.0), wc[::-1], mode="valid")[: n - L + 1]
    _, _, sd = sliding_moments(series, L)
    denom = np.sqrt(L) * sd * wnorm
    with np.errstate(invalid="ignore", divide="ignore"):
        out = dots / denom
    out[sd <= 1e-12] = np.nan
    return np.clip(out, -1.0, 1.0)


@dataclass
class ReplayMatch:
    ncc: float
    lag_days: int | None          # distance from the current window's start to the match
    exact: bool                   # identical up to quantisation
    n_exact_values: int
    window_start: int
    length: int

    def as_dict(self) -> dict:
        return {
            "ncc": float(self.ncc),
            "lag_days": int(self.lag_days) if self.lag_days is not None else None,
            "exact": bool(self.exact),
            "n_exact_values": int(self.n_exact_values),
            "length": int(self.length),
        }


def quantised_equal(a: np.ndarray, b: np.ndarray, step: float = 0.1, tol: float = 1e-6) -> int:
    """Number of positions where ``a`` and ``b`` agree to the data quantisation."""
    if len(a) != len(b) or len(a) == 0:
        return 0
    qa = np.round(np.asarray(a, dtype=float) / step)
    qb = np.round(np.asarray(b, dtype=float) / step)
    ok = np.isfinite(qa) & np.isfinite(qb)
    return int(np.sum(np.abs(qa[ok] - qb[ok]) <= tol / step))


class ReplayDetector:
    """SP5: search the incoming window against the sensor's own earlier history."""

    def __init__(self, length: int = 7, min_source_gap: int = 2, quant_step: float | None = None):
        self.length = int(length)
        self.min_source_gap = int(min_source_gap)
        # GHCN-Daily reports tenths of a degree / tenths of a millimetre.
        self.quant_step = 0.1 if quant_step is None else float(quant_step)

    def evaluate(self, raw: np.ndarray, end_index: int) -> ReplayMatch:
        """Test the window ending at ``end_index`` (inclusive) against earlier history."""
        L = self.length
        start = end_index - L + 1
        if start < L + self.min_source_gap:      # not enough history yet
            return ReplayMatch(float("nan"), None, False, 0, start, L)
        window = raw[start: end_index + 1]
        if not np.all(np.isfinite(window)):
            return ReplayMatch(float("nan"), None, False, 0, start, L)
        hist_end = start - self.min_source_gap
        if hist_end < L:
            return ReplayMatch(float("nan"), None, False, 0, start, L)
        ncc = ncc_profile(raw[:hist_end + 1], window)
        if ncc.size == 0 or not np.isfinite(ncc).any():
            return ReplayMatch(float("nan"), None, False, 0, start, L)
        j = int(np.nanargmax(ncc))
        best = float(ncc[j])
        lag = int(start - j)
        n_exact = quantised_equal(window, raw[j: j + L], self.quant_step)
        # A degenerate window (fewer than 3 distinct values) carries no evidence:
        # a run of seven dry days repeats itself every week on honest data.  The
        # same rule is applied in detect.statistics.replay_exact; a constant run
        # is D4's (stuck-at) signature, not a replay.
        degenerate = np.unique(np.round(window / self.quant_step)).size < 3
        exact = bool(n_exact == L and not degenerate)
        return ReplayMatch(best, lag, exact, n_exact, start, L)

    def score_series(self, raw: np.ndarray, stride: int = 1) -> np.ndarray:
        """NCC statistic for every end index (used by the bootstrap calibration)."""
        n = len(raw)
        out = np.full(n, np.nan)
        for t in range(self.length - 1, n, stride):
            out[t] = self.evaluate(raw, t).ncc
        return out
