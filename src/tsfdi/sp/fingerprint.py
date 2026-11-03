"""SP6 spectral fingerprint: the statistical "texture" of the residual.

SP1-SP3 turn the pair (sensor, reference) into ``z``.  That is a *level*
summary: it asks "is today too high?".  An attacker who replaces the sensor by
the reference expectation (adversary A1) produces a series that is never too
high -- it is *too well behaved*.  Level statistics cannot see that; the second
order statistics can, because a copy has no independent noise.

The fingerprint of a rolling window is

    v = [ log var-ratio,  ACF(1..7) of z,  log band powers of z,  coherence bands ]

with the bands of the specification: low < 1/30 cycles/day, mid, high > 1/7
cycles/day.  The variance ratio is var(z on the window) / var(z in calibration);
a pure copy drives it towards 0, the coherence between sensor and reference
towards 1.  Both are two-sided alarms: an abnormally clean residual is as
suspicious as an abnormally noisy one.

The run-time statistic is a robust Mahalanobis distance of ``v`` to the
calibration cloud.  "Robust" here means: each coordinate is centred on its
median and scaled by 1.4826 MAD before the covariance is estimated with Ledoit-
Wolf shrinkage, so that a few contaminated calibration windows cannot inflate
the Mahalanobis distance of everything else, and the sample covariance stays
invertible when windows overlap heavily (they do: a 180-day window moved by 7
days gives 96 % shared samples).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from .expectation import robust_scale

BANDS = {
    "low": (0.0, 1.0 / 30.0),
    "mid": (1.0 / 30.0, 1.0 / 7.0),
    "high": (1.0 / 7.0, 0.5),
}


def _band_power(f: np.ndarray, p: np.ndarray, lo: float, hi: float) -> float:
    m = (f >= lo) & (f < hi)
    if not np.any(m):
        return float("nan")
    return float(np.trapezoid(p[m], f[m]) if np.any(m.sum() > 1) else np.sum(p[m]))


def _segment_fft(v: np.ndarray, nperseg: int) -> tuple[np.ndarray, np.ndarray]:
    """Welch segments of ``v`` with 50 % overlap, returned as an rFFT batch.

    A hand-rolled implementation (numpy rFFT on a (n_seg, nperseg) matrix) is
    used instead of ``scipy.signal.welch`` because the fingerprint is evaluated
    on ~260 overlapping windows per series and, during bootstrap calibration, on
    120 resampled series per detector: the batch FFT is ~20x faster and the
    estimator is the same (Hann window, 50 % overlap, one-sided density PSD);
    ``tests/test_sp.py`` asserts that the *band-integrated* power -- the
    quantity actually used -- agrees with ``scipy.signal.welch`` /
    ``scipy.signal.coherence`` to within 3 %, and that ``scipy`` itself is
    reproduced to the same tolerance.
    """
    nperseg = int(min(nperseg, len(v)))
    step = max(nperseg // 2, 1)
    starts = list(range(0, len(v) - nperseg + 1, step))
    if not starts:
        starts = [0]
    idx = np.arange(nperseg)[None, :] + np.asarray(starts)[:, None]
    seg = v[idx] if idx.max() < len(v) else None
    if seg is None:
        pad = np.pad(v, (0, idx.max() - len(v) + 1), mode="edge")
        seg = pad[idx]
    win = np.hanning(nperseg)
    seg = (seg - seg.mean(axis=1, keepdims=True)) * win[None, :]
    F = np.fft.rfft(seg, axis=1)
    freqs = np.fft.rfftfreq(nperseg, d=1.0)
    return F, freqs


def _welch_fast(v: np.ndarray, nperseg: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """One-sided Welch PSD, density scaling (fs = 1 sample/day)."""
    F, freqs = _segment_fft(np.asarray(v, dtype=float), nperseg)
    win = np.hanning(2 * (len(freqs) - 1))
    psd = np.mean(np.abs(F) ** 2, axis=0) / np.sum(win ** 2)      # fs = 1
    if len(psd) > 2:
        psd[1:-1] *= 2.0                                          # one-sided density
    return freqs, psd


def _coherence_fast(x: np.ndarray, y: np.ndarray, nperseg: int = 64) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    Fx, freqs = _segment_fft(np.asarray(x[:n], dtype=float), nperseg)
    Fy, _ = _segment_fft(np.asarray(y[:n], dtype=float), nperseg)
    Pxx = np.mean(np.abs(Fx) ** 2, axis=0) + 1e-18
    Pyy = np.mean(np.abs(Fy) ** 2, axis=0) + 1e-18
    Pxy = np.mean(Fx * np.conj(Fy), axis=0)
    return freqs, np.abs(Pxy) ** 2 / (Pxx * Pyy)


def _acf(x: np.ndarray, max_lag: int = 7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < max_lag + 5:
        return np.full(max_lag, np.nan)
    x = x - x.mean()
    den = float(np.dot(x, x))
    if den <= 0:
        return np.full(max_lag, np.nan)
    return np.array([float(np.dot(x[:-k], x[k:]) / den) for k in range(1, max_lag + 1)])


@dataclass
class FingerprintModel:
    """SP6: rolling spectral fingerprint and its Mahalanobis distance."""

    window: int = 180
    min_window: int = 120
    nperseg: int = 64
    stride: int = 7
    centre: np.ndarray | None = None
    scale: np.ndarray | None = None
    inv_cov: np.ndarray | None = None
    calib_dist: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)

    # -------------------------------------------------------------- features
    def window_features(self, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray | None:
        """Fingerprint vector of one window (NaN-padded when data are missing)."""
        ok = np.isfinite(z)
        if ok.sum() < self.min_window:
            return None
        zz = z[ok]
        s_z = robust_scale(zz)
        var_ratio = s_z ** 2 if np.isfinite(s_z) else np.nan
        acf = _acf(zz, 7)
        f, p = _welch_fast(zz - np.mean(zz), self.nperseg)
        bands = [_band_power(f, p, lo, hi) for lo, hi in BANDS.values()]
        oks = np.isfinite(y) & np.isfinite(x)
        if oks.sum() >= self.min_window:
            f2, cxy = _coherence_fast(y[oks], x[oks], self.nperseg)
            coh = [_band_power(f2, cxy, lo, hi) for lo, hi in BANDS.values()]
        else:
            coh = [np.nan, np.nan, np.nan]
        v = np.array([np.log(var_ratio + 1e-12), *acf,
                      *np.log(np.array(bands) + 1e-12), *coh])
        if not np.all(np.isfinite(v)):
            return None
        return v

    def names(self) -> list[str]:
        return (["log_var_ratio"] + [f"acf{i}" for i in range(1, 8)]
                + [f"log_band_{b}" for b in BANDS] + [f"coh_{b}" for b in BANDS])

    # ------------------------------------------------------------------ fit
    def fit(self, z: np.ndarray, y: np.ndarray, x: np.ndarray, mask: np.ndarray | None = None) -> "FingerprintModel":
        V = self._rolling_matrix(z, y, x, mask)
        if V is None or V.shape[0] < 12:
            self.centre = self.scale = self.inv_cov = None
            return self
        centre = np.nanmedian(V, axis=0)
        scale = 1.4826 * np.nanmedian(np.abs(V - centre), axis=0)
        scale[~np.isfinite(scale) | (scale < 1e-9)] = 1.0
        U = (V - centre) / scale
        U = U[np.all(np.isfinite(U), axis=1)]
        if U.shape[0] < 12:
            self.centre = self.scale = self.inv_cov = None
            return self
        cov = _ledoit_wolf(U)
        self.centre, self.scale = centre, scale
        self.inv_cov = np.linalg.pinv(cov)
        self.calib_dist = self._distance(U)
        self.feature_names = self.names()
        return self

    def _rolling_matrix(self, z, y, x, mask=None) -> np.ndarray | None:
        n = len(z)
        rows = []
        for end in range(self.window, n + 1, self.stride):
            sl = slice(end - self.window, end)
            if mask is not None and mask[sl].mean() < 0.5:
                continue
            v = self.window_features(z[sl], y[sl], x[sl])
            if v is not None:
                rows.append(v)
        if not rows:
            return None
        return np.vstack(rows)

    def _distance(self, U: np.ndarray) -> np.ndarray:
        d = np.einsum("ij,jk,ik->i", U, self.inv_cov, U)
        return np.sqrt(np.maximum(d, 0.0))

    # -------------------------------------------------------------- runtime
    def distance_series(self, z: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Mahalanobis distance at every ``stride``-th day (NaN elsewhere).

        The value is *held* between updates: the fingerprint is a slowly varying
        statistic and recomputing a 180-day spectral estimate every day would
        cost more than it buys (measured: see table_efficiency.csv).
        """
        n = len(z)
        out = np.full(n, np.nan)
        if self.inv_cov is None:
            return out
        last = np.nan
        for end in range(1, n + 1):
            if end % self.stride == 0 and end >= self.window:
                v = self.window_features(z[end - self.window: end], y[end - self.window: end], x[end - self.window: end])
                last = np.nan
                if v is not None and np.all(np.isfinite(v)):
                    U = (v - self.centre) / self.scale
                    last = float(np.sqrt(max(U @ self.inv_cov @ U, 0.0)))
            out[end - 1] = last
        return out

    def relative_distance(self, d: np.ndarray, window: int = 730, min_periods: int = 90,
                          margin: int = 30) -> np.ndarray:
        """Self-normalised fingerprint statistic: ``d_t / median(d in the past)``.

        A Mahalanobis distance is only a test statistic *if the feature
        distribution is stationary*.  It is not: on this bundle the median
        calibration distance is ~3.5 (2015-2019) and ~5.5 (2020-2024) for the same
        station, because the station/reanalysis relationship itself changes
        (see the annual bias table in `results/report.md`).  A threshold calibrated
        on 2015-2019 therefore fires throughout the test period without anything
        being wrong -- measured: 2.4 false alarms per sensor-year from D6 alone,
        on every site, temperature and precipitation alike.

        The fix keeps the detector's *purpose* (detect a change in the spectral
        fingerprint) and removes the stationarity assumption: the statistic is the
        distance divided by a robust baseline estimated from the recent past only
        (a trailing 2-year median, excluding the most recent 30 days so that a
        genuine change cannot hide inside its own baseline).  A sudden replacement
        of the sensor by a copy of the reference moves the ratio; a slow drift of
        the station/reanalysis bias moves the numerator and the baseline together
        and leaves the ratio alone.

        Consequence, stated because it is a real limitation: a *very slow*
        corruption of the fingerprint (slower than the 2-year baseline) becomes
        invisible to this channel.  That is the same trade-off as A2 black-box
        evasion and it is why the ramp attack is evaluated down to 1 sigma.
        """
        d = np.asarray(d, dtype=float)
        n = len(d)
        if n == 0:
            return d.copy()
        import pandas as pd

        ser = pd.Series(d)
        # trailing median over the past ``window`` *available* samples, ending
        # ``margin`` days before t (so an event cannot hide inside its own baseline)
        base = ser.rolling(window=window, min_periods=min_periods).median().shift(margin)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = d / base.to_numpy(dtype=float)
        out[~np.isfinite(d)] = np.nan
        return out

    def too_clean_flags(self, z: np.ndarray, y: np.ndarray, x: np.ndarray,
                        var_ratio_min: float = 0.02, coh_max: float = 0.999) -> np.ndarray:
        """Binary flag for the "impossibly clean" branch of D6."""
        n = len(z)
        out = np.zeros(n, dtype=bool)
        for end in range(self.window, n + 1, self.stride):
            v = self.window_features(z[end - self.window:end], y[end - self.window:end], x[end - self.window:end])
            if v is None:
                continue
            var_ratio = float(np.exp(v[0]))
            coh = np.nanmax(v[-3:])
            flag = (var_ratio < var_ratio_min) or (np.isfinite(coh) and coh > coh_max)
            out[end - self.stride: end] = flag
        return out

    def describe(self) -> dict:
        return {
            "window": self.window,
            "stride": self.stride,
            "nperseg": self.nperseg,
            "n_calib_windows": int(len(self.calib_dist)) if self.calib_dist is not None else 0,
            "calib_dist_median": float(np.median(self.calib_dist)) if self.calib_dist is not None else float("nan"),
            "calib_dist_p99": float(np.quantile(self.calib_dist, 0.99)) if self.calib_dist is not None else float("nan"),
        }


def _ledoit_wolf(U: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance (shrink towards a scaled identity)."""
    n, p = U.shape
    S = np.cov(U, rowvar=False, ddof=1)
    S = np.atleast_2d(S)
    mu = np.trace(S) / p
    delta = np.sum((S - mu * np.eye(p)) ** 2)
    # variance of the sample covariance entries (Ledoit-Wolf 2004, eq. 3)
    Uc = U - U.mean(axis=0)
    beta = sum(np.sum((np.outer(row, row) - S) ** 2) for row in Uc) / (n ** 2)
    shrink = min(beta, delta) / delta if delta > 0 else 1.0
    shrink = float(np.clip(shrink, 1e-3, 1.0))
    return shrink * mu * np.eye(p) + (1 - shrink) * S
