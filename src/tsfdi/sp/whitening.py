"""SP3 whitening: from residual to a calibrated innovation ``z``.

Two effects have to be removed before any sequential test is run, and both are
measurable in real data (see ``results/gonogo.md``):

1. **Temporal correlation.**  Weather residuals are strongly autocorrelated
   (lag-1 ACF typically 0.4-0.7).  CUSUM and EWMA assume independent
   observations; with correlated input the in-control run length is far shorter
   than the nominal value, so a threshold calibrated on white noise produces
   hundreds of false alarms per year.  We therefore fit an AR(p), p <= 5,
   selected by AIC, to the calibration residuals and keep the one-step
   prediction error (the innovation).

2. **Heteroscedasticity.**  The innovation scale is seasonally varying: the
   residual is larger in summer (convective variability) and smaller in winter
   at mid latitudes.  A single global scale would under-alarm in winter and
   over-alarm in summer.  We fit ``log sigma(t)`` with a two-harmonic model on
   monthly robust scales and divide.

The output ``z_t`` is (by construction) approximately standardised and
approximately uncorrelated; its *empirical* robust scale is re-checked at the
end and reported.  ``z`` -- not the raw residual -- is the input to D1, D2, D4,
D6 and to the false-alarm calibration.

3. **Marginal shape (precipitation only).**  A daily-rain residual is not
   Gaussian in any coordinate: 70 % of days are dry, so the residual distribution
   has a point mass at zero and a long right tail.  Standardising it by a robust
   scale (MAD x 1.4826) gives a ``z`` whose *empirical* standard deviation is 5-6,
   because the MAD is dominated by the dry-day spike; the sequential tests then
   operate far inside the tail and the false-alarm rate stops being controllable
   (measured: 13-15 alarms per sensor-year against a budget of 1).  We therefore
   map the calibration residual through its own empirical distribution function
   and then through the standard normal quantile function,
   ``w = Phi^-1(F_cal(r))`` -- the rank-Gauss (probability-integral) transform.
   This is monotone, so a positive bias remains a positive shift, and it is
   fitted on the calibration period only.  Beyond the calibration range the map
   is extrapolated linearly in the normal-quantile scale rather than clipped, so
   an extreme wet day still produces a large positive ``z`` (capped at 8) instead
   of a saturated one.  It is enabled per variable kind
   (``sp.rank_gauss_kinds`` in ``config/detectors.yaml``) and is part of the
   ablation, so its contribution is measured, not assumed.

Gaps: an innovation needs p+1 consecutive residuals.  Days that cannot form one
are NaN and are skipped by the detectors; no sample is invented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from .conditioning import Conditioned
from .expectation import PERIOD_DAYS, robust_scale


def _lag_pairs(r: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unfiltered (target, lagged design, row indices) for AR fitting with gaps.

    Returning the row indices is not cosmetic: an earlier version filtered the
    missing rows *inside* this helper and then applied a boolean row mask of the
    full length to the already-shortened array, which silently fitted the AR
    coefficients on the wrong rows (and produced an innovation scale larger than
    the residual scale, i.e. a p = 1 model that explained nothing).
    """
    n = len(r)
    t = np.arange(p, n)
    Y = r[p:]
    X = np.column_stack([r[p - i - 1: n - i - 1] for i in range(p)])
    return Y, X, t


def fit_ar(r: np.ndarray, p: int, rows: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Conditional least squares AR(p) with intercept; returns (phi, sigma_e).

    ``sigma_e`` is the *maximum-likelihood* innovation scale, ``sqrt(SSE / m)``,
    not ``std(resid, ddof=1)``.  This matters for the AIC scan below: if p = 0 is
    scored with a robust (MAD) scale and p >= 1 with a standard deviation, the
    two numbers are not comparable for a heavy-tailed residual and AIC
    systematically picks p = 0 (measured on the real bundle: it did, and the
    unwhitened z then carried a lag-1 autocorrelation of 0.29 into CUSUM).
    """
    if p == 0:
        Y = np.asarray(r, dtype=float) if rows is None else np.asarray(r, dtype=float)[rows]
        Y = Y[np.isfinite(Y)]
        if Y.size < 10:
            return np.zeros(0), robust_scale(r)
        mu = float(np.mean(Y))
        return np.zeros(0), float(np.sqrt(np.mean((Y - mu) ** 2)))
    Y, X, t = _lag_pairs(r, p)
    ok = np.isfinite(Y) & np.all(np.isfinite(X), axis=1)
    if rows is not None:
        ok &= np.asarray(rows, dtype=bool)[t]
    Y, X = Y[ok], X[ok]
    if len(Y) < 10 * (p + 1):
        return np.zeros(p), robust_scale(r)
    A = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ beta
    return beta[1:], float(np.sqrt(np.mean(resid ** 2)))


@dataclass
class RankGauss:
    """Monotone map from a residual to a standard normal score (empirical PIT)."""

    xs: np.ndarray | None = None        # sorted calibration residuals (knots)
    us: np.ndarray | None = None        # Phi^-1((i + 0.5) / n)
    slope_lo: float = 0.0               # tail slopes in the normal-quantile scale
    slope_hi: float = 0.0
    cap: float = 8.0
    n: int = 0

    @classmethod
    def fit(cls, r: np.ndarray, cap: float = 8.0) -> "RankGauss":
        from scipy.stats import norm

        v = np.asarray(r, dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 50:
            return cls()
        v.sort()
        # thin the knots to at most 2000 points (rank-interpolation is unchanged)
        if v.size > 2000:
            idx = np.linspace(0, v.size - 1, 2000).astype(int)
            knots = v[idx]
        else:
            knots = v
        n = v.size
        u = (np.arange(knots.size) + 0.5) / n if knots.size == n else \
            (np.linspace(0.5, n - 0.5, knots.size)) / n
        out = cls(xs=knots, us=norm.ppf(u), n=n, cap=float(cap))
        # tail slopes: local derivative in the last/first 5 % of the sample
        k = max(10, knots.size // 20)
        out.slope_lo = float((out.us[k] - out.us[0]) / max(knots[k] - knots[0], 1e-12))
        out.slope_hi = float((out.us[-1] - out.us[-1 - k]) / max(knots[-1] - knots[-1 - k], 1e-12))
        return out

    @property
    def fitted(self) -> bool:
        return self.xs is not None and self.xs.size > 2

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        if not self.fitted:
            return r.copy()
        out = np.interp(r, self.xs, self.us)          # piecewise-linear, monotone
        below, above = r < self.xs[0], r > self.xs[-1]
        if np.any(below):
            out[below] = self.us[0] + self.slope_lo * (r[below] - self.xs[0])
        if np.any(above):
            out[above] = self.us[-1] + self.slope_hi * (r[above] - self.xs[-1])
        out = np.clip(out, -self.cap, self.cap)
        out[~np.isfinite(r)] = np.nan
        return out

    def describe(self) -> dict:
        return {"enabled": bool(self.fitted), "n_knots": int(0 if self.xs is None else self.xs.size),
                "cap": float(self.cap)}


@dataclass
class Whitener:
    """SP3: AR(p) innovations with a seasonally varying robust scale."""

    p: int = 1
    phi: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intercept: float = 0.0
    sigma_e: float = float("nan")
    seasonal: np.ndarray | None = None      # (n, ) sigma(t) on the fitted grid; see scale()
    seasonal_coef: np.ndarray | None = None  # [c0, a1, b1, a2, b2]
    sigma_z: float = float("nan")           # empirical robust scale of z (should be ~1)
    aic_table: dict = field(default_factory=dict)
    var_ratio: float = float("nan")
    acf1: float = float("nan")
    rank_gauss: bool = False
    pit: RankGauss | None = None
    # causal trailing bias correction (see correct_bias); 0 disables it
    bias_window: int = 365
    bias_margin: int = 30
    bias_fallback: float = 0.0     # calibration-period median, used during burn-in
    # bounded causal scale guard (see guard_scale); disabled for the ablation
    scale_guard: bool = False
    gap_reset_days: int = 7
    guard_window: int = 365
    guard_margin: int = 30
    guard_bounds: tuple[float, float] = (0.75, 1.5)
    aic_rows: int = 0                       # common sample used by the AIC scan
    ar_pmax_effective: int = 0              # longest order with a usable sample

    # ------------------------------------------------------------------ fit
    def fit(self, c: Conditioned, resid: np.ndarray, mask: np.ndarray | None = None) -> "Whitener":
        cfg = config.load("detectors.yaml")["sp"]
        pmax = int(cfg["ar_max_order"])
        sel_mask = c.valid if mask is None else (c.valid & mask)
        finite_sel = resid[sel_mask & np.isfinite(resid)]
        self.bias_fallback = float(np.median(finite_sel)) if finite_sel.size else 0.0
        r = np.where(sel_mask, self.correct_bias(resid, sel_mask), np.nan)
        if self.rank_gauss:
            self.pit = RankGauss.fit(r[sel_mask & np.isfinite(r)])
            r = np.where(sel_mask, self.pit(resid), np.nan)
        n = int(np.isfinite(r).sum())

        # ---- 1. AR order by AIC ----------------------------------------
        # All orders are scored on the *same* rows (the rows where the longest
        # lag is observed), so the conditional likelihoods are comparable.
        n_all = len(r)
        n_finite = int(np.isfinite(r).sum())
        floor = max(300, int(0.15 * n_finite))
        # Longest order whose "all lags observed" sample is still usable.  With
        # 35 % missing days the intersection of six shifted validity masks can
        # collapse to a few dozen rows, so the common sample -- and hence the
        # order -- has to be chosen before any AIC is computed.
        pmax_eff = 0
        rows = np.ones(n_all, dtype=bool)
        for P in range(pmax, -1, -1):
            cand = np.ones(n_all, dtype=bool)
            for k in range(P + 1):
                shifted = np.ones(n_all, dtype=bool)
                shifted[k:] = np.isfinite(r[: n_all - k])
                cand &= shifted
            if cand.sum() >= floor or P == 0:
                pmax_eff, rows = P, cand
                break
        m = int(rows.sum())
        self.aic_rows = m
        self.ar_pmax_effective = int(pmax_eff)
        best = (np.inf, 0, np.zeros(0), robust_scale(r[np.isfinite(r)]))
        if m >= 50:
            for p in range(0, pmax_eff + 1):
                phi, se = fit_ar(r, p, rows=rows)
                if not np.isfinite(se) or se <= 0:
                    continue
                k = p + 1
                aic = m * np.log(se ** 2) + 2 * k
                self.aic_table[p] = float(aic)
                if aic < best[0]:
                    best = (aic, p, phi, se)
        _, self.p, self.phi, self.sigma_e = best
        self.p = int(self.p)

        # ---- 2. innovations --------------------------------------------
        # NB: ``r`` is the residual *after* the optional PIT (normal-score space).
        # Fitting the seasonal scale on the pre-PIT residual here and standardising
        # the post-PIT innovation at run time is a silent scale mismatch that made
        # the precipitation z series ~7 sd wide; the same object must be used for
        # both steps.
        e = self.innovations(r)

        # ---- 3. seasonal scale -----------------------------------------
        self._fit_seasonal_scale(c, e, sel_mask)

        # ---- 4. z ------------------------------------------------------
        z = self.guard_scale(self.standardise(c, e))
        ok = np.isfinite(z)
        if ok.sum() > 100:
            self.sigma_z = robust_scale(z[ok])
            zz = z[ok]
            if len(zz) > 10:
                self.acf1 = float(np.corrcoef(zz[:-1], zz[1:])[0, 1])
            rv = resid[sel_mask & np.isfinite(resid)]
            self.var_ratio = float(np.var(rv) / np.var(zz)) if np.var(zz) > 0 else float("nan")
        return self

    # ------------------------------------------------------------- helpers
    # ------------------------------------------------------ bias correction
    def correct_bias(self, resid: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Remove a *slowly varying* sensor/reference bias (causal, robust).

        The station-minus-reanalysis bias is not stationary.  Measured on
        JENDOUBA (TMAX) the annual mean residual moves from +0.24 degC in 2015 to
        -0.90 degC in 2024 -- about one sigma over the record -- because the
        station (instrument, exposure, site) and the reanalysis both drift.  A
        drift detector with a 5-year-calibrated threshold sees that as a permanent
        bias and alarms for years: the measured false-alarm rate was 6 per
        sensor-year against a budget of 1, with the drift CUSUM contributing 4 of
        them.  No stationary threshold can fix that; the non-stationarity has to be
        removed from the statistic.

        ``resid - median(resid over the trailing ``bias_window`` days, ending
        ``bias_margin`` days ago)`` does exactly that, and it is causal, so it is
        legal in a streaming system.  The cost is explicit and is evaluated rather
        than hidden: a bias that develops *slower* than the window chases its own
        baseline and becomes invisible on the residual-mean channel.  That is
        precisely the A2 black-box attacker's optimal strategy, which is why the
        ramp family is evaluated down to 1 sigma and why the evasion curve is
        reported (Figure 2).

        The median is used, not the mean, so that a long attack cannot drag its own
        baseline; the 30-day margin keeps an active episode out of the window that
        would normalise it.
        """
        r = np.asarray(resid, dtype=float)
        if self.bias_window is None or int(self.bias_window) <= 0:
            return r.copy()
        import pandas as pd

        ser = pd.Series(np.where(mask, r, np.nan) if mask is not None else r)
        base = (ser.rolling(window=int(self.bias_window), min_periods=30)
                   .median().shift(int(self.bias_margin)))
        b = base.to_numpy(dtype=float)
        # Burn-in: before 30 usable days of history (including the first records of
        # a stream) the best available estimate of the bias is the calibration
        # median, which is exactly the "no correction" model with an offset.
        if np.isfinite(self.bias_fallback):
            b = np.where(np.isfinite(b), b, self.bias_fallback)
        else:
            b = np.where(np.isfinite(b), b, 0.0)
        out = r - b
        out[~np.isfinite(r)] = np.nan
        return out

    def guard_scale(self, z: np.ndarray) -> np.ndarray:
        """Divide z by its own slow-moving robust scale, within documented bounds.

        Why this exists: thresholds are calibrated on 2015-2019 and applied to
        2020-2024.  The *robust scale* of z is not constant across that boundary --
        measured over the whole bundle it is 4 % larger in the test period on the
        median pair and up to 16 % larger on some -- and a tail threshold is
        extremely sensitive to that: at the 99.9 % quantile a 10 % scale inflation
        multiplies the exceedance rate by roughly 3, which is exactly the gap
        observed between the in-sample (0.4-0.5/yr) and out-of-sample (2.5/yr)
        false-alarm rates.  Dividing z by a *causal* trailing MAD (previous 365
        days, ending 30 days ago) makes every threshold scale-free without using a
        single test-period day for fitting: the normalisation is computed from the
        past only, so it is legal in a stream.

        Two guards keep the cure from being worse than the disease:
        the correction is clipped to ``guard_bounds`` (0.75-1.5), so a pathological
        window cannot inflate or collapse the statistic; and the median/MAD is
        robust to up to 50 % contamination, so neither a long outage nor a
        months-long attack can drag the reference scale towards itself.
        """
        z = np.asarray(z, dtype=float)
        if not self.scale_guard:
            return z
        import pandas as pd

        med = pd.Series(np.abs(z)).rolling(window=int(self.guard_window), min_periods=60).median()
        g = (1.4826 * med).shift(int(self.guard_margin)).to_numpy(dtype=float)
        g = np.where(np.isfinite(g) & (g > 0), g, 1.0)
        lo, hi = self.guard_bounds
        g = np.clip(g, float(lo), float(hi))
        out = z / g
        out[~np.isfinite(z)] = np.nan
        return out

    def to_normal(self, resid: np.ndarray) -> np.ndarray:
        """Apply the empirical PIT if it is enabled, otherwise pass through."""
        if self.rank_gauss and self.pit is not None and self.pit.fitted:
            return self.pit(resid)
        return np.asarray(resid, dtype=float)

    def gap_reset_mask(self, resid: np.ndarray, days: int | None = None) -> np.ndarray:
        """Mask the observations that follow a long outage (see ``innovations``).

        A 300-day outage followed by a report is not "yesterday's value was
        missing": the AR recursion would be predicting today from a state that is a
        year old, and the first few innovations after the outage are dominated by
        that staleness rather than by the weather.  On the real bundle this is not
        hypothetical -- KHARGA reports 12 days in all of 2021 -- and it inflates the
        alarm rate of exactly the stations the challenge cares about.

        Rule: after a run of missing residuals longer than ``gap_reset_days``
        (default 7, i.e. longer than the AR memory), the next ``p + 1`` present
        observations are not whitened and carry no z.  Short gaps keep the
        state-space convention (missing = conditional mean), which is what makes a
        65 %-complete station usable at all.
        """
        import pandas as pd

        r = np.asarray(resid, dtype=float)
        days = int(self.gap_reset_days if days is None else days)
        if days <= 0:
            return np.zeros(len(r), dtype=bool)
        miss = ~np.isfinite(r)
        # length of the run of missing days ending at each day
        run = np.zeros(len(r), dtype=int)
        c = 0
        for i in range(len(r)):
            c = c + 1 if miss[i] else 0
            run[i] = c
        # a valid day t has a preceding run of missing days given by run[t-1]
        prev = np.concatenate([[0], run[:-1]])
        after_gap = (prev > days) & np.isfinite(r)
        out = after_gap.copy()
        m = int(self.p) + 1
        for k in range(1, m):                    # also suppress the next p observations
            out |= np.concatenate([np.zeros(k, bool), after_gap[:-k]])
        return out

    def innovations(self, resid: np.ndarray) -> np.ndarray:
        """One-step prediction errors of the AR(p) model, gap-aware.

        Missing days are handled the way a state-space model handles a missing
        observation: the *filter state* continues, the missing residual enters the
        recursion at its conditional mean (zero under the fitted AR model), and no
        ``z`` is emitted for a day whose own observation is absent.

        This matters a great deal in practice and is not a detail.  Skipping the
        recursion whenever any of the p lags is missing (the naive alternative)
        throws away every sample that follows a gap: for a station that reports
        65 % of its days -- CARTHAGE in this bundle -- an AR(5) filter would keep
        only 0.65**6 ~ 7 % of the calibration samples and the whitening would be
        estimated from a few dozen points.  Zero-imputation keeps ~65 % and biases
        the innovation variance downwards by a bounded amount, which the
        bootstrap false-alarm calibration then absorbs because it sees exactly the
        same gap structure.
        """
        r = np.asarray(resid, dtype=float)
        n = len(r)
        e = np.full(n, np.nan)
        if self.p == 0:
            return r.copy()
        phi = np.atleast_1d(self.phi)
        filled = np.where(np.isfinite(r), r, 0.0)
        stale = None
        if self.gap_reset_days and self.gap_reset_days > 0:
            stale = self.gap_reset_mask(r)
        for t in range(self.p, n):
            if not np.isfinite(r[t]):
                continue
            if stale is not None and stale[t]:
                continue
            hist = filled[t - self.p: t][::-1]      # r[t-1], r[t-2], ...
            e[t] = r[t] - float(phi @ hist)
        return e

    def _doy(self, dates: np.ndarray) -> np.ndarray:
        d = dates.astype("datetime64[D]").astype(np.int64)
        return 2.0 * np.pi * (d % PERIOD_DAYS) / PERIOD_DAYS

    def _harm(self, ph: np.ndarray, k: int = 2) -> np.ndarray:
        cols = [np.ones_like(ph)]
        for i in range(1, k + 1):
            cols += [np.cos(i * ph), np.sin(i * ph)]
        return np.column_stack(cols)

    def _fit_seasonal_scale(self, c: Conditioned, e: np.ndarray, mask: np.ndarray) -> None:
        """Fit log sigma(t) with two harmonics on monthly robust scales."""
        months = pd.DatetimeIndex(c.dates).month.to_numpy()
        log_s, ph_s = [], []
        for m in range(1, 13):
            sel = (months == m) & mask & np.isfinite(e)
            if sel.sum() < 30:
                continue
            s = robust_scale(e[sel])
            if not np.isfinite(s) or s <= 0:
                continue
            log_s.append(np.log(s))
            ph_s.append(2.0 * np.pi * (m - 0.5) / 12.0)
        if len(log_s) < 6:
            self.seasonal_coef = np.array([np.log(max(robust_scale(e[mask & np.isfinite(e)]), 1e-6)), 0, 0, 0, 0])
        else:
            A = self._harm(np.asarray(ph_s), 2)
            beta, *_ = np.linalg.lstsq(A, np.asarray(log_s), rcond=None)
            self.seasonal_coef = beta
        ph_all = self._doy(c.dates)
        self.seasonal = np.exp(self._harm(ph_all, 2) @ self.seasonal_coef)

    def scale(self, c: Conditioned) -> np.ndarray:
        if self.seasonal_coef is None:
            raise RuntimeError("not fitted")
        ph_all = self._doy(c.dates)
        return np.exp(self._harm(ph_all, 2) @ self.seasonal_coef)

    def standardise(self, c: Conditioned, e: np.ndarray) -> np.ndarray:
        s = self.scale(c)
        z = e / np.where(s > 0, s, np.nan)
        z[c.valid == False] = np.nan  # noqa: E712  (explicit, keep readability)
        return z

    def transform(self, c: Conditioned, resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (innovations, z) for a raw residual road.

        Order matters and is fixed in one place so that every consumer (evaluation,
        streaming, attackers, ablations) applies the same chain:
        trailing-bias correction -> rank-Gauss PIT -> AR innovations -> seasonal scale.
        """
        rc = self.correct_bias(resid)
        e = self.innovations(self.to_normal(rc))
        return e, self.guard_scale(self.standardise(c, e))

    # ------------------------------------------------------------- reports
    def describe(self) -> dict:
        return {
            "ar_order": int(self.p),
            "phi": [float(v) for v in np.atleast_1d(self.phi)],
            "sigma_e": float(self.sigma_e),
            "sigma_z_empirical": float(self.sigma_z),
            "acf1_z": float(self.acf1),
            "seasonal_coef": [float(v) for v in np.atleast_1d(self.seasonal_coef)] if self.seasonal_coef is not None else [],
            "aic": {int(k): float(v) for k, v in self.aic_table.items()},
            "aic_rows": int(self.aic_rows),
            "ar_pmax_effective": int(self.ar_pmax_effective),
            "bias_window": int(self.bias_window) if self.bias_window else 0,
            "scale_guard": bool(self.scale_guard),
            "gap_reset_days": int(self.gap_reset_days or 0),
            "rank_gauss": bool(self.rank_gauss),
            "pit": self.pit.describe() if self.pit is not None else {"enabled": False},
        }
