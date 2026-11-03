"""SP2 expectation: what should this sensor have read today?

The model is the one specified in section 7 of the specification:

    y_t = b0 + b1 * x_t + sum_{k=1..3} [ a_k cos(2 pi k t / 365.2422)
                                       + c_k sin(2 pi k t / 365.2422) ] + r_t

where ``x_t`` is the aligned NASA POWER value and the harmonics absorb the part
of the annual cycle that the reference gets wrong (an in-situ site is warmer or
colder than the reanalysis grid cell; the bias is seasonal, so one number is not
enough).

Why harmonics *and* a reference: the reference explains the synoptic state (the
part of the day-to-day variability that no calendar model can know), the
harmonics explain the site-specific seasonal bias.  Dropping the reference
removes all sub-seasonal information and the residual scale roughly triples --
that is measured in the ablation table.

Why Huber: a single day in the calibration window may already be corrupted (a
bias on one day of the calibration period is exactly the failure mode we are
defending against).  Least squares would let it pull the fit; Huber bounds the
influence of any one observation.  The tuning constant 1.345 gives 95 %
efficiency under Gaussian noise.

Why ``t`` is measured from the epoch in tropical years rather than from the
day of year: a leap year would otherwise insert a one-day phase step in the
harmonics, which is a spurious step in the residual -- the very thing D1 looks
for.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config
from .conditioning import Conditioned

PERIOD_DAYS = 365.2422


def harmonic_design(dates: np.ndarray, n_harm: int = 3, n_days: int | None = None) -> np.ndarray:
    """Harmonic design matrix [cos1, sin1, cos2, sin2, ...] (no intercept)."""
    d = dates.astype("datetime64[D]").astype(np.int64)
    if n_days is None:
        n_days = d - d[0]
    phase = 2.0 * np.pi * np.asarray(n_days, dtype=float) / PERIOD_DAYS
    cols = []
    for k in range(1, n_harm + 1):
        cols.append(np.cos(k * phase))
        cols.append(np.sin(k * phase))
    return np.column_stack(cols)


def _huber_weights(r: np.ndarray, scale: float, c: float = 1.345) -> np.ndarray:
    a = np.abs(r) / max(scale, 1e-12)
    w = np.ones_like(a)
    m = a > c
    w[m] = c / a[m]
    return w


def robust_scale(r: np.ndarray) -> float:
    """MAD scaled to a Gaussian standard deviation (1.4826 factor)."""
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


class InsufficientData(ValueError):
    """The pair has too little valid data for the model to be identified.

    Distinct from a bug: a station whose record is 99 % missing cannot support a
    fitted expectation, and the honest response is to exclude the pair and say so
    (see results/excluded_pairs.csv), not to fit a model to 13 observations.
    """


@dataclass
class HarmonicsReferenceModel:
    """Robust linear model with a reference regressor and annual harmonics."""

    n_harm: int = 3
    huber_c: float = 1.345
    n_iter: int = 25
    coef: np.ndarray | None = None
    sigma: float = float("nan")
    n_train: int = 0
    use_reference: bool = True
    extra: dict = field(default_factory=dict)

    # -- design -----------------------------------------------------------
    def _design(self, c: Conditioned, idx: np.ndarray | None = None) -> np.ndarray:
        dates = c.dates if idx is None else c.dates[idx]
        h = harmonic_design(dates, self.n_harm, n_days=None)
        cols = [np.ones((h.shape[0], 1))]
        if self.use_reference:
            x = c.x if idx is None else c.x[idx]
            cols.append(np.asarray(x, dtype=float).reshape(-1, 1))
        cols.append(h)
        return np.column_stack([np.asarray(v, dtype=float).reshape(h.shape[0], -1) for v in cols])

    # -- fit --------------------------------------------------------------
    def fit(self, c: Conditioned, mask: np.ndarray | None = None) -> "HarmonicsReferenceModel":
        cfg = config.load("detectors.yaml")["sp"]
        self.n_harm = int(cfg["harmonics"])
        self.huber_c = float(cfg["huber_c"])
        self.n_iter = int(cfg["robust_iterations"])

        sel = c.valid.copy() if mask is None else (c.valid & mask)
        X = self._design(c)[sel]
        y = c.y[sel]
        n, p = X.shape
        if n < 5 * p:
            raise InsufficientData(
                f"too few calibration samples ({n}) for {p} parameters: "
                "this pair cannot be modelled and is excluded from the evaluation")

        # start from the least-squares solution on the central 99.8 % of y
        lo, hi = np.quantile(y, [0.001, 0.999])
        inner = (y >= lo) & (y <= hi)
        coef, *_ = np.linalg.lstsq(X[inner], y[inner], rcond=None)
        scale = max(robust_scale(y - X @ coef), 1e-6)

        for _ in range(self.n_iter):
            r = y - X @ coef
            w = _huber_weights(r, scale, self.huber_c)
            sw = np.sqrt(w)
            coef_new, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            scale_new = max(robust_scale(y - X @ coef_new), 1e-6)
            if np.max(np.abs(coef_new - coef)) < 1e-9:
                coef = coef_new
                scale = scale_new
                break
            coef, scale = coef_new, scale_new

        self.coef = coef
        self.n_train = int(n)
        resid = y - X @ coef
        self.sigma = robust_scale(resid)
        self.extra["resid_rms"] = float(np.sqrt(np.mean(resid ** 2)))
        self.extra["train_mask"] = sel
        return self

    # -- predict ----------------------------------------------------------
    def predict(self, c: Conditioned) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("model not fitted")
        X = self._design(c)
        pred = X @ self.coef
        if self.use_reference:
            pred[~np.isfinite(c.x)] = np.nan
        return pred

    def residual(self, c: Conditioned) -> np.ndarray:
        pred = self.predict(c)
        r = c.y - pred
        r[~c.valid] = np.nan
        return r

    # -- interpretation ---------------------------------------------------
    def seasonal_baseline(self, c: Conditioned) -> np.ndarray:
        """Intercept + harmonics only (no reference regressor).

        Caution, learned the hard way: when the model contains the reference, the
        harmonics no longer describe the site's annual cycle -- the reference
        already contains it, so the harmonics only absorb the *residual* seasonal
        bias.  For CARTHAGE that is 1.8 degC instead of the site's 18.7 degC
        annual range, and using it as the percentage basis would shrink a "12 %
        step" by a factor of ten.  The percentage basis is therefore taken from a
        separate harmonic-only *climatology* (``fit_climatology`` below), never
        from this method when ``use_reference`` is True.
        """
        if self.coef is None:
            raise RuntimeError("model not fitted")
        h = harmonic_design(c.dates, self.n_harm)
        off = 2 if self.use_reference else 1
        return self.coef[0] + h @ self.coef[off:]

    def seasonal_range(self, c: Conditioned) -> float:
        base = self.seasonal_baseline(c)
        return float(np.nanmax(base) - np.nanmin(base))

    def describe(self) -> dict:
        return {
            "type": type(self).__name__,
            "n_harm": self.n_harm,
            "sigma": self.sigma,
            "n_train": self.n_train,
            "use_reference": self.use_reference,
            **self.extra,
        }


@dataclass
class GradientBoostingReferenceModel:
    """Tier-2 learned expectation model (kept only if the ablation shows a gain).

    Features: reference at lag -1, 0, +1, the two other variables' reference
    values when available, day-of-year sine/cosine, and a linear time trend.
    Gradient boosting is used because the station-POWER relation is
    heteroscedastic and mildly non-linear (inversions, snow, coastal
    advection); it is kept deliberately shallow so that memory stays small.
    """

    n_harm: int = 3  # unused, kept for interface symmetry
    model: object | None = None
    sigma: float = float("nan")
    n_train: int = 0
    use_reference: bool = True
    extra: dict = field(default_factory=dict)

    def _features(self, c: Conditioned, aux: dict[str, np.ndarray] | None = None) -> np.ndarray:
        x = c.x
        n = len(x)
        x_lag = np.concatenate([[np.nan], x[:-1]]) if n else x
        x_lead = np.concatenate([x[1:], [np.nan]]) if n else x
        ph = 2.0 * np.pi * (c.dates.astype("datetime64[D]").astype(np.int64) % PERIOD_DAYS) / PERIOD_DAYS
        cols = [x, x_lag, x_lead, np.cos(ph), np.sin(ph), np.cos(2 * ph), np.sin(2 * ph),
                np.linspace(0, 1, n)]
        if aux:
            for _k, v in aux.items():
                cols.append(v)
        return np.column_stack(cols)

    def fit(self, c: Conditioned, mask: np.ndarray | None = None) -> "GradientBoostingReferenceModel":
        from sklearn.ensemble import HistGradientBoostingRegressor

        sel = c.valid.copy() if mask is None else (c.valid & mask)
        X = self._features(c)[sel]
        y = c.y[sel]
        X = np.nan_to_num(X, nan=0.0)
        m = HistGradientBoostingRegressor(
            max_depth=4, max_iter=200, learning_rate=0.08, min_samples_leaf=40,
            l2_regularization=1.0, random_state=config.SEED,
        )
        m.fit(X, y)
        self.model = m
        self.n_train = int(sel.sum())
        resid = y - m.predict(X)
        self.sigma = robust_scale(resid)
        self.extra["resid_rms"] = float(np.sqrt(np.mean(resid ** 2)))
        return self

    def predict(self, c: Conditioned) -> np.ndarray:
        X = np.nan_to_num(self._features(c), nan=0.0)
        pred = np.asarray(self.model.predict(X), dtype=float)
        pred[~np.isfinite(c.x)] = np.nan
        return pred

    def residual(self, c: Conditioned) -> np.ndarray:
        r = c.y - self.predict(c)
        r[~c.valid] = np.nan
        return r

    def seasonal_baseline(self, c: Conditioned) -> np.ndarray:
        # no analytic seasonal baseline for the boosted model: use the smooth
        # harmonic model's baseline for the percentage definition instead.
        hm = HarmonicsReferenceModel().fit(c)
        return hm.seasonal_baseline(c)

    def seasonal_range(self, c: Conditioned) -> float:
        base = self.seasonal_baseline(c)
        return float(np.nanmax(base) - np.nanmin(base))

    def describe(self) -> dict:
        return {"type": type(self).__name__, "sigma": self.sigma, "n_train": self.n_train}


def fit_climatology(c: Conditioned, mask: np.ndarray | None = None) -> HarmonicsReferenceModel:
    """Harmonic-only fit of the sensor: the site's climatology.

    Used for (a) the seasonal range that defines what "12 %" means and (b) the
    anomaly road on which the replay detector matches shapes.  It intentionally
    contains *no* reference term: the climatology must be a property of the site,
    not of the reanalysis that is being used as a witness.
    """
    return HarmonicsReferenceModel(use_reference=False, n_harm=config.load("detectors.yaml")["sp"]["harmonics"]).fit(c, mask)


def fit_expectation(c: Conditioned, mask: np.ndarray | None = None, kind: str = "harmonic") -> object:
    """Fit the SP2 model on ``mask`` (calibration by default)."""
    if kind == "harmonic":
        return HarmonicsReferenceModel().fit(c, mask)
    if kind in ("gbm", "boost"):
        return GradientBoostingReferenceModel().fit(c, mask)
    if kind == "no_reference":
        return HarmonicsReferenceModel(use_reference=False).fit(c, mask)
    raise ValueError(kind)
