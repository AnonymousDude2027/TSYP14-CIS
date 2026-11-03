"""The per-(site, variable) detection model: SP chain + calibrated detector suite.

``SiteVariableModel.build`` is the only place where thresholds are produced.
It is fitted on the calibration period, on calibration data only, and it is
cached to ``results/cache`` so that the evaluation runs do not re-calibrate.

The model answers three questions:

    model.statistics(...)  -> the raw statistic series (evidence)
    model.alarms(...)      -> threshold exceedances, per detector and combined
    model.explain(...)     -> the evidence dictionary used by L4

Everything the evaluation needs is derived from those three, so there is exactly
one implementation of each detector (no "evaluation version" that differs from
the "demo version").
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config
from ..io import SeriesBundle, load_pair
from ..sp import Conditioning, Whitener, fit_expectation
from ..sp.expectation import fit_climatology
from ..sp.expectation import robust_scale
from ..sp.fingerprint import FingerprintModel
from ..sp.surrogate import surrogate_replicates
from . import statistics as st
from .thresholds import block_bootstrap_indices, calibrate_threshold

# Detector registry -------------------------------------------------------- #
# name -> (family, target, description)
DETECTORS: dict[str, tuple[str, str, str]] = {
    "D1_step_cusum":      ("D1", "step",   "two-sided CUSUM, k=1.0 sigma"),
    "D1_window_shift_3":  ("D1", "step",   "3-day standardised mean shift"),
    "D1_window_shift_7":  ("D1", "step",   "7-day standardised mean shift"),
    "D2_drift_cusum":     ("D2", "drift",  "two-sided CUSUM, k=0.5 sigma"),
    "D2_ewma":            ("D2", "drift",  "EWMA, lambda=0.1, standardised"),
    "D2_slope_14":        ("D2", "drift",  "14-day OLS slope test"),
    "D2_slope_30":        ("D2", "drift",  "30-day OLS slope test"),
    "D3_replay_ncc":      ("D3", "replay", "max self-NCC of the last 7 days"),
    "D3_replay_exact":    ("D3", "replay", "exact 7-day repeat at data quantisation"),
    "D3_block_contrast":  ("D3", "replay", "L-day level contrast, matched to a block replacement"),
    "D4_spike":           ("D4", "fault",  "|z| beyond the calibration envelope"),
    "D6_fingerprint":     ("D6", "copy",   "robust Mahalanobis distance of the spectral fingerprint"),
}
BOOT_N = {"D3_replay_ncc": 120, "D6_fingerprint": 120}
# Detectors whose false-alarm calibration uses phase-randomised surrogates rather
# than the circular block bootstrap: a block bootstrap duplicates blocks and so
# fabricates the exact repetitions D3 is meant to detect (see sp/surrogate.py).
SURROGATE_CALIBRATED = {"D3_replay_ncc", "D3_replay_exact"}


@dataclass
class SiteVariableModel:
    site: str
    variable: str
    budget: float
    conditioned: object                 # Conditioning output over the full period
    expectation: object                 # SP2
    whitener: Whitener                  # SP3
    fingerprint: FingerprintModel       # SP6
    residual: np.ndarray
    z: np.ndarray
    thresholds: dict[str, float]
    calibration: dict[str, dict]
    sigma_residual: float
    seasonal_range: float
    sigma_z: float
    meta: dict = field(default_factory=dict)
    stuck_threshold: float = float("inf")
    availability: object = None          # D7 AvailabilityMonitor
    climatology: object = None

    # ---------------------------------------------------------------- stats
    @property
    def cal_mask(self) -> np.ndarray:
        return self._cal

    @property
    def kind(self) -> str:
        """SP1 variable kind ("temperature" / "multiplicative"), from Conditioning.

        Several consumers (the injector, the natural-event labeller, the static
        report) need to know whether a percentage bias is a share of the seasonal
        range or a multiplicative factor.  Exposing it on the model keeps that
        decision in one place instead of re-deriving it from the variable name.
        """
        return getattr(self.conditioned, "kind", "temperature")

    @property
    def n(self) -> int:
        return len(self.z)

    def _arrays(self):
        c = self.conditioned
        return c.y, c.x, self.z

    def z_for(self, y_raw: np.ndarray) -> np.ndarray:
        """z for an arbitrary *raw* sensor road, using the fitted SP1/SP2/SP3.

        Single entry point for every consumer (evaluation, streaming pipeline,
        adaptive attackers, ablations) so that the SP1 working-coordinate map
        (square root for precipitation) can never be applied in one place and
        forgotten in another.
        """
        import copy

        from ..sp.conditioning import to_working

        c = copy.copy(self.conditioned)
        c.y = to_working(np.asarray(y_raw, dtype=float), c.kind)
        c.valid = np.isfinite(c.y) & np.isfinite(c.x)
        pred = self.expectation.predict(c)
        r = c.y - pred
        r[~c.valid] = np.nan
        _, z = self.whitener.transform(c, r)
        return z

    def anomaly(self) -> np.ndarray:
        """Sensor anomaly against the *climatology* only (SP2 harmonics, no reference).

        This is the road on which the matched filter (SP5) looks for repeats.
        Matching on the raw level does not work: a strongly seasonal series
        repeats its own annual cycle, so the maximum normalised cross-correlation
        over one year of history saturates at 1.0 every day of the year and the
        detector becomes blind (measured on synthetic and real data alike).  The
        anomaly road removes the deterministic annual cycle while keeping the
        day-to-day shape that a replay actually copies.
        """
        c = self.conditioned
        base = self.climatology.seasonal_baseline(c)
        a = c.y - base
        a[~np.isfinite(c.y)] = np.nan
        return a

    def statistics(self, idx: np.ndarray | None = None,
                   names: list[str] | None = None) -> dict[str, np.ndarray]:
        """Compute detector statistics.  ``idx`` (R, n) triggers bootstrap mode.

        ``names`` restricts the computation, which matters for calibration: the
        cheap array-based statistics are bootstrapped with 400 replicates while
        the two loop-based ones (matched filter, fingerprint) use fewer, and
        nothing is computed twice.
        """
        want = (lambda k: True) if names is None else (lambda k: k in set(names))
        c = self.conditioned
        anom_full = self.anomaly()
        if idx is None:
            y = c.y[:, None]; x = c.x[:, None]; z = self.z[:, None]; raw = c.y[:, None]
            anom = anom_full[:, None]
            R = 1
        else:
            idx = np.atleast_2d(idx)
            raw = c.y[idx].T          # (n, R)
            x = c.x[idx].T
            z = self.z[idx].T
            anom = anom_full[idx].T
            y = raw
            R = idx.shape[0]
        out: dict[str, np.ndarray] = {}
        cfg = config.load("detectors.yaml")["detectors"]

        if want("D1_step_cusum"):
            up1, dn1 = st.cusum(z, k=float(cfg["D1_step_cusum"]["k_sigma"]))
            out["D1_step_cusum"] = np.maximum(up1, dn1)
        for W in cfg["D1_window_shift"]["windows"]:
            if want(f"D1_window_shift_{W}"):
                out[f"D1_window_shift_{W}"] = st.window_mean(z, int(W))
        if want("D2_drift_cusum"):
            up2, dn2 = st.cusum(z, k=float(cfg["D2_drift_cusum"]["k_sigma"]))
            out["D2_drift_cusum"] = np.maximum(up2, dn2)
        if want("D2_ewma"):
            out["D2_ewma"] = st.ewma(z, lam=float(cfg["D2_ewma"]["lam"]))
        for W in cfg["D2_slope"]["windows"]:
            if want(f"D2_slope_{W}"):
                out[f"D2_slope_{W}"] = st.window_slope(z, int(W))
        if want("D3_block_contrast"):
            Lb = int(config.load("detectors.yaml")["detectors"]["D3_replay"]["lengths"][0])
            out["D3_block_contrast"] = st.block_contrast(z, Lb)
        if want("D4_spike"):
            out["D4_spike"] = np.abs(z)

        # --- loop-based statistics (1-D only): replay and fingerprint
        L = int(cfg["D3_replay"]["lengths"][0])
        gap = int(cfg["D3_replay"]["min_source_gap"])
        ncc_kinds = cfg["D3_replay"].get("ncc_kinds")
        ncc_ok = (ncc_kinds is None) or (c.kind in ncc_kinds)
        ncc = np.full_like(z, np.nan)
        ex = np.zeros_like(z)
        fp = np.full_like(z, np.nan)
        raw_fp_store = np.full_like(z, np.nan)
        exact_lengths = cfg["D3_replay"].get("exact_lengths") or {}
        L_exact = int(exact_lengths.get(c.kind, L))
        need_replay = (want("D3_replay_ncc") and ncc_ok) or want("D3_replay_exact")
        need_fp = want("D6_fingerprint") and self.fingerprint is not None and self.fingerprint.inv_cov is not None
        if need_replay or need_fp:
            for r in range(R):
                if want("D3_replay_ncc") and ncc_ok:
                    ncc[:, r] = st.replay_max_ncc(anom[:, r], L=L, min_source_gap=gap)
                if want("D3_replay_exact"):
                    ex[:, r] = st.replay_exact(raw[:, r], L=L_exact, min_source_gap=gap)
                if need_fp:
                    raw_fp = self.fingerprint.distance_series(z[:, r], y[:, r], x[:, r])
                    fp[:, r] = self.fingerprint.relative_distance(raw_fp)
                    raw_fp_store[:, r] = raw_fp
        if want("D3_replay_ncc") and ncc_ok:
            out["D3_replay_ncc"] = ncc
        if want("D3_replay_exact"):
            out["D3_replay_exact"] = ex
        if need_fp:
            out["D6_fingerprint"] = fp
            out["_D6_fingerprint_raw"] = raw_fp_store
        out["_z_valid"] = np.isfinite(z)      # internal: a missing day cannot alarm
        return out

    def surrogate_statistics(self, R: int, rng: np.random.Generator,
                             names: list[str] | None = None) -> dict[str, np.ndarray]:
        """Statistics on ``R`` phase-randomised surrogates of the calibration slice.

        The surrogate keeps the amplitude spectrum of the raw sensor series (and
        of the residual, for z) but contains no repeated subsequence, which is
        the correct null for the replay detector.
        """
        c = self.conditioned
        idx_cal = np.flatnonzero(self._cal)
        raw_cal = c.y[idx_cal]
        anom_cal = self.anomaly()[idx_cal]
        resid_cal = self.residual[idx_cal]
        raw_s, _ = surrogate_replicates(raw_cal, R, rng)
        anom_s, _ = surrogate_replicates(anom_cal, R, rng)
        resid_s, _ = surrogate_replicates(np.nan_to_num(resid_cal), R, rng)
        # whitening is a per-day linear filter plus a deterministic seasonal
        # scale, so it is applied column by column with the already fitted SP3
        # parameters (no refitting on surrogate data).
        # (z is not needed by the replay detectors, which work on the raw series;
        # the surrogates of the raw road are what the matched filter needs.)
        out: dict[str, np.ndarray] = {}
        want = (lambda k: True) if names is None else (lambda k: k in set(names))
        d3 = config.load("detectors.yaml")["detectors"]["D3_replay"]
        L = int(d3["lengths"][0])
        gap = int(d3["min_source_gap"])
        ncc_kinds = d3.get("ncc_kinds")
        ncc_ok = (ncc_kinds is None) or (c.kind in ncc_kinds)
        ncc = np.full_like(raw_s, np.nan)
        ex = np.zeros_like(raw_s)
        for r in range(R):
            if want("D3_replay_ncc"):
                ncc[:, r] = st.replay_max_ncc(anom_s[:, r], L=L, min_source_gap=gap)
            if want("D3_replay_exact"):
                ex[:, r] = st.replay_exact(raw_s[:, r], L=L, min_source_gap=gap)
        if want("D3_replay_ncc") and ncc_ok:
            out["D3_replay_ncc"] = ncc
        if want("D3_replay_exact"):
            out["D3_replay_exact"] = ex
        return out

    def _conditioned_slice(self, idx: np.ndarray):
        """A Conditioned view restricted to ``idx`` (needed by SP3.standardise)."""
        import copy

        c = copy.copy(self.conditioned)
        c.dates = c.dates[idx]
        c.y = c.y[idx]
        c.x = c.x[idx]
        c.valid = c.valid[idx]
        c.t = c.t[idx]
        return c

    # --------------------------------------------------------------- alarms
    def alarms(self, idx: np.ndarray | None = None, stats: dict | None = None) -> dict:
        """Threshold exceedances, alarm onsets (events) and the combined alarm."""
        S = self.statistics(idx) if stats is None else stats
        valid = S.get("_z_valid")
        T = next(v for k, v in S.items() if not k.startswith("_")).shape[0]
        per = {}
        any_exceed = np.zeros((T, 1), dtype=bool)
        for name, s in S.items():
            if name.startswith("_"):
                continue
            h = self.thresholds.get(name, np.inf)
            with np.errstate(invalid="ignore"):
                e = np.isfinite(s) & (s > h)
            if valid is not None:
                e = e & valid
            per[name] = e
            any_exceed |= e
        if next(v for k, v in S.items() if not k.startswith("_")).shape[1] == 1:
            S = {k: v[:, 0] for k, v in S.items()}
            per = {k: v[:, 0] for k, v in per.items()}
            any_exceed = any_exceed[:, 0]
        return {"stats": S, "exceed": per, "exceed_any": any_exceed}

    # ---------------------------------------------------------------- utils
    def availability_events(self, cooldown: int = 14) -> np.ndarray:
        """D7 alarm onsets: reporting gaps longer than the calibration envelope.

        Deliberately *not* part of ``alarms()``: the statistical false-alarm budget
        is a statement about a detector suite on clean data, while an outage is a
        fact about the station.  Merging the two would let a station with a broken
        modem consume the alarm budget of the statistics.
        """
        if self.availability is None:
            return np.zeros(len(self.conditioned.y), dtype=bool)
        return self.availability.events(self.conditioned.gap, cooldown)

    def signal_frame(self) -> pd.DataFrame:
        """Convenience frame used by the dashboard and the figures."""
        c = self.conditioned
        return pd.DataFrame(
            {
                "date": pd.DatetimeIndex(c.dates),
                "sensor": c.y,
                "reference": c.x,
                "expected": self.expectation.predict(c),
                "residual": self.residual,
                "z": self.z,
            }
        )

    def describe(self) -> dict:
        return {
            "site": self.site,
            "variable": self.variable,
            "budget_per_year": self.budget,
            "sigma_residual": self.sigma_residual,
            "sigma_z": self.sigma_z,
            "seasonal_range": self.seasonal_range,
            "sigma_over_range": self.sigma_residual / self.seasonal_range if self.seasonal_range else float("nan"),
            "expectation": self.expectation.describe(),
            "whitener": self.whitener.describe(),
            "fingerprint": self.fingerprint.describe() if self.fingerprint else None,
            "thresholds": {k: float(v) for k, v in self.thresholds.items()},
            **self.meta,
        }

    def save(self) -> None:
        """Write the model description (thresholds, orders, scales) to results/cache."""
        d = config.path("results", "cache", "models")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{self.site}_{self.variable}.json").write_text(
            json.dumps(self.describe(), indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# building
# --------------------------------------------------------------------------- #


def _cal_indices(model_dates: np.ndarray) -> np.ndarray:
    start, end = config.period("calibration")
    d = pd.DatetimeIndex(model_dates)
    return np.asarray((d >= pd.Timestamp(start)) & (d <= pd.Timestamp(end)))


def build_model(
    site: str,
    variable: str,
    budget: float | None = None,
    rng: np.random.Generator | None = None,
    expectation_kind: str = "harmonic",
    use_whitening: bool = True,
    use_reference: bool = True,
    use_fingerprint: bool = True,
    bundle: SeriesBundle | None = None,
    calibrate: bool = True,
    bias_window: int | None = None,
) -> SiteVariableModel:
    """Fit the SP chain and calibrate the detector suite on the calibration period.

    The ``use_*`` flags exist for the ablation only; the default path is the
    full chain.
    """
    cfg = config.load("detectors.yaml")
    budget = float(cfg["calibration"]["operating_budget"]) if budget is None else float(budget)
    rng = np.random.default_rng(config.SEED) if rng is None else rng

    b = bundle if bundle is not None else load_pair(site, variable)
    c = Conditioning().transform(b)
    cal = _cal_indices(c.dates)
    c = Conditioning().fit_trim(c)

    model = fit_expectation(c, mask=cal, kind=expectation_kind)
    if not use_reference:
        model = fit_expectation(c, mask=cal, kind="no_reference")
    clim = fit_climatology(c, mask=cal)
    resid = model.residual(c)

    rg_kinds = cfg["sp"].get("rank_gauss_kinds", [])
    if use_whitening:
        bw = int(cfg["sp"].get("bias_window_days", 365)) if bias_window is None else int(bias_window)
        w = Whitener(rank_gauss=bool(c.kind in rg_kinds),
                     scale_guard=bool(cfg["sp"].get("scale_guard", False)),
                     guard_bounds=tuple(cfg["sp"].get("scale_guard_bounds", (0.75, 1.5))),
                     bias_window=bw,
                     bias_margin=int(cfg["sp"].get("bias_margin_days", 30)),
                     gap_reset_days=int(cfg["sp"].get("gap_reset_days", 7)),
                     ).fit(c, resid, mask=cal)
        _, z = w.transform(c, resid)
    else:
        # ablation: no AR filter, no seasonal scale; z is the residual divided by
        # its single global robust scale
        w = Whitener(p=0, phi=np.zeros(0), seasonal_coef=np.array([0.0, 0, 0, 0, 0]),
                     rank_gauss=False)          # the ablation removes SP3 entirely
        sig = robust_scale(resid[cal])
        w.seasonal_coef = np.array([np.log(max(sig, 1e-9)), 0.0, 0.0, 0.0, 0.0])
        z = w.standardise(c, resid)
        w.sigma_z = robust_scale(z[np.isfinite(z)])

    m = SiteVariableModel(
        site=b.site_id, variable=variable, budget=budget, conditioned=c,
        expectation=model, whitener=w, fingerprint=None, residual=resid, z=z,
        thresholds={}, calibration={}, sigma_residual=float(robust_scale(resid[cal])),
        seasonal_range=float(clim.seasonal_range(c)), sigma_z=float(w.sigma_z),
        meta={"site_name": b.site_name, "lag_days": int(b.lag_days), "climate": b.meta.get("climate")},
        climatology=clim,
    )
    m._cal = cal

    # D4 stuck-at envelope: 1.5 x the longest run of identical quantised values
    # observed in the whole calibration period (a rule with a data-derived
    # envelope, not a bootstrap threshold -- spec section 8, row D4).
    q = np.round(c.y[cal] / 0.1)
    longest, run = 0, 1
    for i in range(1, len(q)):
        if np.isfinite(q[i]) and np.isfinite(q[i - 1]) and q[i] == q[i - 1]:
            run += 1
        else:
            run = 1
        longest = max(longest, run)
    m.stuck_threshold = float(1.5 * max(longest, 2))

    # D7 availability envelope: 1.5 x the longest gap seen in the calibration
    # period, so a station that routinely misses a week is not accused of an
    # outage for missing a week (see detect/availability.py)
    from .availability import AvailabilityMonitor

    m.availability = AvailabilityMonitor.fit(c.gap, cal)

    if use_fingerprint:
        fp = FingerprintModel(
            window=int(cfg["detectors"]["D6_fingerprint"]["window_days"]),
            min_window=int(cfg["detectors"]["D6_fingerprint"]["min_window_days"]),
        )
        fp.fit(np.where(cal, z, np.nan), c.y, c.x)
        m.fingerprint = fp

    if calibrate:
        names = list(DETECTORS.keys())
        if not use_fingerprint:
            names = [n for n in names if n != "D6_fingerprint"]
        ncc_kinds = cfg["detectors"]["D3_replay"].get("ncc_kinds")
        if ncc_kinds is not None and c.kind not in ncc_kinds:
            names = [n for n in names if n != "D3_replay_ncc"]
        calib = _calibrate_impl(m, names, rng)
        m.calibration = calib
        m.thresholds = {n: float(calib[n][budget][0]) for n in names if n in calib}
    return m


def _calibrate_impl(m: "SiteVariableModel", names: list[str], rng: np.random.Generator) -> dict:
    """Calibrate every detector of the suite on the calibration period only.

    Detectors are grouped by bootstrap replicate count so that each statistic is
    computed once per group (the loop-based statistics dominate the cost).
    """
    cfg = config.load("detectors.yaml")["calibration"]
    # A statistic that the model cannot produce (no fingerprint matrix, replay NCC
    # undefined for this variable kind) must be dropped, not requested: asking for
    # it raised KeyError deep inside the bootstrap loop on the two sparsest pairs.
    if "D6_fingerprint" in names and (m.fingerprint is None
                                      or getattr(m.fingerprint, "inv_cov", None) is None):
        names = [n for n in names if n != "D6_fingerprint"]
    if not names:
        return {}
    block = int(cfg["block_length_days"])
    cooldown = int(cfg["cooldown_days"])
    budgets = [float(b) for b in cfg["budgets_per_year"]]
    idx_cal = np.flatnonzero(m._cal)
    n_cal = len(idx_cal)
    k = len(names)                      # Bonferroni split across the suite

    groups: dict[tuple[str, int], list[str]] = {}
    for name in names:
        mode = "surrogate" if name in SURROGATE_CALIBRATED else "block"
        groups.setdefault((mode, int(BOOT_N.get(name, cfg["n_bootstrap"]))), []).append(name)

    out: dict[str, dict] = {}
    for (mode, R), group in groups.items():
        if mode == "block":
            boot = block_bootstrap_indices(n_cal, R, block, rng)     # (R, n_cal)
            full = idx_cal[boot]                                     # positions in full arrays
            stats = m.statistics(full, names=group)
        else:
            stats = m.surrogate_statistics(R, rng, names=group)
        for name in group:
            if name not in stats:
                continue
            arr = stats[name]
            res = {}
            for budget in budgets:
                per_det = budget / k
                h, achieved = calibrate_threshold(
                    arr, per_det, n_cal, cooldown,
                    bisection_steps=int(cfg["bisection_steps"]))
                res[budget] = (float(h), float(achieved), float(per_det))
            out[name] = res
    return out
