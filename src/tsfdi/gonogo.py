"""The go/no-go experiment (specification section 6).

Run *before* any detector work.  For every (site, variable) it answers one
question: is there usable structure in the sensor/reference residual, and how
large a bias can this pair resolve?

Outputs ``results/gonogo.csv`` and ``results/gonogo.md`` with, per pair:

* samples, residual mean, robust sigma (MAD x 1.4826);
* residual ACF at lags 1..7 and a Ljung-Box p-value;
* the estimated calendar lag;
* the seasonal amplitude of the residual variance (max/min of the fitted
  seasonal scale) -- if this is 1.0 there is nothing to gain from SP3's
  seasonal scaling;
* Welch PSD band powers of the residual (a degenerate spectrum -- all energy in
  one band -- kills the spectral fingerprint);
* mean magnitude-squared coherence per band between sensor and reference, since
  the fingerprint needs coherence structure to move when the sensor is replaced
  by a copy;
* the bias table (0.5, 1, 2, 3 sigma and the organizers' 12 %) with the
  sigma-equivalent of each;
* a Monte-Carlo estimate of the smallest step the CUSUM detects at a 1-per-year
  false-alarm budget.

Decision rules applied automatically (and printed):
    R1  sigma_resid < 0.25 x seasonal range            -> residual is usable
    R2  PSD not degenerate (no band holds > 0.9 of power) -> SP6 in scope
    R3  mean high-band coherence < 0.98                -> fingerprint can move
    R4  minimum detectable bias <= 3 sigma             -> claims allowed for this pair
If R4 fails, the pair is reported as a documented limitation rather than quietly
dropped.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .detect import statistics as st
from .detect.engine import build_model
from .detect.statistics import event_onsets_mask
from .detect.thresholds import calibrate_threshold
from .sp.fingerprint import BANDS, _band_power, _coherence_fast, _welch_fast
from .sp.expectation import robust_scale


def acf(x: np.ndarray, max_lag: int = 7) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < max_lag + 5:
        return np.full(max_lag, np.nan)
    x = x - x.mean()
    den = float(np.dot(x, x))
    return np.array([float(np.dot(x[:-k], x[k:]) / den) for k in range(1, max_lag + 1)])


def ljung_box(x: np.ndarray, lags: int = 10) -> tuple[float, float]:
    """Ljung-Box statistic and p-value (implemented locally: no statsmodels needed)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 10:
        return float("nan"), float("nan")
    r = acf(x, lags)
    q = n * (n + 2) * float(np.sum(r ** 2 / np.arange(n - 1, n - lags - 1, -1)))
    return float(q), float(stats.chi2.sf(q, lags))


def min_detectable_bias(model, budget: float = 1.0, n_mc: int = 60, target_power: float = 0.5,
                        grid=(0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0)) -> float:
    """Smallest constant bias (in sigma) whose CUSUM is detected 50 % of the time.

    Monte Carlo on the *calibration* z: insert a step at a random onset, run the
    detector with the threshold already calibrated to the false-alarm budget, and
    count the fraction of injections detected within 60 days.  This is the honest
    inverse of the usual "detection rate at a chosen magnitude" table: it reports
    what the pair can resolve rather than what a chosen attack size achieves.
    """
    z = model.z
    cal = model._cal  # noqa: SLF001
    zc = np.where(cal, z, np.nan)
    h = float(model.thresholds.get("D1_step_cusum", 15.0))
    k = float(config.load("detectors.yaml")["detectors"]["D1_step_cusum"]["k_sigma"])
    rng = np.random.default_rng(config.SEED)
    n = len(zc)
    onsets = rng.integers(120, n - 120, size=n_mc)

    def power(b: float) -> float:
        hits = 0
        for o in onsets:
            zz = zc.copy()
            zz[o:] = zz[o:] + b
            up, dn = st.cusum(zz[:, None], k=k)
            s = np.maximum(up[:, 0], dn[:, 0])
            exc = np.zeros(n, dtype=bool)
            exc[o: o + 60] = s[o: o + 60] > h
            if exc.any():
                hits += 1
        return hits / n_mc

    lo = None
    for b in grid:
        if power(b) >= target_power:
            lo = b
            break
    return float(lo) if lo is not None else float("nan")


@dataclass
class GonogoRow:
    site: str
    variable: str
    n_samples: int
    n_calibration: int
    lag_days: int
    residual_mean: float
    sigma_resid: float
    seasonal_range: float
    sigma_over_range: float
    acf1: float
    acf7: float
    ljung_box_p: float
    seasonal_sigma_ratio: float
    psd_low: float
    psd_mid: float
    psd_high: float
    psd_degenerate: bool
    coh_low: float
    coh_mid: float
    coh_high: float
    ar_order: int
    z_sigma: float
    z_acf1: float
    min_detectable_bias_sigma: float
    rule_R1: bool
    rule_R2: bool
    rule_R3: bool
    rule_R4: bool
    verdict: str

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def analyse(site: str, variable: str, budget: float = 1.0, model=None, run_mc: bool = True) -> GonogoRow:
    m = model if model is not None else build_model(site, variable, budget=budget)
    cal = m._cal  # noqa: SLF001
    resid = m.residual[cal]
    resid = resid[np.isfinite(resid)]
    z = m.z[cal]
    z = z[np.isfinite(z)]

    a = acf(resid, 7)
    q, p = ljung_box(m.residual[cal], 10)
    f, psd = _welch_fast(resid, 64)
    bands = {k: _band_power(f, psd, lo, hi) for k, (lo, hi) in BANDS.items()}
    tot = sum(v for v in bands.values() if np.isfinite(v))
    frac = {k: (v / tot if tot > 0 else np.nan) for k, v in bands.items()}
    degenerate = bool(max(frac.values()) > 0.9)

    ok = np.isfinite(m.conditioned.y[cal]) & np.isfinite(m.conditioned.x[cal])
    yv = m.conditioned.y[cal][ok]
    xv = m.conditioned.x[cal][ok]
    _, coh = _coherence_fast(yv, xv, 64)
    coh_bands = {k: _band_power(f, coh, lo, hi) for k, (lo, hi) in BANDS.items()}

    scale = m.whitener.scale(m._conditioned_slice(np.flatnonzero(cal)))  # noqa: SLF001
    seas_ratio = float(np.nanmax(scale) / np.nanmin(scale)) if np.isfinite(scale).any() else float("nan")

    mdb = min_detectable_bias(m, budget) if run_mc else float("nan")

    # R1 has to be stated in the coordinate the residual lives in.  For a
    # multiplicative variable (precipitation) SP1 works on the square root, so the
    # seasonal range of the *working* series is sqrt(max baseline) - sqrt(min
    # baseline); comparing a residual of 0.19 (mm^0.5) with a range of 0.001 mm
    # gives the meaningless ratio of 135 that the previous table reported.
    if m.kind == "multiplicative":
        base = m.climatology.seasonal_baseline(m.conditioned)
        bf = base[np.isfinite(base)]
        range_work = float(np.sqrt(max(bf.max(), 0.0)) - np.sqrt(max(bf.min(), 0.0))) if bf.size else float("nan")
        scale_for_rule = range_work
        sigma_over_range = float(m.sigma_residual / range_work) if range_work else float("nan")
    else:
        scale_for_rule = float(m.seasonal_range)
        sigma_over_range = float(m.sigma_residual / m.seasonal_range) if m.seasonal_range else float("nan")

    r1 = bool(np.isfinite(scale_for_rule) and scale_for_rule > 0 and m.sigma_residual < 0.25 * scale_for_rule)
    r2 = bool(not degenerate)
    r3 = bool(np.isfinite(coh_bands["high"]) and coh_bands["high"] < 0.98)
    r4 = bool(np.isfinite(mdb) and mdb <= 3.0)
    fails = [n for n, v in (("R1", r1), ("R2", r2), ("R3", r3), ("R4", r4)) if not v]
    verdict = "go" if not fails else "go (limited): " + ",".join(fails)

    return GonogoRow(
        site=m.meta.get("site_name", site), variable=variable,
        n_samples=int(np.isfinite(m.conditioned.y).sum()), n_calibration=int(np.isfinite(resid).sum()),
        lag_days=int(m.meta.get("lag_days", 0)),
        residual_mean=float(np.mean(resid)), sigma_resid=float(robust_scale(resid)),
        seasonal_range=float(m.seasonal_range), sigma_over_range=sigma_over_range,
        acf1=float(a[0]), acf7=float(a[-1]), ljung_box_p=float(p),
        seasonal_sigma_ratio=seas_ratio,
        psd_low=float(frac["low"]), psd_mid=float(frac["mid"]), psd_high=float(frac["high"]),
        psd_degenerate=degenerate,
        coh_low=float(coh_bands["low"]), coh_mid=float(coh_bands["mid"]), coh_high=float(coh_bands["high"]),
        ar_order=int(m.whitener.p), z_sigma=float(m.sigma_z), z_acf1=float(m.whitener.acf1),
        min_detectable_bias_sigma=float(mdb),
        rule_R1=r1, rule_R2=r2, rule_R3=r3, rule_R4=r4, verdict=verdict,
    )


def run(sites: list[str] | None = None, variables: tuple[str, ...] = ("TMAX", "TMIN", "PRCP"),
        run_mc: bool = True, out_dir=None) -> pd.DataFrame:
    rows = []
    for s in config.sites():
        if sites and s.name not in sites:
            continue
        for v in s.variables:
            if v not in variables:
                continue
            t0 = time.perf_counter()
            try:
                from .eval.runner import get_model

                m = get_model(s.name, v)
                row = analyse(s.name, v, model=m, run_mc=run_mc)
                rows.append(row.as_dict())
                print(f"[gonogo] {s.name:22s} {v:5s} {row.verdict:24s} "
                      f"sigma/range={row.sigma_over_range:.3f} mdb={row.min_detectable_bias_sigma:.2f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[gonogo] {s.name} {v} FAILED: {exc}", flush=True)
    df = pd.DataFrame(rows)
    d = out_dir or config.results_dir()
    df.to_csv(d / "gonogo.csv", index=False)
    (d / "gonogo.md").write_text(to_markdown(df), encoding="utf-8")
    return df


def to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Go/no-go experiment\n\nNo results.\n"
    lines = ["# Go/no-go experiment", "",
             "Generated by `python -m tsfdi gonogo`.  Decision rules are in "
             "`src/tsfdi/gonogo.py` and section 6 of the specification.", ""]
    cols = ["site", "variable", "n_samples", "n_calibration", "lag_days", "sigma_resid",
            "seasonal_range", "sigma_over_range", "acf1", "ljung_box_p", "ar_order",
            "z_sigma", "z_acf1", "min_detectable_bias_sigma", "verdict"]
    sub = df[[c for c in cols if c in df.columns]].copy()
    for c in sub.columns:
        if sub[c].dtype.kind == "f":
            sub[c] = sub[c].round(4)
    lines.append("## Per (site, variable)")
    lines.append("")
    lines.append(sub.to_markdown(index=False))
    lines.append("")
    lines.append("## Spectral and coherence summary")
    lines.append("")
    cols2 = ["site", "variable", "psd_low", "psd_mid", "psd_high", "coh_low", "coh_mid",
             "coh_high", "seasonal_sigma_ratio", "psd_degenerate"]
    sub2 = df[[c for c in cols2 if c in df.columns]].copy()
    for c in sub2.columns:
        if sub2[c].dtype.kind == "f":
            sub2[c] = sub2[c].round(4)
    lines.append(sub2.to_markdown(index=False))
    lines.append("")
    n_go = int((df["verdict"] == "go").sum())
    lines.append(f"**{n_go}/{len(df)} pairs pass all four rules.**  Pairs that do not are "
                 f"reported as documented limitations and their claims are restricted to "
                 f"the magnitudes their minimum detectable bias allows.")
    return "\n".join(lines) + "\n"
