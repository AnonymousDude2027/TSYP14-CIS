"""Experiment runner: the standard profile, the magnitude grid and the evasion study.

Everything here obeys three rules that the specification makes non-negotiable:

1. thresholds come from the calibration period (2015-2019) only;
2. nothing is tuned on the evaluation period (2020-2024);
3. held-out attack families (scaling, noise-shaped, A2) are never used for
   calibration or diagnosis training, and are reported separately.

The runner works on *cached models* (``results/cache``) so that the reported
numbers and the reproduced numbers come from the same objects.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..attacks.families import Injector, natural_event_mask
from ..detect import statistics as st
from ..detect.engine import SiteVariableModel, build_model
from ..io import SeriesBundle, load_pair
from .metrics import average_precision, detection_summary

CACHE = "models"


# --------------------------------------------------------------------------- #
# model cache
# --------------------------------------------------------------------------- #


def _cache_path(site: str, var: str, tag: str = "") -> Path:
    d = config.path("results", "cache", CACHE)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{site}_{var}{tag}.pkl"


def get_model(site: str, var: str, *, rebuild: bool = False, **kwargs) -> SiteVariableModel:
    tag = "".join(f"_{k}" for k, v in kwargs.items() if v not in (None, True, False)) + \
          "".join(f"_{k}" for k, v in kwargs.items() if v is False)
    p = _cache_path(site, var, tag)
    if p.is_file() and not rebuild:
        with open(p, "rb") as fh:
            return pickle.load(fh)
    t0 = time.perf_counter()
    m = build_model(site, var, **kwargs)
    m.meta["build_seconds"] = time.perf_counter() - t0
    with open(p, "wb") as fh:
        pickle.dump(m, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return m


def model_bundle(site: str, var: str) -> SeriesBundle:
    return load_pair(site, var)


# --------------------------------------------------------------------------- #
# which pairs can be evaluated at all
# --------------------------------------------------------------------------- #

_EXCL: dict[tuple[str, str], str] | None = None


def _load_exclusions() -> dict[tuple[str, str], str]:
    """Pairs that cannot be modelled, with the reason, written once to results/.

    A pair is excluded when the fitted expectation or the whitening cannot be
    identified (e.g. KHARGA precipitation has 13 valid days in eleven years).
    Excluding it *explicitly* is the only honest option: fitting a model to a
    dozen observations would produce numbers that look like results.  The file is
    part of the deliverable so that the coverage of every table is auditable.
    """
    global _EXCL
    if _EXCL is not None:
        return _EXCL
    out: dict[tuple[str, str], str] = {}
    rows = []
    for s in config.sites():
        for v in s.variables:
            try:
                get_model(s.name, v)
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}".split("\n")[0][:300]
                out[(s.name, v)] = msg
                rows.append({"site": s.name, "variable": v, "reason": msg})
                print(f"[pairs] excluded {s.name} {v}: {msg}", flush=True)
    # always written, even when empty: the file is the audit trail for which
    # pairs every table covers
    pd.DataFrame(rows, columns=["site", "variable", "reason"]).to_csv(
        config.results_dir() / "excluded_pairs.csv", index=False)
    _EXCL = out
    return out


def is_evaluable(site: str, variable: str) -> bool:
    return (site, variable) not in _load_exclusions()


def evaluable_pairs(variables: tuple[str, ...] | list[str] | None = None,
                    sites: list[str] | None = None) -> list[tuple[str, str]]:
    """All (site, variable) pairs of the design minus the explicitly excluded ones."""
    excl = _load_exclusions()
    keep = [s.name for s in config.sites() if sites is None or s.name in sites]
    pairs = []
    for s in config.sites():
        if s.name not in keep:
            continue
        for v in s.variables:
            if variables is not None and v not in variables:
                continue
            if (s.name, v) not in excl:
                pairs.append((s.name, v))
    return pairs


# --------------------------------------------------------------------------- #
# core evaluation of one injection
# --------------------------------------------------------------------------- #


@dataclass
class InjectionOutcome:
    site: str
    variable: str
    family: str
    magnitude_sigma: float
    magnitude_units: float
    onset: int
    duration: int
    detected: bool
    delay: int | None
    events: int
    detectors_fired: list[str]
    auprc: float
    n_false_events_before_onset: int
    label: str = "attack"
    extra: dict | None = None

    def as_row(self) -> dict:
        return {
            "site": self.site, "variable": self.variable, "family": self.family,
            "magnitude_sigma": self.magnitude_sigma, "magnitude_units": self.magnitude_units,
            "onset": self.onset, "duration": self.duration, "detected": self.detected,
            "delay": self.delay, "events": self.events,
            "detectors_fired": "|".join(self.detectors_fired), "auprc": self.auprc,
            "false_events_before_onset": self.n_false_events_before_onset,
            "label": self.label,
        }


def evaluate_injection(model: SiteVariableModel, bundle: SeriesBundle, values: np.ndarray,
                       labels: np.ndarray, onset: int, duration: int, family: str,
                       mag_sigma: float, mag_units: float, deadline_extra: int = 30,
                       label: str = "attack") -> InjectionOutcome:
    """Run the calibrated suite on one corrupted series and score it."""
    import copy

    from ..sp.conditioning import to_working

    m = copy.copy(model)
    c = copy.copy(model.conditioned)
    raw = np.asarray(values, dtype=float)
    c.y = to_working(raw, c.kind)
    # the *gap* mask has to follow the corrupted series: a dropout injection
    # removes values, and the availability channel (D7) and the SP3 gap reset both
    # read it.  Leaving the original mask in place made every injected dropout
    # invisible to both.
    c.gap = ~np.isfinite(raw)
    c.valid = np.isfinite(c.y) & np.isfinite(c.x) & ~c.gap
    pred = model.expectation.predict(c)
    r = c.y - pred
    r[~c.valid] = np.nan
    _, z = model.whitener.transform(c, r)
    m.conditioned = c
    m.z = z
    m.residual = r
    al = m.alarms()
    exceed_any = al["exceed_any"]
    mask = st.event_onsets_mask(exceed_any, 14)
    events = np.flatnonzero(mask)

    deadline = int(duration + deadline_extra)
    hits = events[(events >= onset) & (events <= onset + deadline)]
    before = events[events < onset]
    detected = len(hits) > 0
    delay = int(hits[0] - onset) if detected else None

    # which detectors fired inside the attack window
    fired = []
    for name, e in al["exceed"].items():
        w = e[onset: min(len(e), onset + max(duration, 7))]
        if np.any(w):
            fired.append(name)

    # AUPRC over days (no point adjustment)
    score = _max_standardised(al["stats"], model.thresholds)
    auprc = average_precision(labels, score)

    return InjectionOutcome(
        site=model.site, variable=model.variable, family=family,
        magnitude_sigma=float(mag_sigma), magnitude_units=float(mag_units),
        onset=int(onset), duration=int(duration), detected=bool(detected), delay=delay,
        events=int(len(events)), detectors_fired=fired, auprc=float(auprc),
        n_false_events_before_onset=int(len(before)), label=label,
    )


def _max_standardised(stats: dict, thresholds: dict) -> np.ndarray:
    """Combine detector statistics into one score: max over |S / h|.

    Dividing by the calibrated threshold puts every detector on the same scale,
    so the combination is a *calibrated* score and the resulting AUPRC is not
    dominated by whichever statistic happens to have the largest numeric range.
    """
    cand = []
    for name, s in stats.items():
        if name.startswith("_"):
            continue
        h = thresholds.get(name, np.nan)
        if np.isfinite(h) and h > 0:
            cand.append(np.asarray(s, dtype=float) / h)
    if not cand:
        return np.full_like(np.asarray(list(stats.values())[0], dtype=float), np.nan)
    M = np.vstack(cand)
    with np.errstate(invalid="ignore"):
        return np.nanmax(np.where(np.isfinite(M), M, -np.inf), axis=0)


# --------------------------------------------------------------------------- #
# standard profile
# --------------------------------------------------------------------------- #


def standard_profile_runs(sites: list[str] | None = None, variables: tuple[str, ...] = ("TMAX", "TMIN", "PRCP"),
                          n_rep: int = 5, seed_offset: int = 0) -> pd.DataFrame:
    """Reproduce the organizers' profile: 12 % step, 15 % ramp, 7-sample replay."""
    prof = config.load("standard_profile.yaml")["profile"]
    rows = []
    # iterate the *design* pairs (a site only carries the variables its config
    # lists) minus the explicitly excluded ones -- never "every variable"
    for site, var in evaluable_pairs(variables=variables, sites=sites):
        if True:
            try:
                m = get_model(site, var)
                b = model_bundle(m.site, var)
            except Exception as exc:  # noqa: BLE001
                # the pair is in results/excluded_pairs.csv with the same reason;
                # an "ERROR" row in a results table is not a result
                print(f"[pairs] skipping {site} {var}: {exc}", flush=True)
                continue
            n = len(b.sensor)
            test_start = int(np.flatnonzero(b.dates >= pd.Timestamp(config.period("test")[0]))[0])
            inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, n, test_start=test_start)
            for rep in range(n_rep):
                rng = np.random.default_rng(config.SEED + 1000 * rep + seed_offset)
                for family, pct in (("step", prof["step"]["magnitude_percent"]),
                                    ("ramp", prof["ramp"]["magnitude_percent"])):
                    dur = (prof["step"]["duration_days"] if family == "step"
                           else prof["ramp"]["rise_days"] + prof["ramp"]["hold_days"])
                    spec = inj.make_spec(family, rng, duration=dur, magnitude_percent=pct,
                                         magnitude_sigma=None)
                    res = inj.inject(b.sensor, spec)
                    out = evaluate_injection(m, b, res.values, res.labels, spec.onset, dur,
                                             family, res.meta.get("magnitude_sigma", np.nan),
                                             res.meta.get("magnitude_units", np.nan))
                    row = out.as_row()
                    row.update({"percent": pct, "rep": rep, "magnitude_percent": pct})
                    rows.append(row)
                # replay
                spec = inj.make_spec("replay", rng, length=prof["replay"]["length"])
                res = inj.inject(b.sensor, spec)
                out = evaluate_injection(m, b, res.values, res.labels, spec.onset,
                                         prof["replay"]["length"], "replay",
                                         res.meta.get("magnitude_sigma", np.nan), np.nan)
                row = out.as_row()
                row.update({"percent": np.nan, "rep": rep, "magnitude_percent": np.nan})
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# generalization grid
# --------------------------------------------------------------------------- #


def generalization_grid(sites: list[str] | None = None,
                        variables: tuple[str, ...] = ("TMAX", "TMIN", "PRCP"),
                        attacks: tuple[str, ...] | None = None,
                        magnitudes: tuple[float, ...] | None = None,
                        n_per_cell: int | None = None, seed_offset: int = 0,
                        include_replay_lengths: bool = True) -> pd.DataFrame:
    g = config.load("attacks.yaml")["grid"]
    attacks = tuple(g["attacks"]) if attacks is None else tuple(attacks)
    magnitudes = tuple(g["magnitudes_sigma"]) if magnitudes is None else tuple(magnitudes)
    n_per_cell = int(g["injections_per_cell"]) if n_per_cell is None else int(n_per_cell)
    rows = []
    # iterate the *design* pairs (a site only carries the variables its config
    # lists) minus the explicitly excluded ones -- never "every variable"
    for site, var in evaluable_pairs(variables=variables, sites=sites):
        if True:
            try:
                m = get_model(site, var)
                b = model_bundle(m.site, var)
            except Exception as exc:  # noqa: BLE001
                # the pair is in results/excluded_pairs.csv with the same reason;
                # an "ERROR" row in a results table is not a result
                print(f"[pairs] skipping {site} {var}: {exc}", flush=True)
                continue
            n = len(b.sensor)
            test_start = int(np.flatnonzero(b.dates >= pd.Timestamp(config.period("test")[0]))[0])
            inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, n, test_start=test_start)
            for family in attacks:
                if family == "replay":
                    continue
                for mag in magnitudes:
                    for rep in range(n_per_cell):
                        rng = np.random.default_rng(config.SEED + 7919 * rep + seed_offset)
                        spec = inj.make_spec(family, rng, magnitude_sigma=mag,
                                             magnitude_percent=None)
                        res = inj.inject(b.sensor, spec)
                        out = evaluate_injection(m, b, res.values, res.labels, spec.onset,
                                                 spec.duration, family,
                                                 res.meta.get("magnitude_sigma", mag),
                                                 res.meta.get("magnitude_units", np.nan))
                        row = out.as_row()
                        # the *nominal* grid point, kept separately from the realised
                        # magnitude so that a cell can never be split by rounding
                        row.update({"rep": rep, "magnitude_nominal": float(mag),
                                    "gain": res.meta.get("gain", np.nan)})
                        rows.append(row)
            # replay: its natural axis is the window length, not the bias size
            if include_replay_lengths and "replay" in attacks:
                for L in (3, 5, 7, 14, 21):
                    for rep in range(n_per_cell):
                        rng = np.random.default_rng(config.SEED + 104729 * rep + seed_offset)
                        spec = inj.make_spec("replay", rng, length=L)
                        res = inj.inject(b.sensor, spec)
                        out = evaluate_injection(m, b, res.values, res.labels, spec.onset, L,
                                                 "replay", np.nan, np.nan)
                        row = out.as_row()
                        row.update({"rep": rep, "replay_length": L,
                                    "magnitude_nominal": np.nan})
                        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# false alarms on clean data, including natural extremes
# --------------------------------------------------------------------------- #


def false_alarm_runs(sites: list[str] | None = None,
                     variables: tuple[str, ...] = ("TMAX", "TMIN", "PRCP")) -> pd.DataFrame:
    rows = []
    pairs = [(s, v) for (s, v) in evaluable_pairs(variables=variables, sites=sites)]
    for site, var in pairs:
        if True:
            try:
                m = get_model(site, var)
            except Exception:  # noqa: BLE001
                continue
            al = m.alarms()
            mask = st.event_onsets_mask(al["exceed_any"], 14)
            test = m._cal  # noqa: SLF001
            testmask = ~m._cal
            years = testmask.sum() / 365.25
            # natural extremes from the reference
            nat = np.zeros(len(m.conditioned.y), dtype=bool)
            cfg = config.load("detectors.yaml")["natural"]
            for name, c in cfg.items():
                if c["variable"] != var:
                    continue
                nat |= natural_event_mask(m.conditioned.x, m.kind, c["percentile"], c["min_run"])
            n_events = int(mask[testmask].sum())
            n_in_nat = int((mask & nat & testmask).sum())
            per_det = {k: int(st.event_onsets_mask(v, 14)[testmask].sum())
                       for k, v in al["exceed"].items()}
            rows.append({
                "site": site, "variable": var, "years": float(years),
                "events": n_events, "events_per_year": n_events / years if years else np.nan,
                "events_in_natural": n_in_nat,
                "natural_days": int((nat & testmask).sum()),
                "events_calibration": int(mask[m._cal].sum()),
                **{f"events_{k}": v for k, v in per_det.items()},
            })
    return pd.DataFrame(rows)
