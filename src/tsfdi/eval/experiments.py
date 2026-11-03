"""Orchestration of every experiment reported in the paper.

Each step writes a *raw* CSV (one row per injection, or per evaluation unit) to
``results/raw/``.  The tables and figures are then built from the raw files, never
from hard-coded numbers, so ``make experiments && make report-figures`` reproduces
everything after ``make verify`` checks it.

Site subsets.  The full magnitude grid is infeasible in the time available on the
target hardware (34 site-variable pairs x 5 attacks x 6 magnitudes x 30 injections
= 30 600 streaming evaluations at ~0.5 s each), so the protocol is:

* the **primary** variable TMAX is evaluated with 30 injections per cell on eight
  sites spanning six climate classes (the specification's 30-injection default);
* the secondary variables TMIN and PRCP use 10 injections per cell on the same or
  a slightly smaller site set, and the reduction is reported in the table caption.

This is a documented reduction in *coverage*, not in protocol: nothing is tuned
on the reduced grid, and the cells that are run keep the full 30-replicate
default.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..attacks.adaptive import a1_reference_aware, a2_blackbox, a2_greedy
from .ablation import run_variant
from .baselines import run_baselines
from .metrics import detection_summary
from .runner import (evaluate_injection, evaluable_pairs, false_alarm_runs,
                     generalization_grid, get_model, model_bundle, standard_profile_runs)

RAW = "raw"

# Eight sites, six climate classes, both challenge-relevant regions and the
# European/Nordic contrast; MIAMI_INTL and NAVACERRADA are held out from tuning.
GRID_SITES_TMAX = ["CARTHAGE", "JENDOUBA", "GAFSA", "MERSA_MATRUH", "NAVACERRADA",
                   "FELDBERG", "HELSINKI_KAISANIEMI", "MIAMI_INTL"]
GRID_SITES_TMIN = ["CARTHAGE", "GAFSA", "MERSA_MATRUH", "NAVACERRADA", "FELDBERG", "HELSINKI_KAISANIEMI"]
GRID_SITES_PRCP = ["CARTHAGE", "JENDOUBA", "NAVACERRADA", "FELDBERG", "CABAUW", "MIAMI_INTL"]
ABLATION_SITES = ["CARTHAGE", "GAFSA", "FELDBERG", "HELSINKI_KAISANIEMI"]


def _raw_path(name: str) -> Path:
    d = config.results_dir() / RAW
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def _save(df: pd.DataFrame, name: str) -> None:
    p = _raw_path(name)
    df.to_csv(p, index=False)
    print(f"[experiments] wrote {p.relative_to(config.repo_root())} ({len(df)} rows)", flush=True)


# --------------------------------------------------------------------------- #
# individual experiments
# --------------------------------------------------------------------------- #


def exp_standard_profile(quick: bool = False) -> pd.DataFrame:
    n_rep = 2 if quick else 5
    sites = ["CARTHAGE", "GAFSA", "FELDBERG", "MIAMI_INTL"] if quick else None
    df = standard_profile_runs(sites=sites, n_rep=n_rep)
    _save(df, "table1_standard_profile.csv")
    return df


def exp_grid(quick: bool = False) -> pd.DataFrame:
    # The 5 sigma cell is dropped from the default grid: it is detected in 100 % of
    # injections on every pair, so it costs 10 % of the runtime and adds one row of
    # constants.  0.5-3 sigma is where the frontier lives.
    mags = (1.0, 3.0) if quick else (0.5, 1.0, 1.5, 2.0, 3.0)
    n = 2 if quick else None
    frames = []
    t0 = time.time()
    frames.append(generalization_grid(
        sites=GRID_SITES_TMAX[:3] if quick else GRID_SITES_TMAX, variables=("TMAX",),
        magnitudes=mags, n_per_cell=(2 if quick else 30)))
    print(f"[experiments] TMAX grid {time.time() - t0:.0f}s", flush=True)
    frames.append(generalization_grid(
        sites=GRID_SITES_TMIN[:2] if quick else GRID_SITES_TMIN, variables=("TMIN",),
        magnitudes=mags, n_per_cell=(2 if quick else 10)))
    frames.append(generalization_grid(
        sites=GRID_SITES_PRCP[:2] if quick else GRID_SITES_PRCP, variables=("PRCP",),
        magnitudes=mags, n_per_cell=(2 if quick else 10)))
    df = pd.concat(frames, ignore_index=True)
    _save(df, "table2_grid_raw.csv")
    return df


def exp_false_alarms(quick: bool = False) -> pd.DataFrame:
    df = false_alarm_runs(sites=None if not quick else ["CARTHAGE", "FELDBERG"])
    _save(df, "table3_false_alarms_raw.csv")
    return df


def exp_baselines(quick: bool = False) -> pd.DataFrame:
    from ..attacks.families import Injector

    rows = []
    sites = ["CARTHAGE", "FELDBERG"] if quick else ["CARTHAGE", "GAFSA", "FELDBERG", "HELSINKI_KAISANIEMI"]
    for site in sites:
        for var in ("TMAX",):
            m = get_model(site, var)
            b = model_bundle(m.site, var)
            test_start = int(np.flatnonzero(b.dates >= np.datetime64(config.period("test")[0]))[0])
            inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, len(b.sensor), test_start=test_start)
            injections = []
            for fam in ("step", "ramp"):
                for mag in ((1.5, 3.0) if quick else (1.0, 1.5, 2.0, 3.0, 5.0)):
                    for rep in range(2 if quick else 5):
                        rng = np.random.default_rng(config.SEED + 17 * rep)
                        spec = inj.make_spec(fam, rng, magnitude_sigma=mag, magnitude_percent=None)
                        res = inj.inject(b.sensor, spec)
                        injections.append({"values": res.values, "labels": res.labels,
                                           "onset": spec.onset, "duration": spec.duration,
                                           "family": fam, "magnitude_sigma": mag})
            rows.append(run_baselines(m, b, injections))
    out = pd.concat(rows, ignore_index=True)
    _save(out, "table_baselines_raw.csv")

    # the proposed suite at a *tighter* operating point, so that the comparison can
    # be made at a matched false-alarm rate rather than at a matched nominal budget
    from .baselines import baseline_false_alarms
    from .runner import evaluate_injection

    matched = []
    for budget_pt in (0.1, 0.5, 1.0):
        for site in sites:
            m = get_model(site, "TMAX")
            m.thresholds = {k: float(m.calibration[k][budget_pt][0])
                            for k in m.thresholds if budget_pt in m.calibration[k]}
            b = model_bundle(m.site, "TMAX")
            test_start = int(np.flatnonzero(b.dates >= np.datetime64(config.period("test")[0]))[0])
            inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, len(b.sensor), test_start=test_start)
            dets, delays = [], []
            for fam in ("step", "ramp"):
                for mag in (1.0, 1.5, 2.0, 3.0):
                    for rep in range(5):
                        rng = np.random.default_rng(config.SEED + 17 * rep)
                        spec = inj.make_spec(fam, rng, magnitude_sigma=mag, magnitude_percent=None)
                        res = inj.inject(b.sensor, spec)
                        out = evaluate_injection(m, b, res.values, res.labels, spec.onset,
                                                 spec.duration, fam, mag, np.nan)
                        dets.append(out.detected)
                        if out.delay is not None:
                            delays.append(out.delay)
            al = m.alarms()
            mask = __import__("tsfdi.detect.statistics", fromlist=["x"]).event_onsets_mask(al["exceed_any"], 14)
            test = ~m._cal
            fa = float((mask & test).sum() / (test.sum() / 365.25))
            matched.append({"level": "PROPOSED", "budget": budget_pt, "site": site,
                            "n": len(dets), "detection_rate": float(np.mean(dets)),
                            "delay_median": float(np.median(delays)) if delays else np.nan,
                            "fa_per_year": fa})
        print(f"[experiments] matched-budget point {budget_pt}: "
              f"{np.mean([r['detection_rate'] for r in matched if r['budget'] == budget_pt]):.2f} detection",
              flush=True)
    _save(pd.DataFrame(matched), "table_baselines_matched_raw.csv")

    fa_rows = []
    for site in sites:
        for var in ("TMAX",):
            fa_rows.extend(baseline_false_alarms(get_model(site, var)))
    fa = pd.DataFrame(fa_rows)
    _save(fa, "table_baselines_fa_raw.csv")
    print(fa.groupby("baseline")["events_per_year"].median().round(2).to_string(), flush=True)
    return out


def exp_ablation(quick: bool = False) -> pd.DataFrame:
    sites = ABLATION_SITES[:2] if quick else ABLATION_SITES
    frames = []
    for name in ("full", "no_whitening", "no_reference", "no_fingerprint", "gbm_expectation"):
        try:
            frames.append(run_variant(name, sites, magnitudes=(1.5, 3.0) if quick else (1.0, 1.5, 2.0, 3.0),
                                      n_per_cell=2 if quick else 6))
            print(f"[experiments] ablation {name} done", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[experiments] ablation {name} failed: {exc}", flush=True)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    _save(df, "table_ablation_raw.csv")
    return df


def exp_evasion(quick: bool = False) -> pd.DataFrame:
    """A1a/A1b/A1c, A2 white-box greedy and A2 black-box, on a fixed site set."""
    rows = []
    sites = ["CARTHAGE", "FELDBERG"] if quick else ["CARTHAGE", "GAFSA", "NAVACERRADA",
                                                    "FELDBERG", "HELSINKI_KAISANIEMI", "MIAMI_INTL"]
    n_test_days = (3653 - 1826)
    for site in sites:
        m = get_model(site, "TMAX")
        for level, fn in (("A1a", lambda: a1_reference_aware(m, "none")),
                          ("A1b", lambda: a1_reference_aware(m, "white")),
                          ("A1c", lambda: a1_reference_aware(m, "ar1")),
                          ("A2", lambda: a2_greedy(m)),
                          ("A2-blackbox", lambda: a2_blackbox(m))):
            t0 = time.time()
            try:
                r = fn()
            except Exception as exc:  # noqa: BLE001
                print(f"[experiments] evasion {site} {level} failed: {exc}", flush=True)
                continue
            # which channel caught it: re-score the modified series with the fitted
            # suite, using the same attribution window as every other table
            from .runner import evaluate_injection

            b_ = model_bundle(m.site, "TMAX")
            idx = np.flatnonzero(np.asarray(r.labels, dtype=bool))
            onset_l = int(idx[0]) if idx.size else len(r.values) // 2
            dur_l = int(idx.size) if idx.size else 240
            try:
                ev_out = evaluate_injection(m, b_, r.values, np.asarray(r.labels, dtype=bool),
                                            onset_l, dur_l, level, float("nan"), float("nan"))
                fired = "|".join(ev_out.detectors_fired)
                ev_delay = ev_out.delay
            except Exception:  # noqa: BLE001
                fired, ev_delay = "", None
            rows.append({
                "site": site, "variable": "TMAX", "level": level,
                "detected": r.detected, "detection_day": r.detection_day,
                "detectors_fired": fired, "delay_from_midpoint": ev_delay,
                "cumulative_bias_sigma": r.cumulative_bias / max(m.sigma_residual, 1e-9),
                "mean_bias_sigma": float(np.mean(np.abs(r.delta[r.labels])) / m.sigma_residual) if r.labels.any() else 0.0,
                "n_days_affected": int(r.labels.sum()),
                "seconds": time.time() - t0,
                **{f"meta_{k}": v for k, v in r.meta.items()},
            })
            print(f"[experiments] evasion {site} {level}: detected={r.detected} "
                  f"bias={rows[-1]['cumulative_bias_sigma']:.1f}sigma ({time.time() - t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "table_evasion_raw.csv")
    return df


def exp_evasion_curve(quick: bool = False) -> pd.DataFrame:
    """Damage against the false-alarm budget (Figure 2; the Urbina-style metric).

    For each budget in ``calibration.budgets_per_year`` the *same* fitted model is
    re-thresholded from its own calibration table (no refitting: the calibration
    dictionary already holds the threshold for every budget), and three attackers
    are run:

    * **A0** — the standard profile (12 % step): the damage is the bias it inserts
      before the first alarm, so it shrinks as the budget loosens;
    * **A2** — black-box bisection: the largest constant bias the suite does not
      detect, i.e. the damage an adaptive attacker can insert *for free*;
    * **A2, L3 removed** — the same attack with the replay and fingerprint channels
      disabled, which is what the "before hardening" curve means.

    Damage is reported in sigma x sensor-days of undetected bias, which is the unit
    the specification asks for and the only unit in which a looser budget can be
    compared with a tighter one.
    """
    sites = ["CARTHAGE", "FELDBERG"] if quick else ["CARTHAGE", "GAFSA", "MIAMI_INTL", "FELDBERG"]
    budgets = [0.1, 0.5, 1.0, 2.0] if not quick else [1.0, 2.0]
    # One injection window per budget makes the curve look non-monotone purely
    # because the *window* changes: the achievable undetected bias depends on which
    # days it covers (a calm spell hides more than a stormy one).  Averaging over
    # four windows x both signs removes that artefact and is what the figure shows.
    n_win = 1 if quick else 4
    rows = []
    for site in sites:
        m = get_model(site, "TMAX")
        sigma = float(m.sigma_residual)
        full = dict(m.thresholds)
        for budget in budgets:
            m.thresholds = {k: float(m.calibration[k][budget][0]) for k in full if budget in m.calibration[k]}
            n = len(m.conditioned.y)
            cal_end = int(np.flatnonzero(~m._cal)[0]) if (~m._cal).any() else n // 2
            span = max(1, n - cal_end - 300)
            starts = [cal_end + int(span * (i + 0.5) / n_win) for i in range(n_win)]
            signs = [1, -1]

            def _bb(model, with_l3: bool = True):
                """Robust summary of the undetected bias over windows and signs.

                A window in which the *clean* series already alarms is not
                attributable at all: the Schmitt cooldown swallows any new onset
                and the bisection runs to its 6 sigma cap.  Such windows are
                counted (``n_blocked``), not averaged in -- otherwise one blocked
                window out of eight dominates the mean and the curve loses its
                meaning.
                """
                keep = dict(model.thresholds)
                if not with_l3:
                    model.thresholds = {k: (np.inf if k.startswith(("D3", "D6")) else v)
                                        for k, v in keep.items()}
                ach, blocked = [], 0
                try:
                    for st0 in starts:
                        for sg in signs:
                            r_ = a2_blackbox(model, start=st0, rng=np.random.default_rng(config.SEED + sg))
                            a_ = float(r_.meta.get("achieved_sigma", 0.0))
                            if a_ >= 5.5 or bool(r_.detected):
                                blocked += 1
                                continue
                            ach.append(a_)
                finally:
                    model.thresholds = keep
                med = float(np.median(ach)) if ach else float("nan")
                return med, med * 240.0, blocked

            # A2: the largest bias the suite does not detect
            achieved, damage, blocked = _bb(m, True)
            rows.append({"site": site, "level": "A2", "budget": budget,
                         "damage_sigma_days": damage, "mean_bias_sigma": achieved,
                         "detected": False, "n_days_affected": 240,
                         "n_windows": len(starts) * len(signs), "n_blocked": blocked})
            # A2 without the L3 hardening
            achieved2, damage2, blocked2 = _bb(m, False)
            rows.append({"site": site, "level": "A2_no_L3", "budget": budget,
                         "damage_sigma_days": damage2, "mean_bias_sigma": achieved2,
                         "detected": False, "n_days_affected": 240,
                         "n_windows": len(starts) * len(signs), "n_blocked": blocked2})
            # A0: the standard profile at this budget
            from ..attacks.families import Injector

            b = model_bundle(m.site, "TMAX")
            test_start = int(np.flatnonzero(b.dates >= np.datetime64(config.period("test")[0]))[0])
            inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, len(b.sensor),
                           test_start=test_start)
            spec = inj.make_spec("step", np.random.default_rng(config.SEED + 5), magnitude_percent=12.0)
            res = inj.inject(b.sensor, spec)
            out = evaluate_injection(m, b, res.values, res.labels, spec.onset, spec.duration,
                                     "step", float(res.meta.get("magnitude_sigma", np.nan)), np.nan)
            mag = float(res.meta.get("magnitude_sigma", np.nan))
            days = spec.duration + (30 if not out.detected else int(out.delay or 0))
            rows.append({"site": site, "level": "A0", "budget": budget,
                         "damage_sigma_days": mag * days, "mean_bias_sigma": mag,
                         "detected": out.detected, "n_days_affected": days})
            print(f"[experiments] evasion curve {site} budget {budget}: "
                  f"A2 {rows[-3]['damage_sigma_days']:.0f}, "
                  f"no-L3 {rows[-2]['damage_sigma_days']:.0f}, "
                  f"A0 {rows[-1]['damage_sigma_days']:.0f} sigma-days", flush=True)
        m.thresholds = full
    df = pd.DataFrame(rows)
    _save(df, "table_evasion_curve_raw.csv")
    return df


def exp_availability(quick: bool = False) -> pd.DataFrame:
    """D7: how often, and how long, the record goes silent (table 7)."""
    from ..detect import statistics as st

    rows = []
    for site, var in evaluable_pairs():
        try:
            m = get_model(site, var)
        except Exception:  # noqa: BLE001
            continue
        ev = m.availability_events()
        test = ~m._cal
        years = float(test.sum() / 365.25)
        runs = m.availability.run_length_at(m.conditioned.gap)
        rows.append({"site": site, "variable": var,
                     "envelope_days": int(m.availability.envelope_days),
                     "events_total": int(ev.sum()),
                     "events_test": int(ev[test].sum()),
                     "events_per_year_test": float(ev[test].sum() / years) if years else np.nan,
                     "missing_days": int(m.conditioned.gap.sum()),
                     "missing_fraction": float(m.conditioned.gap.mean()),
                     "longest_gap_days": int(runs.max()) if len(runs) else 0,
                     "mean_gap_run_days": float(runs[runs > 0].mean()) if (runs > 0).any() else 0.0})
    df = pd.DataFrame(rows)
    _save(df, "table7_availability_raw.csv")
    print(df[["events_per_year_test", "missing_fraction", "longest_gap_days"]].describe().round(2).to_string(),
          flush=True)
    return df


def exp_diagnosis(quick: bool = False) -> pd.DataFrame:
    """Classify the cause of every alarm: confusion matrix over 4 classes."""
    from ..attacks.families import Injector, natural_event_mask
    from ..diagnose.evidence import extract
    from ..diagnose.rules import rule_table
    from ..detect import statistics as st

    rows = []
    sites = ["CARTHAGE", "FELDBERG"] if quick else ["CARTHAGE", "GAFSA", "NAVACERRADA",
                                                    "FELDBERG", "HELSINKI_KAISANIEMI", "MIAMI_INTL"]
    # A spike fault is a large single-sample excursion (a sensor glitch, a
    # communication bit-flip), and the calibrated D4 threshold is 11-12 sigma on
    # real stations because a natural extreme also produces a large single-sample
    # residual; a 5 sigma spike is below it and is missed by construction.  The
    # diagnosis experiment therefore injects 12 sigma, i.e. a spike the suite is
    # supposed to see, so that what is measured is the *diagnosis*, not the
    # detectability (that is Table 2's job).
    cases = [("step", 3.0, "attack"), ("ramp", 3.0, "attack"), ("replay", np.nan, "attack"),
             ("stuck_at", np.nan, "fault"), ("spike", 12.0, "fault"), ("dropout", np.nan, "fault")]
    for site in sites:
        m = get_model(site, "TMAX")
        b = model_bundle(m.site, "TMAX")
        test_start = int(np.flatnonzero(b.dates >= np.datetime64(config.period("test")[0]))[0])
        inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, len(b.sensor), test_start=test_start)
        for family, mag, truth in cases:
            for rep in range(2 if quick else 6):
                rng = np.random.default_rng(config.SEED + 3 * rep + 101)
                spec = inj.make_spec(family, rng, magnitude_sigma=(mag if np.isfinite(mag) else None))
                res = inj.inject(b.sensor, spec)
                import copy

                from ..sp.conditioning import to_working

                mm = copy.copy(m)
                cc = copy.copy(m.conditioned)
                cc.y = to_working(np.asarray(res.values, float), cc.kind)
                cc.gap = ~np.isfinite(np.asarray(res.values, float))
                cc.valid = np.isfinite(cc.y) & np.isfinite(cc.x) & ~cc.gap
                pred = m.expectation.predict(cc)
                r = cc.y - pred
                r[~cc.valid] = np.nan
                _, z = m.whitener.transform(cc, r)
                mm.conditioned, mm.z, mm.residual = cc, z, r
                al = mm.alarms()
                mask = st.event_onsets_mask(al["exceed_any"], 14)
                # diagnose the first alarm *attributable to the injection*
                # (onset .. onset+duration+30), not the first alarm of the record:
                # the earlier version diagnosed a pre-injection false alarm on
                # every run, which is why every row of the confusion matrix looked
                # identical no matter what had been injected.
                deadline = int(spec.onset + spec.duration + 30)
                # diagnose at the first day a detector *exceeds*, not at the first
                # cooldown-collapsed event onset: an injection that lands within 14
                # days of an unrelated alarm would otherwise have its own evidence
                # replaced by that earlier event, and the experiment would measure
                # the cooldown rather than the diagnosis.
                ev_idx = np.flatnonzero(
                    np.asarray(al["exceed_any"], dtype=bool)[int(spec.onset):
                                                            max(deadline, int(spec.onset) + 1)])
                detected = ev_idx.size > 0
                if not detected and family == "dropout":
                    # a dropout leaves no residual to exceed a threshold: the
                    # availability channel is the only thing that can see it
                    av = mm.availability_events()
                    av_idx = np.flatnonzero(av[int(spec.onset): max(deadline, int(spec.onset) + 1)])
                    if av_idx.size:
                        t_av = int(av_idx[0]) + int(spec.onset)
                        ev = extract(mm, t_av, ["D7_availability"], window=14)
                        ev.availability_run_days = int(mm.availability.run_length_at(mm.conditioned.gap)[t_av])
                        ev.availability_envelope_days = int(mm.availability.envelope_days)
                        diag = rule_table(ev)
                        rows.append({"site": site, "family": family, "truth": truth,
                                     "predicted": diag.label, "fired": "D7_availability",
                                     "alarm": True, "detected": True,
                                     "confidence": diag.confidence,
                                     "confidence_band": diag.band, "t_alarm": t_av})
                        continue
                if not detected:
                    rows.append({"site": site, "family": family, "truth": truth,
                                 "predicted": "undetermined", "fired": "", "alarm": False,
                                 "detected": False, "confidence": np.nan, "confidence_band": "n/a",
                                 "t_alarm": -1})
                    continue
                t = int(ev_idx[0]) + int(spec.onset)
                fired = [k for k, v in al["exceed"].items() if v[max(0, t - 7): t + 1].any()]
                ev = extract(mm, t, fired, window=14)
                diag = rule_table(ev)
                rows.append({"site": site, "family": family, "truth": truth,
                             "predicted": diag.label, "fired": "|".join(fired), "alarm": True,
                             "detected": True,
                             "confidence": diag.confidence, "confidence_band": diag.band, "t_alarm": t})
        # natural events: labelled from the reference, never injected
        cfg = config.load("detectors.yaml")["natural"]["heat_wave"]
        nat = natural_event_mask(m.conditioned.x, m.kind, cfg["percentile"], cfg["min_run"])
        al = m.alarms()
        mask = st.event_onsets_mask(al["exceed_any"], 14) & nat
        for t in np.flatnonzero(mask)[: (2 if quick else 6)]:
            fired = [k for k, v in al["exceed"].items() if v[max(0, t - 7): t + 1].any()]
            ev = extract(m, int(t), fired, window=14, reference_extreme=True, natural_mask=nat)
            diag = rule_table(ev)
            rows.append({"site": site, "family": "natural_heatwave", "truth": "natural",
                         "predicted": diag.label, "fired": "|".join(fired), "alarm": True,
                         "confidence": diag.confidence, "confidence_band": diag.band, "t_alarm": int(t)})
        # clean alarms (no injection): truth is "natural or undetermined" by construction
        al = m.alarms()
        mask = st.event_onsets_mask(al["exceed_any"], 14) & ~nat & ~m._cal
        for t in np.flatnonzero(mask)[: (2 if quick else 4)]:
            fired = [k for k, v in al["exceed"].items() if v[max(0, t - 7): t + 1].any()]
            ev = extract(m, int(t), fired, window=14)
            diag = rule_table(ev)
            rows.append({"site": site, "family": "clean_alarm", "truth": "undetermined",
                         "predicted": diag.label, "fired": "|".join(fired), "alarm": True,
                         "confidence": diag.confidence, "confidence_band": diag.band, "t_alarm": int(t)})
    df = pd.DataFrame(rows)
    _save(df, "table4_diagnosis_raw.csv")
    return df


def exp_efficiency(quick: bool = False) -> pd.DataFrame:
    """Latency and memory: streaming per reading, model build, peak RSS."""
    import resource
    import tracemalloc

    from ..pipeline import StreamingPipeline

    rows = []
    sites = ["CARTHAGE", "FELDBERG"] if quick else ["CARTHAGE", "GAFSA", "NAVACERRADA",
                                                    "FELDBERG", "HELSINKI_KAISANIEMI", "MIAMI_INTL"]
    for site in sites:
        m = get_model(site, "TMAX")
        b = model_bundle(m.site, "TMAX")
        tracemalloc.start()
        pipe = StreamingPipeline(m)
        t0 = time.perf_counter()
        out = pipe.run()
        dt = time.perf_counter() - t0
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append({
            "site": site, "variable": "TMAX", "n_readings": len(out.alarms),
            "total_seconds": dt, "ms_per_reading": 1e3 * dt / len(out.alarms),
            "alarm_events": int(out.alarms.sum()),
            "peak_python_mb": peak / 1e6,
            "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            **{k: v for k, v in out.latency_ms.items()},
        })
        print(f"[experiments] efficiency {site}: {rows[-1]['ms_per_reading']:.3f} ms/reading", flush=True)
    df = pd.DataFrame(rows)
    _save(df, "table5_efficiency_raw.csv")
    return df


STEPS = {
    "standard_profile": exp_standard_profile,
    "grid": exp_grid,
    "false_alarms": exp_false_alarms,
    "baselines": exp_baselines,
    "ablation": exp_ablation,
    "evasion": exp_evasion,
    "evasion_curve": exp_evasion_curve,
    "diagnosis": exp_diagnosis,
    "efficiency": exp_efficiency,
    "availability": exp_availability,
}


def run_all(quick: bool = False, sites=None, steps=None) -> int:
    """Run the requested steps and return the number that failed.

    A step that raises is reported and the others still run (a full sweep takes
    tens of minutes and one flaky site should not throw the run away), but the
    failure count is returned and the CLI exits non-zero, so a stale table can
    never be mistaken for a fresh one.  Each invocation is *appended* to
    ``results/experiments_meta.json`` with its own step list: the published
    tables are produced by several commands, and a file that only remembered the
    last one would misdescribe how the results were made.
    """
    t0 = time.time()
    todo = list(steps) if steps else list(STEPS)
    failed: list[str] = []
    for name in todo:
        if name not in STEPS:
            continue
        t1 = time.time()
        print(f"\n=== {name} ===", flush=True)
        try:
            STEPS[name](quick=quick)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[experiments] {name} FAILED: {exc}", flush=True)
            failed.append(name)
        print(f"--- {name} in {time.time() - t1:.0f}s", flush=True)
    record = {"quick": quick, "steps": todo, "failed": failed,
              "seconds": round(time.time() - t0, 1),
              "seed": config.SEED,
              "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    path = config.results_dir() / "experiments_meta.json"
    history = []
    if path.is_file():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            history = old if isinstance(old, list) else [old]
        except Exception:  # noqa: BLE001 - a corrupt log must not stop the run
            history = []
    history.append(record)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nall experiments in {time.time() - t0:.0f}s"
          + (f" ({len(failed)} FAILED: {', '.join(failed)})" if failed else ""))
    return len(failed)
