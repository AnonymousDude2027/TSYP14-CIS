"""The operational pipeline L0 -> L5, one reading at a time.

``StreamingPipeline.run`` feeds a (possibly corrupted) sensor road, one day at a
time, through

    L0  provenance verification (signature, sequence, timestamp, hash chain)
    L1  SP1 conditioning + SP2 expectation + SP3 whitening          -> z_t
    L2  D1, D2, D3 (contrast), D4 detectors on z_t
    L3  D6 fingerprint at stride, D7 availability, SP4 shape features
    L4  diagnosis (rule table, optionally the logistic regression)
    L5  action ladder, decision impact, explanation text

and returns a timeline plus the alert objects.  It is the code path used by
``python -m tsfdi stream`` and by the dashboard, and ``tests/test_pipeline.py``
checks that it reproduces the vectorised evaluation path.

Latency (the metric the brief asks for) is measured here, per reading, and
broken down by layer, so the report can state what the security overhead
actually is instead of asserting that it is small.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config
from .detect import statistics as st
from .detect.online import OnlineSuite
from .diagnose.evidence import extract
from .diagnose.impact import decision_flips
from .diagnose.response import alert_object, choose_action
from .diagnose.rules import rule_table
from .io import SeriesBundle
from .provenance.anchor import SoftwareAnchor
from .provenance.record import GENESIS, from_values, sign_record
from .provenance.verify import ProvenanceVerifier
from .sp.conditioning import to_working
from .sp.expectation import harmonic_design
from .sp.matched import ReplayDetector


@dataclass
class StreamResult:
    site: str
    variable: str
    dates: pd.DatetimeIndex
    z: np.ndarray
    sensor: np.ndarray
    assumed_used: np.ndarray          # what the analyst ends up using after the response
    alarms: np.ndarray                # bool: alarm event onset
    alarm_stats: dict = field(default_factory=dict)
    alerts: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)
    latency_ms: dict = field(default_factory=dict)
    points: list = field(default_factory=list)


class StreamingPipeline:
    def __init__(self, model, anchor: SoftwareAnchor | None = None, provenance: bool = True,
                 diagnoser=None, cooldown: int | None = None):
        self.model = model
        # the same anchor as the signed data bundle, so that the fingerprint quoted
        # in a streaming run is the fingerprint quoted everywhere (a fresh key per
        # run would make "the fingerprint appears in every result file" meaningless)
        self.anchor = anchor or SoftwareAnchor.load_or_create(
            config.path("data", "keys", "anchor.pem"))
        self.provenance = provenance
        self.diagnoser = diagnoser
        self.cfg = config.load("detectors.yaml")["calibration"]
        self.cooldown = int(self.cfg["cooldown_days"]) if cooldown is None else int(cooldown)

    # ------------------------------------------------------------------ run
    def run(self, bundle: SeriesBundle | None = None, *, fallback_on_quarantine: bool = True,
            verbose: bool = False) -> StreamResult:
        m = self.model
        c = m.conditioned
        n = len(c.y)
        y = c.y if bundle is None else np.asarray(bundle.sensor, dtype=float)
        dates = pd.DatetimeIndex(c.dates)

        # --- z for the (possibly corrupted) road ---------------------------
        t0 = time.perf_counter()
        z = _z_for(m, y)
        t_sp3 = (time.perf_counter() - t0) / n

        suite = OnlineSuite.from_config()
        verifier = ProvenanceVerifier(self.anchor.public_key(), skew_seconds=36 * 3600)
        replay = ReplayDetector(length=OnlineSuite.from_config().L)
        thresholds = dict(m.thresholds)

        alarm_mask = np.zeros(n, dtype=bool)
        used = np.array(y, dtype=float)
        alerts: list[dict] = []
        prov_errors: list[dict] = []
        points = []
        trigger = st.SchmittAlarm(self.cooldown)   # the batch path's exact rule
        prev_hash = GENESIS          # L0 chain state: links consecutive sent records
        events_recent: list[int] = []
        latency_l0 = latency_l2 = latency_l3 = 0.0

        # ---- L3 statistics that are expensive to recompute day by day ----
        # Every one of these is *causal*: the value at day t is a function of
        # data up to day t only (window means, a prefix maximum over earlier
        # windows, a hash of earlier windows, a trailing 180-day spectrum).
        # Computing the series once and reading element t is therefore not
        # look-ahead; ``tests/test_pipeline.py::test_the_expensive_statistics_
        # are_causal`` re-derives element t from the prefix and checks equality,
        # which is what licenses this shortcut.  Doing it this way also
        # guarantees that the streaming path and the vectorised evaluation path
        # use bit-identical statistics (no "demo version" of a detector).
        y_work = to_working(y, c.kind) if hasattr(c, "kind") else y
        L3 = _causal_l3_series(m, z, y_work, y, c.x)
        fp_series = L3["D6_fingerprint"]

        seq = 0                     # L0 sequence number: counts *sent* records only
        for t in range(n):
            # ---- L0 ------------------------------------------------------
            s0 = time.perf_counter()
            if np.isfinite(y[t]):
                # The sensor signs the value it reports and chains it onto the
                # previously *sent* record, so a missing day is a cadence gap in
                # the sequence (the verifier's max_gap_seconds), not a chain break.
                rec = from_values(f"{m.site}:{m.variable}", seq, dates[t], m.variable,
                                  float(y[t]), prev_hash=prev_hash)
                sign_record(rec, self.anchor.private_key)
                prev_hash = rec.hash
                seq += 1
                if not verifier.check(rec):
                    prov_errors.append(verifier.errors[-1])
            latency_l0 += time.perf_counter() - s0

            # ---- L2 ------------------------------------------------------
            s2 = time.perf_counter()
            stats = suite.step(z[t])
            stats = {k: v for k, v in stats.items()}
            stats["D6_fingerprint"] = fp_series[t]
            stats["D3_replay_ncc"] = L3["D3_replay_ncc"][t]
            stats["D3_replay_exact"] = L3["D3_replay_exact"][t]
            fired = suite.alarm(stats, thresholds)
            latency_l2 += time.perf_counter() - s2

            # ---- L3: the self-similarity detail is only needed to *explain*
            # an alarm, so the O(t) search runs on the alarm days only.
            s3 = time.perf_counter()
            match = None
            if is_replay_hit(fired):
                match = replay.evaluate(y, t)
            fired_names = [k for k, v in fired.items() if v]
            any_fire = bool(fired_names)
            latency_l3 += time.perf_counter() - s3

            is_event = trigger.step(any_fire, t)
            if is_event:
                events_recent.append(t)
                alarm_mask[t] = True
            points.append({"t": t, "z": float(z[t]) if np.isfinite(z[t]) else None,
                           "fired": fired_names, "event": bool(is_event)})

            # ---- L4 / L5 -------------------------------------------------
            if is_event:
                if match is not None:
                    explain_replay = {"ncc": float(match.ncc),
                                      "lag_days": match.lag_days,
                                      "exact": bool(match.exact),
                                      "n_exact_values": int(match.n_exact_values)}
                else:
                    explain_replay = None
                window = max(0, t - int(config.load("decisions.yaml")["response"]["quarantine_window_days"]))
                recent = [e for e in events_recent if e >= window]
                ev = extract(m, t, fired_names,
                             provenance_violation=bool(prov_errors),
                             window=14)
                diag = rule_table(ev)
                if self.diagnoser is not None:
                    try:
                        p = self.diagnoser.prob_dict(ev.feature_vector())
                        best = max(p, key=p.get)
                        from .diagnose.rules import Diagnosis

                        diag = Diagnosis(best, diag.band, diag.rationale, p)
                    except Exception:  # noqa: BLE001 - fall back to the rule table
                        pass
                action = choose_action(len(recent), diag, bool(prov_errors))
                if action.quarantine and fallback_on_quarantine:
                    # the analyst uses the reference expectation from here on
                    lo, hi = t, min(n, t + self.cooldown)
                    used[lo:hi] = m.expectation.predict(c)[lo:hi]
                alert = alert_object(ev, diag, action, None,
                                     timestamp=str(dates[t]),
                                     provenance={"ok": not prov_errors,
                                                 "violations": prov_errors[-3:]})
                alert["evidence"]["replay_match"] = explain_replay
                alerts.append(alert)

        # ---- impact (oracle and operational) ------------------------------
        if alerts and fallback_on_quarantine:
            from .diagnose.impact import decision_flips as dflip

            window_mask = np.zeros(n, dtype=bool)
            for a in alerts:
                lo = max(0, a["t_alarm"] - 7)
                hi = min(n, a["t_alarm"] + self.cooldown)
                window_mask[lo:hi] = True
            used_by_var = {m.variable: used}
            true_by_var = {m.variable: c.y}
            imp = dflip(true_by_var, used_by_var, dates, window_mask)
            for a in alerts:
                a["impact"] = {"flips_total": int(imp["flips_total"]),
                               "per_rule": imp["per_rule"]}

        return StreamResult(
            site=m.site, variable=m.variable, dates=dates, z=z, sensor=used,
            assumed_used=used, alarms=alarm_mask,
            alarm_stats={"thresholds": thresholds},
            alerts=alerts,
            provenance={"ok": not prov_errors, "violations": prov_errors[:5],
                        "fingerprint": self.anchor.fingerprint()},
            latency_ms={"L0_provenance_per_reading": 1e3 * latency_l0 / n,
                        "L2_detectors_per_reading": 1e3 * latency_l2 / n,
                        "L3_consistency_per_reading": 1e3 * latency_l3 / n,
                        "SP3_whitening_per_reading": 1e3 * t_sp3},
            points=points,
        )


def is_replay_hit(fired: dict) -> bool:
    """True when a fired detector belongs to the replay family (D3)."""
    return any(str(k).startswith("D3_") for k, v in fired.items() if v)


def _causal_l3_series(m, z: np.ndarray, y_work: np.ndarray, y_raw: np.ndarray,
                      x: np.ndarray) -> dict[str, np.ndarray]:
    """The three L3 statistics, computed once on the whole road (see the note above).

    ``y_work`` is the sensor road in SP1's working coordinate (the space in which
    ``z`` and the fingerprint features are defined); ``y_raw`` is the reported
    value, which is what the exact-repetition hash must be computed on, because a
    real sensor repeats the number it reports, not its square root.
    """
    cfg = config.load("detectors.yaml")["detectors"]
    L = int(cfg["D3_replay"]["lengths"][0])
    gap = int(cfg["D3_replay"]["min_source_gap"])
    exact_lengths = cfg["D3_replay"].get("exact_lengths") or {}
    L_exact = int(exact_lengths.get(m.conditioned.kind, L))
    anom = m.anomaly()
    if m.fingerprint is not None and m.fingerprint.inv_cov is not None:
        raw_fp = m.fingerprint.distance_series(z, y_work, x)
        fp = m.fingerprint.relative_distance(raw_fp)     # same statistic as detect.engine
    else:
        fp = np.full(len(z), np.nan)
    return {
        "D3_replay_ncc": st.replay_max_ncc(anom, L=L, min_source_gap=gap),
        "D3_replay_exact": st.replay_exact(y_raw, L=L_exact, min_source_gap=gap),
        "D6_fingerprint": fp,
    }


def _z_for(m, y: np.ndarray) -> np.ndarray:
    """z for an arbitrary raw sensor road (delegates to the model's single SP chain)."""
    return m.z_for(y)
