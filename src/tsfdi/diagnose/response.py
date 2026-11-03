"""L5 response: the action ladder and the alert object.

Ladder (specification section 11), with the counting rule made explicit:

    OK                 no alarm in the last ``quarantine_window_days``
    WATCH              1 alarm event
    QUARANTINE         2-3 events: the suspect readings are excluded and replaced
                       by the reference expectation (the reference is the only
                       independent road available)
    SUSPEND            4+ events: the operator is asked to suspend the sensor
    SECURITY_INCIDENT  a provenance violation, or a confirmed attack -- this is
                       the only state that escalates to the security team rather
                       than to the data team

The alert is a JSON object (required fields in the specification), and the
explanation text is templated from the evidence.  No language model is used: an
explanation that a jury, an auditor or a court can re-derive from the numbers is
worth more than a fluent paragraph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from .evidence import Evidence
from .rules import Diagnosis

LEVELS = ["OK", "WATCH", "QUARANTINE", "SUSPEND", "SECURITY_INCIDENT"]


@dataclass
class Action:
    level: str
    reason: str
    quarantine: bool = False

    def as_dict(self) -> dict:
        return {"level": self.level, "reason": self.reason, "quarantine": self.quarantine}


def choose_action(n_events_recent: int, diagnosis: Diagnosis, provenance_violation: bool) -> Action:
    cfg = config.load("decisions.yaml")["response"]
    if provenance_violation:
        return Action("SECURITY_INCIDENT",
                      "L0 rejected at least one record: the stream is not authentic", quarantine=True)
    if diagnosis.label == "attack" and diagnosis.band == "high":
        return Action("SECURITY_INCIDENT",
                      f"high-confidence attack diagnosis: {diagnosis.rationale[0]}", quarantine=True)
    if getattr(diagnosis, "code", "") == "persistent_shift":
        return Action("SUSPEND",
                      "persistent level shift with no attack evidence: the sensor should be "
                      "inspected and re-baselined before its readings are used again",
                      quarantine=True)
    if n_events_recent >= int(cfg["suspend_after_alarms"]):
        return Action("SUSPEND", f"{n_events_recent} alarm events in the last "
                                 f"{cfg['quarantine_window_days']} days: sensor should be inspected", quarantine=True)
    if n_events_recent >= int(cfg["quarantine_after_alarms"]):
        return Action("QUARANTINE", f"{n_events_recent} alarm events: readings excluded, "
                                    f"reference used as fallback", quarantine=True)
    if n_events_recent >= int(cfg["watch_after_alarms"]):
        return Action("WATCH", "an alarm event was recorded; monitoring continues")
    return Action("OK", "no alarm in the recent window")


def explanation_text(ev: Evidence, diag: Diagnosis, action: Action, impact: dict | None) -> str:
    """Templated, evidence-citing explanation (no language model)."""
    parts = [
        f"{ev.site} / {ev.variable}: alarm on day index {ev.t_alarm}.",
        f"Detectors that fired: {', '.join(ev.detectors_fired) if ev.detectors_fired else 'none'}.",
        f"Onset shape (SP4): {ev.onset_shape}; worst |z| in the window: {ev.z_scale_at_alarm:.2f} sigma." if np.isfinite(ev.z_scale_at_alarm) else
        f"Onset shape (SP4): {ev.onset_shape}.",
    ]
    if np.isfinite(ev.mean_z_in_window):
        parts.append(f"Mean residual over the window: {ev.mean_z_in_window:+.2f} sigma.")
    if diag.rationale:
        parts.append("Diagnosis: " + "; ".join(diag.rationale) + ".")
    parts.append(f"Classification: {diag.label} (confidence {diag.confidence:.2f}, {diag.band}).")
    if impact:
        parts.append(f"Estimated effect: {impact.get('flips_total', 0)} day(s) on which a documented "
                     f"decision rule returns a different answer "
                     f"({', '.join(f'{k}={v}' for k, v in (impact.get('per_rule') or {}).items() if v)}).")
    parts.append(f"Recommended action: {action.level} -- {action.reason}.")
    return " ".join(parts)


def alert_object(ev: Evidence, diag: Diagnosis, action: Action, impact: dict | None,
                 timestamp: str | None = None, provenance: dict | None = None) -> dict:
    """The required alert JSON (specification section 11)."""
    return {
        "site": ev.site,
        "variable": ev.variable,
        "t_alarm": int(ev.t_alarm),
        "timestamp": timestamp,
        "detectors_fired": list(ev.detectors_fired),
        "class_probs": diag.probs or {diag.label: 1.0},
        "confidence": diag.confidence,
        "confidence_band": diag.band,
        "evidence": {
            "onset_shape": ev.onset_shape,
            "z_scale": ev.z_scale_at_alarm,
            "mean_z": ev.mean_z_in_window,
            "run_length_days": ev.run_length_days,
            "n_variables_affected": ev.n_variables_affected,
            "reference_extreme": ev.reference_extreme,
            "fingerprint_distance": ev.fingerprint_distance,
            "fingerprint_too_clean": ev.fingerprint_too_clean,
            "fault_envelope_exceeded": ev.fault_envelope_exceeded,
            "replay_exact_hit": ev.replay_exact_hit,
            "block_contrast": ev.block_contrast,
        },
        "provenance": provenance or {"ok": not ev.provenance_violation},
        "impact": impact or {},
        "action": action.level,
        "action_detail": action.reason,
        "explanation_text": explanation_text(ev, diag, action, impact),
    }
