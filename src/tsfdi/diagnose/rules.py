"""Tier-1 diagnosis: a transparent rule table.

The rules are deliberately boring and readable, because a diagnosis that cannot
be explained in one sentence is not usable by an operator, and because the
Phase-1 report must be able to justify every branch.

Order matters.  The *undetermined* class is not a failure: a stuck-at fault and
a constant replay are identical in the data (both produce a run of identical
values), so an honest system must be able to say "integrity incident, cause
undetermined" instead of guessing.  That is the specification's requirement, and
it is also the only defensible behaviour.

Two confidence numbers are reported, and they are different things:

* ``confidence`` (float in [0, 1]) is the *posterior of the chosen class*: from
  the rule table it is the hand-set prior below (a documented, deliberately
  conservative number, 0.6-0.97), and from Tier 2 it is the logistic-regression
  probability.  This is the number the alert JSON carries (spec 11) and the one
  that gets a reliability diagram (Table 4b).
* ``confidence`` band (low | medium | high) is the three-level ordinal the rule
  table states directly from how many independent lines of evidence agree.

Tier 2 (logistic regression) provides calibrated probabilities on top of the
rule table, which remains the fallback and the reference implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

from .evidence import CLASSES, Evidence


@dataclass
class Diagnosis:
    label: str
    band: str                       # low | medium | high (ordinal evidence strength)
    rationale: list[str]
    probs: dict[str, float] | None = None
    code: str = ""                  # machine-readable sub-type (e.g. "persistent_shift")

    @property
    def confidence(self) -> float:
        """Posterior of the chosen class (0.5 if the table supplied nothing)."""
        if not self.probs:
            return 0.5
        return float(self.probs.get(self.label, 0.5))

    def as_dict(self) -> dict:
        return {"label": self.label, "confidence": self.confidence, "code": self.code,
                "confidence_band": self.band,
                "rationale": list(self.rationale), "probs": self.probs}


def _one_hot(label: str, c: float = 0.8) -> dict[str, float]:
    p = {k: (1 - c) / 3 for k in CLASSES}
    p[label] = c
    return p


def rule_table(ev: Evidence) -> Diagnosis:
    """Map evidence to (class, confidence, explanation)."""
    fams = set(ev.detector_families)
    lines: list[str] = []

    # 1. Provenance violations cannot be produced by an honest sensor.
    if ev.provenance_violation:
        lines.append("provenance violation at L0 (bad signature, reordered or gapped record)")
        return Diagnosis("attack", "high", lines, _one_hot("attack", 0.97))

    # 1b. The reporting cadence itself failed: a run of missing days longer than
    #     the envelope learnt from the clean period.  No statistic can see this --
    #     a silenced sensor produces no residual -- so it is a rule, not a test,
    #     and it is a *fault*: a dead modem and a deliberate suppression are
    #     indistinguishable in the data.
    if ev.availability_envelope_days and ev.availability_run_days > ev.availability_envelope_days:
        lines.append(f"{ev.availability_run_days} consecutive missing days, against a clean-data "
                     f"envelope of {ev.availability_envelope_days}")
        lines.append("the sensor stopped reporting: an outage fault, or a deliberate suppression "
                     "of the record; the two are indistinguishable without operator information")
        return Diagnosis("fault", "high", lines, _one_hot("fault", 0.8), code="data_outage")

    # 2. A run of identical values beyond the clean envelope is a fault *unless*
    #    the values are exact copies of earlier readings, which only a replay can
    #    produce.
    if ev.fault_envelope_exceeded and ev.replay_exact_hit:
        lines.append("run of identical values beyond the clean envelope AND an exact repeat of earlier readings")
        return Diagnosis("undetermined", "medium",
                         lines + ["stuck-at fault and constant replay are indistinguishable here"],
                         _one_hot("undetermined", 0.6))
    if ev.fault_envelope_exceeded:
        lines.append(f"stuck-at run of {ev.stuck_run_days} days exceeds the clean envelope")
        return Diagnosis("fault", "high", lines, _one_hot("fault", 0.85))

    # 3. A reference-confirmed natural extreme with a small residual is a natural event.
    if ev.reference_extreme and abs(ev.mean_z_in_window) < 2.0:
        lines.append("the reference itself shows an extreme event and the residual is small")
        return Diagnosis("natural", "high", lines, _one_hot("natural", 0.9))

    # 4. Too-clean residual: the signature of a reference-copying attacker (A1).
    if ev.fingerprint_too_clean:
        lines.append("residual is abnormally clean (variance ratio or coherence outside the calibration cloud)")
        return Diagnosis("attack", "high", lines, _one_hot("attack", 0.88))

    # 5. Exact repeat: replay.
    if ev.replay_exact_hit:
        lines.append("the last 7 readings repeat an earlier 7-reading window exactly")
        return Diagnosis("attack", "high", lines, _one_hot("attack", 0.93))

    # 6c. One sample far outside the envelope with quiet neighbours is a spike: a
    #     glitch, a bit-flip or a transient connection fault.  This must sit before
    #     the step/ramp rules, because a large single sample also drives the
    #     short-window mean-shift detectors (which is how D1 ends up "firing" on a
    #     spike) and would otherwise be labelled an attack.
    if ev.single_sample_excursion and ({"D4"} & fams):
        lines.append(f"a single sample at {ev.z_scale_at_alarm:.1f} sigma with quiet neighbours "
                     f"and the spike-channel envelope exceeded")
        lines.append("a transient sensor glitch explains this; an injected step would persist")
        return Diagnosis("fault", "medium", lines, _one_hot("fault", 0.75), code="spike_fault")

    # 6. Finite-duration block replacement: contrast high, level returns.
    if ev.block_contrast > 3.0 and ev.run_length_days <= 10:
        lines.append(f"L-day block contrast {ev.block_contrast:.1f} sigma with a short run: a bounded replacement")
        return Diagnosis("attack", "medium", lines, _one_hot("attack", 0.7))

    # 6b. Persistent level shift: the residual stays elevated for months without
    #     any replay, fingerprint, provenance or fault-envelope evidence.  The
    #     honest reading is a change in the station/reference relationship
    #     (instrument change, relocation, calibration drift) -- the specification
    #     lists slow calibration drift as a *benign fault* -- and the useful action
    #     is to ask for a re-baseline rather than to keep re-alarming.
    #     Measured motivation: one bundled station (KHARGA) reports 12 days in all
    #     of 2021 and comes back with a level shift of about +0.4 K that lasts two
    #     years; without this branch the drift detector produces ~6 alarms per year
    #     for that period and every one of them is labelled "undetermined".
    if (ev.run_length_days >= 180 and not ev.replay_exact_hit
            and not ev.fingerprint_too_clean and not ev.provenance_violation):
        lines.append(f"residual has stayed above +1 sigma for {ev.run_length_days} consecutive days "
                     f"with no replay, fingerprint or provenance evidence")
        lines.append("a sustained level shift of the station/reference relationship "
                     "(instrument change, relocation, calibration drift) explains this better "
                     "than any attack of bounded duration")
        return Diagnosis("fault", "medium", lines, _one_hot("fault", 0.7), code="persistent_shift")

    # 7. Shape-based reasoning for sustained biases.
    if ev.onset_shape == "step" and ({"D1"} & fams):
        lines.append("step-like onset detected by D1 (CUSUM and short-window mean shift)")
        conf = "high" if len(fams) >= 2 else "medium"
        return Diagnosis("attack", conf, lines, _one_hot("attack", 0.85 if conf == "high" else 0.7))
    if ev.onset_shape == "ramp" or ({"D2"} & fams):
        lines.append("slow drift consistent with D2 (EWMA / slope test); a slow calibration drift of the "
                     "sensor is equally consistent, so the cause is reported as undetermined")
        conf = "medium" if len(fams) >= 2 else "low"
        return Diagnosis("undetermined", conf, lines, _one_hot("undetermined", 0.6))
    if ev.onset_shape == "spike":
        lines.append("single-sample excursion: spike fault or a transient sensor glitch")
        return Diagnosis("fault", "low", lines, _one_hot("fault", 0.55))

    lines.append("an alarm fired but no shape, provenance or fingerprint evidence discriminates the cause")
    return Diagnosis("undetermined", "low", lines, _one_hot("undetermined", 0.5))
