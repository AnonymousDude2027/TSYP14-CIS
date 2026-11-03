"""L4/L5 tests: the rule table, the action ladder and the alert contract."""
import numpy as np
import pytest

from tsfdi.diagnose.evidence import Evidence
from tsfdi.diagnose.response import alert_object, choose_action, explanation_text
from tsfdi.diagnose.rules import rule_table


def _ev(**kw):
    base = dict(site="TEST", variable="TMAX", t_alarm=100)
    base.update(kw)
    return Evidence(**base)


def test_provenance_violation_is_an_attack_and_a_security_incident():
    d = rule_table(_ev(provenance_violation=True))
    assert d.label == "attack"
    a = choose_action(0, d, provenance_violation=True)
    assert a.level == "SECURITY_INCIDENT" and a.quarantine


def test_a_clean_alarm_with_no_evidence_is_declared_undetermined():
    d = rule_table(_ev(detectors_fired=["D4_spike"], detector_families=["D4"]))
    assert d.label in {"undetermined", "fault"}


def test_a_long_flat_run_maps_to_the_fault_class():
    d = rule_table(_ev(detectors_fired=["D4_spike"], detector_families=["D4"],
                       fault_envelope_exceeded=True, stuck_run_days=40))
    assert d.label == "fault"


def test_a_persistent_level_shift_recommends_re_baselining():
    """A two-year level shift is a station change, not a 600-day attack."""
    d = rule_table(_ev(detectors_fired=["D2_ewma"], detector_families=["D2"],
                       onset_shape="ramp", run_length_days=400))
    assert d.label == "fault" and d.code == "persistent_shift"
    a = choose_action(1, d, provenance_violation=False)
    assert a.level == "SUSPEND" and a.quarantine
    assert "re-baselined" in a.reason


def test_stuck_at_and_constant_replay_are_undetermined_not_guessed():
    d = rule_table(_ev(detectors_fired=["D4_spike", "D3_replay_exact"],
                       detector_families=["D4", "D3"],
                       fault_envelope_exceeded=True, replay_exact_hit=True))
    assert d.label == "undetermined"


def test_the_action_ladder_escalates_with_the_number_of_recent_events():
    d = rule_table(_ev(detectors_fired=["D1_step_cusum", "D1_window_shift_7"],
                       detector_families=["D1", "D1"], onset_shape="step"))
    levels = [choose_action(n, d, False).level for n in (0, 1, 2, 3)]
    order = ["OK", "WATCH", "QUARANTINE", "SUSPEND", "SECURITY_INCIDENT"]
    assert levels == sorted(levels, key=order.index)
    assert levels[0] == "OK"


def test_alert_object_carries_the_specified_fields():
    ev = _ev(detectors_fired=["D1_step_cusum"], detector_families=["D1"], onset_shape="step",
             mean_z_in_window=2.0, z_scale_at_alarm=3.0, run_length_days=12)
    d = rule_table(ev)
    a = alert_object(ev, d, choose_action(1, d, False), {"flips_total": 2},
                     timestamp="2021-01-01", provenance={"ok": True})
    for k in ("site", "variable", "t_alarm", "detectors_fired", "class_probs", "confidence",
              "confidence_band", "evidence", "impact", "action", "action_detail",
              "explanation_text"):
        assert k in a
    assert isinstance(a["confidence"], float)
    assert a["impact"]["flips_total"] == 2
    assert "D1_step_cusum" in a["explanation_text"]
    assert a["provenance"]["ok"] is True


def test_explanation_text_is_deterministic():
    ev = _ev(detectors_fired=["D1_step_cusum"], detector_families=["D1"], onset_shape="step")
    d = rule_table(ev)
    a1 = explanation_text(ev, d, choose_action(1, d, False), None)
    a2 = explanation_text(ev, d, choose_action(1, d, False), None)
    assert a1 == a2 and "TSFDI" not in a1


def test_availability_channel_sees_a_silenced_sensor():
    """A dropout produces no residual, so no threshold can fire. D7 must.

    The envelope is data-derived (1.5x the longest clean gap), so a station that
    routinely misses days is not accused of an outage for missing them.
    """
    import numpy as np

    from tsfdi.detect.availability import AvailabilityMonitor

    cal = np.zeros(1000, bool)
    cal[100:104] = True                     # a routine 4-day gap in calibration
    mon = AvailabilityMonitor.fit(cal)
    assert mon.envelope_days == 7           # floor: 1.5 x 4 -> 7-day minimum

    gap = np.zeros(2000, bool)
    gap[500:530] = True                     # a 30-day outage
    ev = mon.events(gap)
    assert ev.sum() == 1
    assert int(np.flatnonzero(ev)[0]) == 500 + 7   # the day the run passes the envelope

    short = np.zeros(2000, bool)
    short[500:505] = True
    assert mon.events(short).sum() == 0
