"""End-to-end tests of the L0-L5 streaming path (``tsfdi.pipeline``).

The pipeline is the code path the demo, the dashboard and ``python -m tsfdi
stream`` all use, so it has to satisfy three things the vectorised evaluation
path already satisfies: it must accept honest data at L0, it must raise exactly
the alarms the evaluation path would raise (otherwise the reported numbers
describe a system nobody runs), and every expensive statistic it reuses must be
causal, i.e. computable from the past alone.
"""
import numpy as np
import pandas as pd
import pytest

import tsfdi.config as C
from tsfdi.attacks.families import Injector
from tsfdi.detect import statistics as st
from tsfdi.pipeline import StreamingPipeline
from tsfdi.provenance.anchor import SoftwareAnchor

ANCHOR = SoftwareAnchor.from_seed(b"\x00" * 31 + b"\x01")


@pytest.fixture(scope="module")
def cooldown() -> int:
    return int(C.load("detectors.yaml")["calibration"]["cooldown_days"])


@pytest.fixture(scope="module")
def clean_run(synth_model, synth_bundle):
    return StreamingPipeline(synth_model, anchor=ANCHOR).run(bundle=synth_bundle)


def _corrupt(bundle, onset: int, size: float, length: int):
    from tsfdi.io import SeriesBundle

    y = np.array(bundle.sensor, dtype=float)
    y[onset:onset + length] += size
    return SeriesBundle(site_id=bundle.site_id, site_name=bundle.site_name,
                        variable=bundle.variable,
                        reference_variable=bundle.reference_variable,
                        units=bundle.units, kind=bundle.kind, dates=bundle.dates,
                        sensor=y, reference=bundle.reference, sensor_ok=np.isfinite(y),
                        reference_ok=bundle.reference_ok)


# --------------------------------------------------------------- L0 honesty
def test_an_honest_stream_passes_provenance(clean_run):
    assert clean_run.provenance["ok"], clean_run.provenance["violations"]
    assert clean_run.provenance["fingerprint"]


# ------------------------------------------- online == vectorised (no drift)
def test_streaming_alarms_equal_the_vectorised_alarms(clean_run, synth_model, cooldown):
    ev = st.event_onsets_mask(np.asarray(synth_model.alarms()["exceed_any"], dtype=bool),
                              cooldown=cooldown)
    batch = set(np.flatnonzero(ev).tolist())
    online = set(np.flatnonzero(clean_run.alarms).tolist())
    assert batch == online, (sorted(batch - online)[:5], sorted(online - batch)[:5])


def test_the_expensive_statistics_are_causal(synth_model, synth_bundle):
    """Element t of each L3 series must follow from data up to t alone."""
    m = synth_model
    z = m.z_for(synth_bundle.sensor)
    anom = m.anomaly()
    L, gap = 7, 2
    full_ncc = st.replay_max_ncc(anom, L=L, min_source_gap=gap)
    full_ex = st.replay_exact(synth_bundle.sensor, L=L, min_source_gap=gap)
    for cut in (40, 120, 400):
        p_ncc = st.replay_max_ncc(anom[: cut + 1], L=L, min_source_gap=gap)[cut]
        p_ex = st.replay_exact(synth_bundle.sensor[: cut + 1], L=L, min_source_gap=gap)[cut]
        if np.isfinite(full_ncc[cut]):
            assert np.isclose(p_ncc, full_ncc[cut], atol=1e-12)
        assert p_ex == full_ex[cut]


# ------------------------------------------------------- injection response
def test_a_five_sigma_step_is_caught_within_ten_days(synth_model, synth_bundle):
    m = synth_model
    onset, length = 900, 45
    bad = _corrupt(synth_bundle, onset, 5.0 * float(m.sigma_residual), length)
    run = StreamingPipeline(m, anchor=ANCHOR).run(bundle=bad)
    hits = np.flatnonzero(run.alarms)
    after = hits[hits >= onset]                 # ignore unrelated white-noise alarms
    assert after.size, "a 5 sigma bias of 45 days produced no alarm"
    first = int(after[0])
    assert first <= onset + 10, (onset, first)
    t_alarms = [a["t_alarm"] for a in run.alerts]
    assert first in t_alarms


def test_quarantine_falls_back_to_the_reference_expectation(synth_model, synth_bundle):
    m = synth_model
    onset = 1200
    bad = _corrupt(synth_bundle, onset, 6.0 * float(m.sigma_residual), 40)
    run = StreamingPipeline(m, anchor=ANCHOR).run(bundle=bad, fallback_on_quarantine=True)
    hits = np.flatnonzero(run.alarms)
    after = hits[hits >= onset]
    assert after.size
    t_alarm = int(after[0])
    t = t_alarm + 1                             # one day after the alarm
    exp = m.expectation.predict(m.conditioned)
    assert np.isclose(run.assumed_used[t], exp[t]), "quarantined day still uses the sensor"
    assert not np.isclose(run.assumed_used[t], float(bad.sensor[t]))
    # before the first alarm the analyst is still using the raw sensor
    assert np.isclose(run.assumed_used[0], float(bad.sensor[0]))


# ------------------------------------------------------------- alert content
def test_alert_carries_every_required_field(synth_model, synth_bundle):
    m = synth_model
    run = StreamingPipeline(m, anchor=ANCHOR).run(
        bundle=_corrupt(synth_bundle, 1500, 6.0 * float(m.sigma_residual), 30))
    assert run.alerts
    a = run.alerts[0]
    for key in ("site", "variable", "t_alarm", "timestamp", "detectors_fired",
                "class_probs", "confidence", "evidence", "provenance", "impact",
                "action", "action_detail", "explanation_text"):
        assert key in a, key
    assert a["action"] in {"LOG", "FLAG", "QUARANTINE", "ESCALATE", "SECURITY_INCIDENT"}
    assert 0.0 <= float(a["confidence"]) <= 1.0
    assert "impact" in a and "flips_total" in a["impact"]


def test_latency_is_measured_per_layer(clean_run):
    lat = clean_run.latency_ms
    for key in ("L0_provenance_per_reading", "L2_detectors_per_reading",
                "SP3_whitening_per_reading"):
        assert key in lat and np.isfinite(lat[key]) and lat[key] >= 0.0
    assert lat["L0_provenance_per_reading"] < 20.0     # ms/reading, sanity bound


def test_a_missing_day_is_a_gap_not_a_chain_break(synth_model, synth_bundle):
    """Dropouts must not look like tampering at L0 (they are an L3 fault)."""
    from tsfdi.io import SeriesBundle

    y = np.array(synth_bundle.sensor, dtype=float)
    y[300:310] = np.nan
    holed = SeriesBundle(site_id=synth_bundle.site_id, site_name=synth_bundle.site_name,
                         variable=synth_bundle.variable,
                         reference_variable=synth_bundle.reference_variable,
                         units=synth_bundle.units, kind=synth_bundle.kind,
                         dates=synth_bundle.dates, sensor=y,
                         reference=synth_bundle.reference, sensor_ok=np.isfinite(y),
                         reference_ok=synth_bundle.reference_ok)
    run = StreamingPipeline(synth_model, anchor=ANCHOR).run(bundle=holed)
    assert run.provenance["ok"], run.provenance["violations"][:2]
