"""Streamlit dashboard (specification section 13).

Run with::

    streamlit run src/tsfdi/app/dashboard.py

Panels: sensor vs reference, residual z, detector statistics, alarm state,
diagnosis with the explanation text and the recommended action, decision impact
and provenance status.  Everything is computed offline from the bundle.

If Streamlit is unavailable (the jury VM may have nothing beyond Python), the
identical content is produced by ``python -m tsfdi.report_static`` into
``results/static_report.html`` -- the dashboard is a convenience, never a
single point of failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tsfdi import config  # noqa: E402


def _run(site: str, variable: str, attack: str, magnitude: float, percent: float | None):
    from tsfdi.attacks.families import Injector
    from tsfdi.eval.runner import get_model, model_bundle
    from tsfdi.pipeline import StreamingPipeline

    m = get_model(site, variable)
    b = model_bundle(m.site, variable)
    values = b.sensor
    spec = None
    if attack and attack != "none":
        test_start = int(np.flatnonzero(b.dates >= np.datetime64(config.period("test")[0]))[0])
        inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, len(values), test_start=test_start)
        spec = inj.make_spec(attack, np.random.default_rng(config.SEED),
                             magnitude_sigma=magnitude, magnitude_percent=percent)
        res = inj.inject(values, spec)
        values = res.values
    out = StreamingPipeline(m).run(b.with_sensor(values))
    return m, b, out, spec


def main() -> None:  # pragma: no cover - UI code
    import streamlit as st

    st.set_page_config(page_title="TSFDI - tamper-evident sensing", layout="wide")
    st.title("Tamper-evident in-situ sensing")
    st.caption("False-data injection detection for daily climate series compared against NASA POWER. "
               "All numbers are computed offline from the committed data bundle.")

    sites = [s.name for s in config.sites()]
    c1, c2, c3, c4 = st.columns(4)
    site = c1.selectbox("Site", sites, index=sites.index("CARTHAGE") if "CARTHAGE" in sites else 0)
    variable = c2.selectbox("Variable", ["TMAX", "TMIN", "PRCP"], index=0)
    attack = c3.selectbox("Injection", ["none", "step", "ramp", "replay", "scaling",
                                        "noise_shaped", "stuck_at", "dropout", "spike"], index=0)
    magnitude = c4.slider("Magnitude (robust sigma)", 0.0, 6.0, 3.0, 0.5)

    with st.spinner("running the pipeline L0-L5 ..."):
        m, b, out, spec = _run(site, variable, attack, magnitude, None)

    frame = pd.DataFrame({"date": out.dates, "sensor_true": b.sensor, "sensor_used": out.sensor,
                          "reference": b.reference, "z": out.z})
    st.subheader("1. Sensor versus reference")
    st.line_chart(frame.set_index("date")[["sensor_true", "reference"]])
    st.subheader("2. Whitened residual z (SP3)")
    st.line_chart(frame.set_index("date")[["z"]])
    st.subheader("3. Alarm state and diagnosis")
    st.metric("alarm events", int(out.alarms.sum()))
    ev_dates = frame.loc[out.alarms.astype(bool), "date"]
    if len(ev_dates):
        st.write("alarm days:", ", ".join(d.strftime("%Y-%m-%d") for d in ev_dates[:20]))
    for a in out.alerts[:5]:
        st.markdown(f"**{a['timestamp']} - {a['action']}**")
        st.write(a["explanation_text"])
        st.json({k: a[k] for k in ("detectors_fired", "class_probs", "confidence", "evidence", "impact")})
    st.subheader("4. Provenance status (L0)")
    st.json(out.provenance)
    st.subheader("5. Latency (ms per reading, measured)")
    st.json({k: round(v, 4) for k, v in out.latency_ms.items()})


if __name__ == "__main__":  # pragma: no cover
    main()
