"""Static HTML report -- the dashboard's dependency-free fallback.

``python -m tsfdi.report_static`` writes ``results/static_report.html``: one
self-contained file (inline CSS, embedded SVG charts, no external requests, no
JavaScript) that shows, for a chosen site/variable/injection, the same panels the
Streamlit dashboard shows.  If Streamlit is missing on the jury machine -- the
specification explicitly requires that the CLI and the report do not depend on it
-- this file is the demo.

The charts are drawn as inline SVG directly from the computed series, so the file
works with no network access and can be attached to a report.
"""
from __future__ import annotations

import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd

from . import config


def _svg_line(x: np.ndarray, y: np.ndarray, width: int = 900, height: int = 180,
              colour: str = "#1f77b4", ylabel: str = "", hline: float | None = None,
              vbounds: tuple[float, float] | None = None) -> str:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return "<p><em>no data</em></p>"
    x0, x1 = float(np.min(x[ok])), float(np.max(x[ok]))
    lo, hi = (vbounds if vbounds else (float(np.min(y[ok])), float(np.max(y[ok]))))
    if hi <= lo:
        hi = lo + 1.0
    pad = 26

    def sx(v):
        return pad + (v - x0) / max(x1 - x0, 1e-9) * (width - 2 * pad)

    def sy(v):
        return height - pad - (v - lo) / (hi - lo) * (height - 2 * pad)

    pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x[ok], y[ok]))
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img">',
             f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fbfbfb" stroke="#e5e5e5"/>',
             f'<polyline fill="none" stroke="{colour}" stroke-width="1" points="{pts}"/>']
    if hline is not None and lo <= hline <= hi:
        parts.append(f'<line x1="{pad}" x2="{width - pad}" y1="{sy(hline):.1f}" y2="{sy(hline):.1f}" '
                     f'stroke="#c00" stroke-dasharray="4 3" stroke-width="1"/>')
    parts.append(f'<text x="{pad}" y="14" font-size="11" fill="#444">{html.escape(ylabel)} '
                 f'[{lo:.2f}, {hi:.2f}]</text>')
    parts.append("</svg>")
    return "".join(parts)


def _table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="tbl", float_format=lambda v: f"{v:.4g}")


def build(site: str = "CARTHAGE", variable: str = "TMAX", attack: str = "step",
          magnitude: float = 3.0) -> Path:
    from .attacks.families import Injector
    from .eval.runner import get_model, model_bundle
    from .pipeline import StreamingPipeline

    m = get_model(site, variable)
    b = model_bundle(m.site, variable)
    values = b.sensor
    spec = None
    if attack != "none":
        test_start = int(np.flatnonzero(b.dates >= np.datetime64(config.period("test")[0]))[0])
        inj = Injector(m.sigma_residual, m.seasonal_range, m.kind, len(values), test_start=test_start)
        spec = inj.make_spec(attack, np.random.default_rng(config.SEED),
                             magnitude_sigma=magnitude, magnitude_percent=None)
        res = inj.inject(values, spec)
        values = res.values
    out = StreamingPipeline(m).run(b.with_sensor(values))

    t = np.arange(len(b.sensor), dtype=float)
    blocks = []
    blocks.append("<h2>1. Sensor versus reference</h2>")
    blocks.append(_svg_line(t, b.sensor, colour="#1f77b4", ylabel="true sensor"))
    blocks.append(_svg_line(t, values, colour="#d62728", ylabel="delivered sensor"))
    blocks.append(_svg_line(t, b.reference, colour="#222222", ylabel="NASA POWER reference"))
    blocks.append("<h2>2. Whitened residual z (SP3)</h2>")
    blocks.append(_svg_line(t, out.z, colour="#2ca02c", ylabel="z"))
    blocks.append("<h2>3. Alarms and diagnosis</h2>")
    blocks.append(f"<p><strong>{int(out.alarms.sum())} alarm events</strong> "
                  f"({attack}{'' if attack == 'none' else f', {magnitude} sigma'}); "
                  f"latency {out.latency_ms['L0_provenance_per_reading']:.4f} ms/reading at L0, "
                  f"{out.latency_ms['L2_detectors_per_reading']:.4f} ms/reading at L2.</p>")
    for a in out.alerts[:5]:
        blocks.append(f"<div class='alert'><strong>{html.escape(str(a['timestamp']))} - "
                      f"{html.escape(a['action'])}</strong><p>{html.escape(a['explanation_text'])}</p>"
                      f"<pre>{html.escape(str(a['class_probs']))}</pre></div>")
    blocks.append("<h2>4. Provenance status (L0)</h2>")
    blocks.append(f"<pre>{html.escape(str(out.provenance))}</pre>")
    blocks.append("<h2>5. Calibrated thresholds</h2>")
    blocks.append(_table(pd.DataFrame([{"detector": k, "threshold": v}
                                       for k, v in m.thresholds.items()])))

    css = """
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;
         padding:0 1rem;color:#111;line-height:1.45}
    h1{font-size:1.4rem} h2{font-size:1.05rem;margin-top:1.6rem;border-bottom:1px solid #eee;padding-bottom:.2rem}
    .tbl{border-collapse:collapse;font-size:.85rem} .tbl td,.tbl th{border:1px solid #e5e5e5;padding:.2rem .45rem}
    .alert{background:#fff6f6;border-left:3px solid #c00;padding:.5rem .8rem;margin:.6rem 0;font-size:.9rem}
    pre{background:#f7f7f7;padding:.5rem;overflow-x:auto;font-size:.78rem}
    .muted{color:#666;font-size:.85rem}
    """
    title = f"TSFDI static report - {site} / {variable} / injection: {attack}"
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
    <style>{css}</style></head><body>
    <h1>{html.escape(title)}</h1>
    <p class="muted">Generated offline from the committed data bundle by
    <code>python -m tsfdi.report_static</code>. No external resources, no JavaScript.
    The same content is available interactively through the Streamlit dashboard.</p>
    {''.join(blocks)}
    </body></html>"""
    p = config.results_dir() / "static_report.html"
    p.write_text(doc, encoding="utf-8")
    print(f"[report_static] wrote {p}")
    return p


if __name__ == "__main__":  # pragma: no cover
    build()
