"""``make verify``: compare the current metrics with ``expected_results.json``.

The jury is meant to run the pipeline on a fresh virtual machine and compare
numbers.  That only works if "the same" is defined with a tolerance and a metric
list, so this module reads ``expected_results.json`` (committed) and the current
``results/`` files, and fails loudly on any difference above ``tol``.

Scope, stated plainly: this function only *reads* ``results/``.  It is a
reproduction check because ``make all`` (and ``run.sh``) regenerate
``results/`` with ``experiments`` + ``report-figures`` immediately before
calling it; invoked on its own against the committed results it certifies
internal consistency, not reproducibility.  ``--regenerate`` rewrites
``expected_results.json`` from the current tree, which is the only way it should
be updated after an intended change to the pipeline.

Tolerances are relative for continuous metrics and absolute for rates, because a
detection rate of 0.87 that becomes 0.85 on another machine is not a regression
(a Monte-Carlo standard error with 30 injections is ~0.06), whereas a false-alarm
rate that doubles is.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from . import config

EXPECTED = "expected_results.json"


def _metrics() -> dict:
    """Extract the checkable scalar metrics from the current results directory."""
    out: dict[str, float] = {}
    r = config.results_dir()

    def _add(name: str, value) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(v):
            out[name] = v

    t1 = r / "tables" / "table1_standard_profile.csv"
    if t1.is_file():
        d = pd.read_csv(t1)
        for _, row in d.iterrows():
            _add(f"table1.{row['family']}.{row['variable']}.detection_rate", row.get("detection_rate"))
            _add(f"table1.{row['family']}.{row['variable']}.delay_median", row.get("delay_median"))

    t3 = r / "tables" / "table3_false_alarms.csv"
    if t3.is_file():
        d = pd.read_csv(t3)
        _add("table3.fa_median_per_year", d["events_per_year"].median())
        _add("table3.fa_max_per_year", d["events_per_year"].max())

    t2 = r / "tables" / "table2_detection_by_attack.csv"
    if t2.is_file():
        d = pd.read_csv(t2)
        d = d[(d["variable"] == "TMAX")]
        for _, row in d.iterrows():
            _add(f"table2.{row['family']}.{row['magnitude_sigma']}.detection_rate", row.get("detection_rate"))

    b = r / "bench_provenance.csv"
    if b.is_file():
        d = pd.read_csv(b)
        for _, row in d.iterrows():
            if isinstance(row["value"], (int, float)):
                _add(f"bench.{row['metric']}", row["value"])

    c = r / "calibration.csv"
    if c.is_file():
        d = pd.read_csv(c)
        _add("calibration.n_models", len(d))
        _add("calibration.sigma_over_range_median", d["sigma_over_range"].median())
    g = r / "gonogo.csv"
    if g.is_file():
        d = pd.read_csv(g)
        _add("gonogo.n_pairs", len(d))
        _add("gonogo.mdb_median", d["min_detectable_bias_sigma"].median())
    return out


def _tolerance_for(name: str, value: float, tol: float) -> tuple[float, float]:
    """The acceptance interval for one metric.

    Three classes, because one tolerance cannot be right for all of them:

    * **rates** (detection rates, alarms per sensor-year, proportions) are compared
      with an absolute tolerance.  A detection rate of 0.90 based on 30 injections
      has a Monte-Carlo standard error of ~0.055, and a false-alarm rate of 1.0/yr
      based on ~5 events has one of ~0.45/yr; a relative tolerance would either
      pass everything or fail everything.
    * **quantiles of small integers** (median delay in days) are compared with an
      absolute tolerance of 1 day plus the relative one.
    * **continuous measurements** (sigma, timing, memory) are compared relative to
      ``tol``, with a floor so that a value which should be 0.02 cannot fail because
      it came out as 0.0200001.
    """
    a = abs(float(value))
    if name.endswith(("_rate", ".rate")) or "proportion" in name:
        # Fixed seeds make a re-run reproducible; the tolerance only has to absorb
        # floating-point differences at a threshold boundary (one injection in a
        # cell of 20 is 0.05, so 0.06 allows a single boundary flip and no more).
        pad = max(0.06, 0.02 * a)
        return value - pad, value + pad
    if "per_year" in name:
        # an alarm rate is a count divided by years: a fraction of an event per year
        pad = max(0.1, 0.2 * a)
        return value - pad, value + pad
    if "delay" in name or "n_models" in name or "n_pairs" in name:
        pad = 1.0 + tol * a
        return value - pad, value + pad
    pad = max(tol * a, 0.02)
    if "us_per" in name or "ms_per" in name or "seconds" in name or "mb" in name:
        pad = max(0.15 * a, 0.5)      # timing measurements vary with the machine
    return value - pad, value + pad


def compare_expected(tol: float = 0.02, regenerate: bool = False) -> bool:
    current = _metrics()
    p = config.repo_root() / EXPECTED
    if regenerate or not p.is_file():
        p.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {p} with {len(current)} metrics")
        return True
    expected = json.loads(p.read_text(encoding="utf-8"))
    problems = []
    for k, v in expected.items():
        if k not in current:
            problems.append(f"missing metric: {k}")
            continue
        cur = current[k]
        lo, hi = _tolerance_for(k, v, tol)
        if not (lo <= cur <= hi):
            problems.append(f"{k}: expected {v:.6g}, got {cur:.6g} "
                            f"(allowed {lo:.6g}..{hi:.6g})")
    if problems:
        print(f"VERIFY FAILED ({len(problems)} of {len(expected)} metrics)")
        for q in problems[:25]:
            print("  -", q)
        return False
    print(f"VERIFY OK: {len(expected)} metrics within tolerance {tol:.0%}")
    return True
