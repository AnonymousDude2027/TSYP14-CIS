"""L5 impact: for what purpose was the data modified?

We cannot know the attacker's intent, so we estimate the *effect on decisions*:
the number of days on which a documented decision rule would have returned a
different answer for the true series and for the series the analyst actually
received (specification section 11).

Two numbers are produced, and the difference between them is itself a result:

* ``flips_oracle`` -- decisions compared with the true series.  This is what the
  evaluation can measure and what the report tables quote.  It is an upper
  bound on what an operator could know.
* ``flips_operational`` -- decisions compared with the series the pipeline
  actually uses after its response (QUARANTINE falls back to the reference
  expectation).  This is what the system can estimate at run time, without
  knowing the truth, and it is what the dashboard shows.

If the two are close, the response layer is effective; if they diverge, quarantine
is buying less than it appears to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .. import config


def _series_by_var(site: str, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return values


def decision_sets(values_by_var: dict[str, np.ndarray], dates: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    """Apply the configured decision rules; returns name -> boolean decision per day."""
    rules = config.load("decisions.yaml")["rules"]
    out: dict[str, np.ndarray] = {}
    for name, r in rules.items():
        var = r["variable"]
        if var not in values_by_var:
            continue
        v = np.asarray(values_by_var[var], dtype=float)
        w = int(r["window"])
        thr = float(r["threshold"])
        op = r["op"]
        if w > 1:
            s = pd.Series(v).rolling(w, min_periods=max(2, w // 2)).sum().to_numpy()
        else:
            s = v
        with np.errstate(invalid="ignore"):
            if op == "lt":
                d = s < thr
            elif op == "gt":
                d = s > thr
            else:
                raise ValueError(op)
        # a NaN input yields no decision (not a flip)
        d = np.where(np.isfinite(s), d, False).astype(bool)
        out[name] = d
    return out


def decision_flips(true_values: dict[str, np.ndarray], used_values: dict[str, np.ndarray],
                   dates: pd.DatetimeIndex, window: np.ndarray | None = None) -> dict:
    """Count days on which any decision differs, inside ``window``."""
    dt = decision_sets(true_values, dates)
    du = decision_sets(used_values, dates)
    per_rule = {}
    total = np.zeros(len(dates), dtype=bool)
    for name in dt:
        diff = dt[name] != du[name]
        if window is not None:
            diff = diff & window
        per_rule[name] = int(diff.sum())
        total |= diff
    return {"flips_total": int(total.sum()), "per_rule": per_rule,
            "flip_days": np.flatnonzero(total)}
