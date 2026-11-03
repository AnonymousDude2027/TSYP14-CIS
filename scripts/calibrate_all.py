"""Fit and calibrate every (site, variable) model; caches them to results/cache/models.

Usage:  python scripts/calibrate_all.py [budget] [--resume]

``--resume`` keeps models already present in the cache (useful when a long run is
interrupted); without it every model is rebuilt, which is what ``make calibrate``
does so that the published numbers never depend on a stale cache.
"""
import sys, time, os
sys.path.insert(0, "src")
from tsfdi import config
from tsfdi.eval.runner import get_model, _cache_path

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
resume = "--resume" in flags
budget = float(args[0]) if args else None

n_ok = n_fail = n_skip = 0
t_all = time.time()
for s in config.sites():
    for v in s.variables:
        t0 = time.time()
        try:
            p = _cache_path(s.name, v)
            if resume and os.path.isfile(p):
                n_skip += 1
                continue
            m = get_model(s.name, v, rebuild=True, budget=budget)
            al = m.alarms()
            from tsfdi.detect.statistics import event_onsets_mask
            import numpy as np
            ev = np.flatnonzero(event_onsets_mask(al["exceed_any"], 14))
            test = ~m._cal
            fa = ev[test[ev]].size / (test.sum() / 365.25)
            print(f"[ok] {s.name:22s} {v:5s} sigma={m.sigma_residual:6.3f} range={m.seasonal_range:6.2f} "
                  f"FA_test={fa:5.2f}/yr  ({time.time()-t0:5.1f}s)", flush=True)
            n_ok += 1
        except Exception as exc:
            print(f"[FAIL] {s.name} {v}: {exc}", flush=True)
            n_fail += 1
print(f"done {n_ok} ok, {n_fail} failed, {n_skip} skipped in {time.time()-t_all:.0f}s")

# Aggregate table (same columns as ``python -m tsfdi calibrate`` writes), built
# from the cache so that a resumed run still produces results/calibration.csv.
import numpy as np, pandas as pd
from tsfdi.detect.statistics import event_onsets_mask
rows = []
for s in config.sites():
    for v in s.variables:
        p = _cache_path(s.name, v)
        if not os.path.isfile(p):
            continue
        m = get_model(s.name, v)
        al = m.alarms()
        ev = np.flatnonzero(event_onsets_mask(al["exceed_any"], 14))
        test = ~m._cal
        rows.append({
            "site": s.name, "variable": v, "sigma_residual": m.sigma_residual,
            "sigma_z": m.sigma_z, "seasonal_range": m.seasonal_range,
            "sigma_over_range": m.sigma_residual / m.seasonal_range if m.seasonal_range else np.nan,
            "ar_order": m.whitener.p, "rank_gauss": bool(getattr(m.whitener, "rank_gauss", False)),
            "fa_test_per_year": ev[test[ev]].size / (test.sum() / 365.25),
            "fa_calibration_per_year": ev[m._cal[ev]].size / (m._cal.sum() / 365.25),
            **{f"h_{k}": v2 for k, v2 in m.thresholds.items()},
        })
if rows:
    df = pd.DataFrame(rows)
    df.to_csv(config.results_dir() / "calibration.csv", index=False)
    print(f"[calibrate] wrote results/calibration.csv ({len(df)} rows)")
