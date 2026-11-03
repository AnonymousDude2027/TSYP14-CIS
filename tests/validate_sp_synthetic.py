"""Validate SP1-SP3 and the D1/D2 false-alarm calibration on synthetic data.

The synthetic generator has a *known* ground truth, which real data never does:

    y_t = 10 + 12 sin(2 pi t / 365.2422) + 0.9 x_t + AR(1) noise (phi=0.6)
                                                       + seasonally varying sigma

Checks
 1. SP2 recovers the coefficients it was given.
 2. SP3 produces z with robust scale ~1 and near-zero lag-1 ACF (the raw
    residual has ACF ~0.6).
 3. On clean data the calibrated suite meets its false-alarm budget.
 4. On clean data the *unwhitened* baseline (ablation B1) does not: that is the
    quantitative justification for SP3.
 5. A 12 %-of-range step is detected.
"""
import numpy as np
import pandas as pd

from tsfdi import config
from tsfdi.io import SeriesBundle
from tsfdi.sp import Conditioning, Whitener, fit_expectation
from tsfdi.sp.expectation import robust_scale
from tsfdi.detect.engine import build_model
from tsfdi.detect.statistics import event_onsets, event_onsets_mask

RNG = np.random.default_rng(7)
N = 3653
dates = pd.date_range("2015-01-01", periods=N, freq="D")


def synth(seed=7, phi=0.6, sigma0=1.6, amp=12.0, beta=0.9):
    rng = np.random.default_rng(seed)
    t = np.arange(N)
    x = 12 * np.sin(2 * np.pi * t / 365.2422 - 1.0) + rng.normal(0, 2.5, N)
    season = 1.0 + 0.35 * np.sin(2 * np.pi * t / 365.2422)
    e = np.zeros(N)
    for i in range(1, N):
        e[i] = phi * e[i - 1] + rng.normal(0, sigma0 * season[i])
    y = 8.0 + beta * x + amp * np.sin(2 * np.pi * t / 365.2422) + 3 * np.sin(4 * np.pi * t / 365.2422) + e
    return y, x


y, x = synth()
b = SeriesBundle(
    site_id="SYN00000001", site_name="SYNTH", variable="TMAX", reference_variable="T2M_MAX",
    units="degC", kind="temperature", dates=dates, sensor=y, reference=x,
    sensor_ok=np.ones(N, bool), reference_ok=np.ones(N, bool),
)

c = Conditioning().transform(b)
cal = np.asarray(dates < pd.Timestamp("2020-01-01"))
model = fit_expectation(c, mask=cal)
print("SP2 coefficients (expect intercept 8, beta 0.9, seasonal range 24):")
print("  b0=%.3f  b1=%.3f  seasonal_range=%.2f  sigma_resid=%.3f"
      % (model.coef[0], model.coef[1], model.seasonal_range(c), model.sigma))

resid = model.residual(c)
w = Whitener().fit(c, resid, mask=cal)
_, z = w.transform(c, resid)
zc = z[cal]
print("SP3: AR order=%d phi=%s sigma_z=%.3f  acf1(z)=%.3f (raw residual acf1=%.3f)"
      % (w.p, np.round(w.phi, 3), robust_scale(zc), w.acf1,
         float(np.corrcoef(resid[cal][:-1], resid[cal][1:])[0, 1])))

# --- 3/4: false-alarm calibration on clean data, full vs unwhitened -----------
m = build_model("SYNTH", "TMAX", budget=1.0, rng=np.random.default_rng(11),
                bundle=b, calibrate=True)
test = np.asarray(dates >= pd.Timestamp("2020-01-01"))
al = m.alarms()
ex = al["exceed_any"]
print("\nCalibrated thresholds:", {k: round(v, 2) for k, v in m.thresholds.items()})
print("clean test-period alarm events (whole suite):", int(event_onsets(ex[test], 14)),
      "  -> per year: %.2f" % (float(event_onsets(ex[test], 14)) * 365.25 / test.sum()))

# unwhitened ablation: same detectors, z without SP3
m2 = build_model("SYNTH", "TMAX", budget=1.0, rng=np.random.default_rng(11),
                 bundle=b, calibrate=True, use_whitening=False)
al2 = m2.alarms()
print("unwhitened alarm events on clean test period:",
      int(event_onsets(al2["exceed_any"][test], 14)),
      "  -> per year: %.2f" % (float(event_onsets(al2["exceed_any"][test], 14)) * 365.25 / test.sum()))

# --- 5: detect a 12 % step ----------------------------------------------------
y2 = y.copy()
rng = np.random.default_rng(3)
onset = 2500
rng_step = 0.12 * m.seasonal_range
y2[onset:onset + 120] += rng_step
b2 = SeriesBundle(**{**b.__dict__, "sensor": y2})
c2 = Conditioning().transform(b2)
pred = m.expectation.predict(c2)
r2 = c2.y - pred
r2[~c2.valid] = np.nan
_, z2 = m.whitener.transform(c2, r2)
# recompute alarms on the attacked series (thresholds unchanged: calibrated on clean data)
import copy
m3 = copy.copy(m)
m3.conditioned = c2
m3.z = z2
m3.residual = r2
al3 = m3.alarms()
mask = event_onsets_mask(al3["exceed_any"], 14)
ons = np.flatnonzero(mask)
fired = [k for k, v in al3["exceed"].items() if v[onset:onset + 120].any()]
print("\n12%% step = %.2f degC (sigma_resid=%.2f -> %.1f sigma)"
      % (rng_step, m.sigma_residual, rng_step / m.sigma_residual))
print("  alarm events at/after onset:", ons[ons >= onset][:3], " first delay:", int(ons[ons >= onset][0] - onset) if np.any(ons >= onset) else None)
print("  detectors that fired in window:", fired)
