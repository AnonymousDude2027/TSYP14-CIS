"""Attack, fault and natural-event generators: determinism, magnitudes, labels."""
import numpy as np
import pandas as pd
import pytest

from tsfdi.attacks.families import (Injector, natural_event_mask,
                                    percent_of_range_to_sigma)


def _inj(n=2000, sigma=1.5, seed=0):
    return Injector(sigma=sigma, seasonal_range=20.0, kind="temperature", n=n,
                    test_start=300, seed=seed)


def test_step_bias_has_the_requested_size():
    inj = _inj()
    spec = inj.make_spec("step", np.random.default_rng(0), onset=500, duration=100,
                         magnitude_sigma=3.0, magnitude_percent=None)
    y = np.zeros(2000) + 20.0
    res = inj.inject(y, spec)
    assert abs(res.delta[500] - 3.0 * 1.5) < 1e-9
    assert np.all(res.delta[599] != 0) and res.delta[600] == 0
    assert res.labels.sum() == 100


def test_a_negative_step_reports_its_magnitude():
    """Regression: abs(delta.max()) is 0 for a negative step (untouched days are 0),
    so every downward step recorded a magnitude of zero and the grid collapsed."""
    y = np.full(500, 10.0)
    inj = _inj(n=500)
    for sign in (-1, 1):
        spec = inj.make_spec("step", np.random.default_rng(1), onset=100, duration=50,
                             magnitude_sigma=2.0, sign=sign)
        res = inj.inject(y, spec)
        assert res.meta["magnitude_sigma"] == pytest.approx(2.0, abs=1e-9)
        assert res.meta["magnitude_units"] == pytest.approx(2.0 * 1.5, abs=1e-9)
        assert np.sign(res.delta[110]) == sign


def test_noise_shaped_takes_its_amplitude_from_the_grid():
    y = np.full(800, 10.0)
    inj = _inj(n=800)
    sizes = []
    for mag in (0.5, 3.0):
        spec = inj.make_spec("noise_shaped", np.random.default_rng(2), onset=200, duration=100,
                             magnitude_sigma=mag)
        res = inj.inject(y, spec)
        sizes.append(float(np.std(res.delta[res.labels])))
    assert sizes[1] > 3 * sizes[0]


def test_scaling_is_a_fractional_gain_on_the_grid_axis():
    y = np.full(600, 10.0)
    inj = _inj(n=600)
    spec = inj.make_spec("scaling", np.random.default_rng(3), onset=100, duration=200,
                         magnitude_sigma=1.0)
    res = inj.inject(y, spec)
    assert res.meta["gain"] > 0
    assert res.meta["magnitude_sigma"] == pytest.approx(1.0, abs=1e-6)


def test_multiplicative_step_and_scaling_are_different_attacks():
    """Regression: a multiplicative step used to be applied as a factor, which made
    it byte-identical to the scaling family and collapsed two families into one."""
    rng = np.random.default_rng(4)
    y = np.where(rng.random(600) < 0.4, rng.gamma(1.0, 2.0, 600), 0.0)
    inj = Injector(sigma=0.19, seasonal_range=20.0, kind="multiplicative", n=600, test_start=100)
    out = {}
    for fam in ("step", "scaling"):
        spec = inj.make_spec(fam, np.random.default_rng(5), onset=300, duration=60,
                             magnitude_sigma=2.0, sign=1)
        out[fam] = inj.inject(y, spec).delta
    assert not np.allclose(out["step"], out["scaling"])
    # and the step is additive in the SP1 (square-root) coordinate, not in mm
    spec = inj.make_spec("step", np.random.default_rng(5), onset=300, duration=60,
                         magnitude_sigma=2.0, sign=1)
    d = inj.inject(y, spec).delta
    wet = np.flatnonzero((y[300:360] > 1) & np.isfinite(y[300:360]))
    dw = np.sqrt(y[300:360][wet] + d[300:360][wet]) - np.sqrt(y[300:360][wet])
    assert np.std(dw) < 1e-9 and abs(np.mean(dw) - 2.0 * 0.19) < 1e-6


def test_percent_of_range_is_converted_with_the_seasonal_range():
    assert percent_of_range_to_sigma(12.0, 20.0, 1.0) == pytest.approx(2.4)
    inj = _inj()
    spec = inj.make_spec("step", np.random.default_rng(1), onset=500, duration=50,
                         magnitude_percent=12.0)
    res = inj.inject(np.zeros(2000) + 20.0, spec)
    assert np.isclose(abs(res.delta[500]), 0.12 * 20.0)


def test_replay_copies_existing_values_exactly():
    rng = np.random.default_rng(2)
    y = np.round(20 + 5 * rng.normal(size=2000), 1)
    inj = _inj()
    spec = inj.make_spec("replay", np.random.default_rng(2), onset=800, length=7, source_offset=60)
    res = inj.inject(y, spec)
    assert np.array_equal(res.values[800:807], y[740:747])
    assert res.labels.sum() == 7


def test_replay_with_gain_and_noise_breaks_exactness():
    rng = np.random.default_rng(3)
    y = np.round(20 + 5 * rng.normal(size=2000), 1)
    inj = _inj()
    spec = inj.make_spec("replay", np.random.default_rng(3), onset=800, length=7,
                         source_offset=60, gain=1.05, noise_sigma=0.3)
    res = inj.inject(y, spec)
    assert not np.array_equal(res.values[800:807], y[740:747])


def test_ramp_reaches_full_magnitude_then_holds():
    inj = _inj()
    spec = inj.make_spec("ramp", np.random.default_rng(4), onset=500, magnitude_sigma=4.0,
                         magnitude_percent=None)
    spec.rise_days, spec.duration = 40, 120
    res = inj.inject(np.zeros(2000) + 20.0, spec)
    assert abs(res.delta[500]) < 1e-9                 # starts at zero
    assert abs(res.delta[540] - 4.0 * 1.5) < 1e-6     # reaches full size at the end of the rise
    assert abs(res.delta[600] - 4.0 * 1.5) < 1e-6     # holds


def test_injection_is_deterministic_for_a_fixed_seed():
    y = np.linspace(10, 30, 2000)
    specs = [_inj(seed=7).make_spec("replay", np.random.default_rng(0), onset=800,
                                    gain=1.0, noise_sigma=0.5) for _ in range(2)]
    # identical spec, compared NaN-aware (unused magnitudes are NaN by design)
    d0, d1 = specs[0].as_dict(), specs[1].as_dict()
    for k in d0:
        assert (d0[k] == d1[k]) or (isinstance(d0[k], float) and np.isnan(d0[k]) and np.isnan(d1[k])), k
    a = _inj(seed=7).inject(y, specs[0])
    b = _inj(seed=7).inject(y, specs[1])
    assert np.allclose(a.values, b.values, equal_nan=True)
    # and a different seed must give a different realisation for a noisy family
    sp = _inj(seed=7).make_spec("replay", np.random.default_rng(0), onset=800,
                                gain=1.0, noise_sigma=0.5)
    c = _inj(seed=7).inject(y, sp)
    d = _inj(seed=99).inject(y, sp)
    assert not np.allclose(c.values, d.values, equal_nan=True)


def test_dropout_creates_missing_days_and_labels_them():
    inj = _inj()
    spec = inj.make_spec("dropout", np.random.default_rng(5), onset=400, duration=15)
    res = inj.inject(np.zeros(2000) + 10.0, spec)
    assert np.isnan(res.values[400:415]).all()
    assert res.labels.sum() == 15


def test_stuck_at_repeats_the_previous_value():
    y = np.arange(2000, dtype=float)
    inj = _inj()
    spec = inj.make_spec("stuck_at", np.random.default_rng(6), onset=400, duration=20)
    res = inj.inject(y, spec)
    assert np.allclose(res.values[400:420], y[399])


def test_natural_event_mask_finds_runs():
    rng = np.random.default_rng(9)
    v = rng.normal(10, 1.0, 400)          # a realistic spread so the 95th percentile is meaningful
    v[100:105] = 30.0
    m = natural_event_mask(v, "temperature", 95, 3)
    assert m[100:105].all()
    assert not m[:99].any()
    assert m.sum() <= 12          # the run plus chance exceedances at the 95th percentile


def test_multiplicative_magnitude_is_a_factor():
    inj = Injector(sigma=1.0, seasonal_range=0.0, kind="multiplicative", n=300, test_start=50)
    spec = inj.make_spec("step", np.random.default_rng(8), onset=100, duration=50,
                         magnitude_percent=12.0)
    res = inj.inject(np.zeros(300) + 10.0, spec)
    assert np.isclose(res.delta[100], 1.2)
