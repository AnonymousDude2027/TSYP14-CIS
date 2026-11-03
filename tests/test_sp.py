"""SP1-SP6: numerical correctness against known ground truth."""
import numpy as np
import pandas as pd
import pytest

from conftest import N, make_bundle, synth_series
from tsfdi.sp import Conditioning, Whitener, fit_expectation
from tsfdi.sp.expectation import fit_climatology, harmonic_design, robust_scale
from tsfdi.sp.fingerprint import BANDS, _band_power, _coherence_fast, _welch_fast
from tsfdi.sp.matched import ReplayDetector, ncc_profile, quantised_equal
from tsfdi.sp.surrogate import phase_randomize


def test_sp2_recovers_synthetic_coefficients(synth_bundle):
    from tsfdi import config
    c = Conditioning().transform(synth_bundle)
    cal = np.asarray(c.dates < np.datetime64("2020-01-01"))
    m = fit_expectation(c, mask=cal)
    # the generator used intercept 8.0 and slope 0.9 on the reference
    assert abs(m.coef[0] - 8.0) < 0.6
    assert abs(m.coef[1] - 0.9) < 0.03
    assert m.sigma < 2.6        # sigma of the AR(1) innovation is 1.6-1.9 after harmonics


def test_climatology_range_is_the_seasonal_range(synth_bundle):
    c = Conditioning().transform(synth_bundle)
    cal = np.asarray(c.dates < np.datetime64("2020-01-01"))
    clim = fit_climatology(c, mask=cal)
    rng_ = clim.seasonal_range(c)
    # generator: 0.9*x + 12*sin + 3*sin(2x) with x a 12 degC sinusoid, so the
    # total annual peak-to-trough is roughly 0.9*24 + 24 + 6 ~ 51 degC; the
    # harmonic fit captures the first two harmonics of that composite.
    assert 25 < rng_ < 55


def test_sp3_whitens_the_residual(synth_bundle):
    c = Conditioning().transform(synth_bundle)
    cal = np.asarray(c.dates < np.datetime64("2020-01-01"))
    m = fit_expectation(c, mask=cal)
    r = m.residual(c)
    w = Whitener().fit(c, r, mask=cal)
    _, z = w.transform(c, r)
    zc = z[cal]
    zc = zc[np.isfinite(zc)]
    # the AR(1) structure is removed: lag-1 ACF collapses, scale is ~1
    raw_acf1 = np.corrcoef(r[cal][:-1][np.isfinite(r[cal][:-1])], r[cal][1:][np.isfinite(r[cal][:-1])])[0, 1]
    assert abs(w.acf1) < 0.15
    assert raw_acf1 > 0.4
    assert 0.8 < robust_scale(zc) < 1.25


def test_sp3_handles_gaps_by_state_carrying():
    """AR filter must not discard the p days after every gap."""
    y, x = synth_series(gaps=0.3)
    b = make_bundle(y, x)
    c = Conditioning().transform(b)
    cal = np.asarray(c.dates < np.datetime64("2020-01-01"))
    m = fit_expectation(c, mask=cal)
    r = m.residual(c)
    w = Whitener().fit(c, r, mask=cal)
    _, z = w.transform(c, r)
    # nearly every present observation must produce a z
    present = np.isfinite(r) & cal
    assert np.isfinite(z[present]).mean() > 0.97


def test_sp1_does_not_invent_values():
    y, x = synth_series(n=400, gaps=0.4)
    b = make_bundle(y, x)
    c = Conditioning().transform(b)
    assert np.isnan(c.y[~np.isfinite(y)]).all()
    assert c.gap.sum() == int(np.isnan(y).sum())


def test_fingerprint_spectral_estimator_matches_scipy():
    from scipy.signal import welch
    rng = np.random.default_rng(0)
    v = rng.normal(size=180)
    f1, p1 = _welch_fast(v, 64)
    f2, p2 = welch(v, fs=1.0, nperseg=64)
    # band-integrated power is the quantity used; assert 3 % agreement
    for lo, hi in BANDS.values():
        a, b = _band_power(f1, p1, lo, hi), _band_power(f2, p2, lo, hi)
        assert abs(a - b) / max(b, 1e-12) < 0.03


def test_coherence_of_a_signal_with_itself_is_one():
    rng = np.random.default_rng(1)
    v = rng.normal(size=200)
    f, c = _coherence_fast(v, v, 64)
    assert np.nanmax(np.abs(c - 1.0)) < 1e-6


def test_surrogate_preserves_spectrum_and_destroys_repeats():
    rng = np.random.default_rng(2)
    v = np.sin(np.linspace(0, 40, 600)) + rng.normal(0, 0.1, 600)
    s, _ = phase_randomize(v, rng)
    P1 = np.abs(np.fft.rfft(v)) ** 2
    P2 = np.abs(np.fft.rfft(s)) ** 2
    assert np.corrcoef(P1[1:], P2[1:])[0, 1] > 0.999          # spectrum kept
    assert not np.allclose(v, s)                              # realisation changed


def test_ncc_profile_matches_bruteforce():
    rng = np.random.default_rng(3)
    s = rng.normal(size=60)
    w = s[20:27].copy()
    prof = ncc_profile(s, w)
    brute = np.array([np.corrcoef(s[j:j + 7], w)[0, 1] for j in range(len(s) - 7 + 1)])
    assert np.nanmax(np.abs(prof - brute)) < 1e-10
    assert np.nanargmax(prof) == 20


def test_replay_detector_finds_an_exact_copy():
    rng = np.random.default_rng(4)
    y = np.round(rng.normal(20, 3, 300), 1)
    y[200:207] = y[100:107]
    det = ReplayDetector(length=7, min_source_gap=2)
    m = det.evaluate(y, 206)
    assert m.exact
    assert abs(m.lag_days - 100) <= 1


def test_quantised_equal_respects_data_resolution():
    a = np.array([1.0, 2.0, 3.0])
    assert quantised_equal(a, a + 0.01) == 3
    assert quantised_equal(a, a + 0.2) == 0
