"""SP3 tests: AR order selection, gap handling, and the rank-Gauss transform.

Every assertion here is about something that was measured to be wrong at least
once during development, so these are regression tests with a reason.
"""
import numpy as np
import pytest

from tsfdi.sp.whitening import RankGauss, Whitener, fit_ar


def _conditioned(r):
    from tsfdi.sp.conditioning import Conditioned

    n = len(r)
    dates = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    return Conditioned(dates=dates, y=r, x=r, valid=np.isfinite(r),
                       t=np.arange(n, dtype=float), gap=~np.isfinite(r),
                       reference_gap=np.zeros(n, bool), units="degC", variable="TMAX",
                       kind="temperature", site_id="TEST", site_name="TEST")


def _ar1(n=4000, phi=0.6, seed=0, gaps=0.0):
    rng = np.random.default_rng(seed)
    e = rng.normal(0, 1, n)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = phi * r[i - 1] + e[i]
    if gaps:
        r[rng.random(n) < gaps] = np.nan
    return r


def test_ar_order_selection_recovers_a_strong_ar1():
    r = _ar1(phi=0.6)
    phi, se = fit_ar(r, 1)
    assert abs(phi[0] - 0.6) < 0.03
    assert se == pytest.approx(1.0, abs=0.05)      # innovation sd of the DGP


def test_aic_prefers_ar1_over_ar0_when_the_residual_is_autocorrelated():
    """Regression: p = 0 was scored with a MAD and p >= 1 with a standard
    deviation, so AIC always chose p = 0 and the autocorrelation went straight
    into CUSUM."""
    from tsfdi.sp.conditioning import Conditioned

    r = _ar1(phi=0.6, n=2000)
    n = len(r)
    c = _conditioned(r)
    w = Whitener().fit(c, r)
    assert w.p >= 1
    assert abs(np.atleast_1d(w.phi)[0] - 0.6) < 0.05


def test_whitening_removes_the_autocorrelation_of_z():
    r = _ar1(phi=0.6, n=4000)
    n = len(r)
    dates = np.arange(n).astype("datetime64[D]").astype("datetime64[ns]")
    c = _conditioned(r)
    w = Whitener().fit(c, r)
    _, z = w.transform(c, r)
    zz = z[np.isfinite(z)]
    acf1 = float(np.corrcoef(zz[:-1], zz[1:])[0, 1])
    assert abs(acf1) < 0.06
    assert 0.8 < float(np.std(zz)) < 1.25


def test_gaps_do_not_shift_the_ar_fit_onto_the_wrong_rows():
    """Regression: the row mask was applied to an already-filtered array."""
    full = _ar1(phi=0.5, n=6000, seed=3)
    gappy = full.copy()
    rng = np.random.default_rng(4)
    gappy[rng.random(len(full)) < 0.35] = np.nan
    phi_f, _ = fit_ar(full, 1)
    phi_g, _ = fit_ar(gappy, 1)
    assert abs(phi_f[0] - phi_g[0]) < 0.05


def test_rank_gauss_is_monotone_and_normalising():
    rng = np.random.default_rng(0)
    r = np.abs(rng.gamma(0.6, 1.0, 20000)) - 0.5      # zero-inflated, heavy right tail
    pit = RankGauss.fit(r)
    w = pit(r)
    assert np.all(np.diff(pit(np.sort(r))) >= -1e-12)   # monotone
    assert abs(float(np.nanstd(w)) - 1.0) < 0.05
    assert float(np.nanmax(w)) <= pit.cap + 1e-9
    # a positive shift stays positive, and an out-of-sample extreme is extrapolated
    assert float(np.nanmedian(pit(r + 1.0))) > float(np.nanmedian(pit(r)))
    assert float(pit(np.array([1e6]))[0]) == pit.cap


def test_rank_gauss_is_a_no_op_when_it_is_not_fitted():
    r = np.array([1.0, 2.0, np.nan, 4.0])
    assert np.allclose(RankGauss()(r), r, equal_nan=True)
