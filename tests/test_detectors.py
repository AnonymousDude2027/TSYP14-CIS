"""Detector statistics, event counting and the online/batch equivalence."""
import numpy as np
import pytest

from tsfdi.detect import statistics as st
from tsfdi.detect.online import OnlineSuite


def test_cusum_recursion():
    z = np.array([0.0, 2.0, 2.0, 2.0, -5.0, 0.0])
    up, dn = st.cusum(z[:, None], k=1.0)
    # S+ = max(0, S+ + z - 1)
    expected = np.zeros(len(z))
    s = 0.0
    for i, v in enumerate(z):
        s = max(0.0, s + v - 1.0)
        expected[i] = s
    assert np.allclose(up[:, 0], expected)


def test_cusum_ignores_missing_days():
    z = np.array([np.nan, 2.0, np.nan, 2.0])
    up, _ = st.cusum(z[:, None], k=1.0)
    assert np.allclose(up[:, 0], [0.0, 1.0, 1.0, 2.0])


def test_window_mean_matches_direct_computation():
    rng = np.random.default_rng(0)
    z = rng.normal(size=200)
    z[rng.random(200) < 0.2] = np.nan
    W = 7
    got = st.window_mean(z[:, None], W)[:, 0]
    for t in [W - 1, 50, 120, 199]:
        win = z[t - W + 1: t + 1]
        ok = np.isfinite(win)
        want = abs(np.nansum(win)) / np.sqrt(ok.sum())
        assert abs(got[t] - want) < 1e-10


def test_window_slope_matches_bruteforce_with_gaps():
    """Regression test for the bug where n (valid count) met index sums over all
    calendar slots, making the determinant negative and the statistic all-NaN."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=800)
    z[rng.random(800) < 0.3] = np.nan
    W = 30
    got = st.window_slope(z[:, None], W)[:, 0]
    assert np.isfinite(got).mean() > 0.95
    for t in range(W - 1, 800, 37):
        win = z[t - W + 1: t + 1]
        ok = np.isfinite(win)
        if ok.sum() < 5:
            continue
        i = np.arange(W, dtype=float)[ok]
        A = np.column_stack([np.ones_like(i), i])
        beta, *_ = np.linalg.lstsq(A, win[ok], rcond=None)
        want = abs(beta[1]) * np.sqrt(np.sum((i - i.mean()) ** 2))
        assert abs(got[t] - want) < 1e-8


def test_block_contrast_detects_a_rectangular_pulse():
    rng = np.random.default_rng(2)
    z = rng.normal(size=400)
    z[200:207] += 4.0
    s = st.block_contrast(z[:, None], 7)[:, 0]
    assert s[206] > 6          # inside the pulse, contrasted with the previous week
    assert np.nanpercentile(s[:150], 99) < 4


def test_event_counting_is_schmitt_triggered():
    e = np.zeros(200, dtype=bool)
    e[10:60] = True            # one long excursion
    e[70:75] = True
    e[86:95] = True            # third excursion: re-armed and past the cooldown
    # three excursions exist in this array, so exactly three events are counted;
    # the naive rule (no de-latch) would count 1 + 1 + 1 + (60-10)//14 = 6.
    assert int(st.event_onsets(e).sum()) == 3
    mask = st.event_onsets_mask(e)
    assert mask[10] and mask[70] and not mask[25]


def test_cooldown_merges_rapid_rearm():
    e = np.zeros(100, dtype=bool)
    e[10] = True
    e[12] = True
    e[40] = True
    assert int(st.event_onsets(e, cooldown=14).sum()) == 2


def test_online_suite_matches_batch_statistics(synth_model):
    """The streaming path and the vectorised path must agree exactly."""
    z = synth_model.z[1800:2400]
    suite = OnlineSuite.from_config()
    online = {k: [] for k in ("D1_step_cusum", "D2_ewma", "D4_spike")}
    for v in z:
        out = suite.step(v)
        for k in online:
            online[k].append(out.get(k, np.nan))
    batch_cusum = np.maximum(*st.cusum(z[:, None], k=1.0))[:, 0]
    batch_ewma = st.ewma(z[:, None], lam=0.1)[:, 0]
    assert np.allclose(online["D1_step_cusum"][30:], batch_cusum[30:], equal_nan=True, atol=1e-9)
    assert np.allclose(online["D2_ewma"][30:], batch_ewma[30:], equal_nan=True, atol=1e-9)


def test_online_suite_matches_batch_window_tests(synth_model):
    z = synth_model.z[1800:2400]
    suite = OnlineSuite.from_config()
    rows = [suite.step(v) for v in z]
    for W in (3, 7):
        online = np.array([r.get(f"D1_window_shift_{W}", np.nan) for r in rows])
        batch = st.window_mean(z[:, None], W)[:, 0]
        # skip the first 30 samples: the streaming chain knows history from before
        # the slice, the vectorised call on the slice does not.  After one full
        # window (the longest W is 30) the two must agree exactly.
        assert np.allclose(online[30:], batch[30:], equal_nan=True, atol=1e-9)


def test_calibrated_threshold_meets_the_budget_on_synthetic_null():
    """A threshold calibrated on a white-noise null must respect the budget."""
    from tsfdi.detect.thresholds import calibrate_threshold
    rng = np.random.default_rng(5)
    reps = rng.normal(size=(1826, 200))
    up, _ = st.cusum(reps, k=1.0)
    h, achieved = calibrate_threshold(up, budget=1.0, n_days=1826, cooldown=14)
    assert achieved <= 1.05 * 1.0 + 0.25
    # the value depends on the (conservative) latching event rule: with k=1.0 on
    # white noise the CUSUM excursions are long, so few *events* occur and the
    # calibrated threshold sits a little above 2 sigma rather than at 5 sigma
    assert h > 1.5


def test_degenerate_windows_are_not_replays():
    """Seven dry days repeat every week; that is weather, not a replay attack."""
    from tsfdi.detect.statistics import replay_exact

    assert float(replay_exact(np.zeros(90)).sum()) == 0.0
    y = np.zeros(90)
    y[20:70] = 1.5                                   # two-valued but not repeated shape
    assert float(replay_exact(y).sum()) <= 1.0
    rng = np.random.default_rng(0)
    v = rng.normal(10.0, 1.0, 300)
    v[100:107] = v[50:57]                            # a genuine 7-day copy
    fired = np.flatnonzero(replay_exact(v))
    assert fired.size == 1 and int(fired[0]) == 106
