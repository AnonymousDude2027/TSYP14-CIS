"""Detector statistics as time series.

Every detector in this system is expressed as a **non-negative statistic time
series** plus a threshold:

    alarm(t)  <=>  S_t > h

This uniformity is deliberate.  It means

* one calibration procedure (block bootstrap on the calibration period, spec
  section 8) produces every threshold, so no detector can be tuned on test data;
* the false-alarm rate of the whole suite follows from the per-detector rates
  (with a Bonferroni-style correction over the number of statistics in the
  suite, see ``thresholds.py``);
* the attack-window class probabilities in L4 can use the same statistics as
  evidence.

All functions accept a 2-D array ``Z`` of shape (T, R): R independent replicates
in the columns.  R = 1 is the online path; R = 400 during bootstrap calibration.
The code is written once and used in both places, so calibration cannot drift
away from run time (a unit test asserts that the two paths agree exactly).

Gaps (NaN) are skipped, never filled: a window with n available samples uses n,
and the statistics are normalised by the effective sample size.  This keeps the
null distribution of the tests correct when a station reports 70 % of its days
(all West-African and Saharan stations in the bundle do).
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _as2d(Z: np.ndarray) -> np.ndarray:
    Z = np.asarray(Z, dtype=float)
    return Z[:, None] if Z.ndim == 1 else Z


def _prefix_nan(A: np.ndarray) -> np.ndarray:
    """Cumulative sums treating NaN as 0, with a leading zero row."""
    A0 = np.where(np.isfinite(A), A, 0.0)
    return np.vstack([np.zeros((1, A.shape[1])), np.cumsum(A0, axis=0)])


def _prefix_count(A: np.ndarray) -> np.ndarray:
    C = np.isfinite(A).astype(float)
    return np.vstack([np.zeros((1, A.shape[1])), np.cumsum(C, axis=0)])


def _window_diff(P: np.ndarray, W: int, T: int) -> np.ndarray:
    """P[t+1] - P[t+1-W] for t = W-1..T-1 -> shape (T-W+1, R)."""
    return P[W:] - P[:-W]


# --------------------------------------------------------------------------- #
# D1 / D2: sequential tests
# --------------------------------------------------------------------------- #


def cusum(Z: np.ndarray, k: float = 1.0, reset_on_miss: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Two-sided CUSUM on z with reference value ``k`` (in sigma units).

        S+_t = max(0, S+_{t-1} + z_t - k)
        S-_t = max(0, S-_{t-1} - z_t - k)

    Returns (S_up, S_dn), each of shape (T, R).  Missing days leave the state
    unchanged (the CUSUM is defined on the *evidence so far*, not on the
    calendar).  The statistic is not reset when a threshold is crossed: the
    alarm latches, which is what an operational pipeline needs, and the event
    counter (``event_onsets``) is what prevents one excursion from being counted
    twice.
    """
    Z = _as2d(Z)
    T, R = Z.shape
    sp = np.zeros((T, R))
    sn = np.zeros((T, R))
    up = np.zeros(R)
    dn = np.zeros(R)
    for t in range(T):
        z = Z[t]
        m = np.isfinite(z)
        up = np.where(m, np.maximum(0.0, up + np.where(m, z - k, 0.0)), up)
        dn = np.where(m, np.maximum(0.0, dn + np.where(m, -z - k, 0.0)), dn)
        sp[t] = up
        sn[t] = dn
    return sp, sn


def ewma(Z: np.ndarray, lam: float = 0.1) -> np.ndarray:
    """Standardised EWMA statistic |E_t| / sd(E), sd(E) = sqrt(lam/(2-lam))."""
    Z = _as2d(Z)
    T, R = Z.shape
    e = np.zeros(R)
    out = np.zeros((T, R))
    denom = np.sqrt(lam / (2.0 - lam))
    for t in range(T):
        z = Z[t]
        m = np.isfinite(z)
        e = np.where(m, lam * np.where(m, z, 0.0) + (1 - lam) * e, e)
        out[t] = np.where(m, np.abs(e) / denom, np.abs(e) / denom)
    return out


def window_mean(Z: np.ndarray, W: int) -> np.ndarray:
    """Standardised mean shift over W days: |sum z| / sqrt(n)  (n = available days).

    Under H0 (z iid standardised) this is the absolute value of a standard
    normal deviate, independent of the number of missing days in the window.
    """
    Z = _as2d(Z)
    T, R = Z.shape
    out = np.full((T, R), np.nan)
    if T < W:
        return out
    P1 = _prefix_nan(Z)
    n = _window_diff(_prefix_count(Z), W, T)
    s = _window_diff(P1, W, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        stat = np.abs(s) / np.sqrt(np.maximum(n, 1e-12))
    stat = np.where(n >= 2, stat, np.nan)
    out[W - 1:] = stat
    return out


def window_slope(Z: np.ndarray, W: int) -> np.ndarray:
    """Windowed OLS slope test for a slow drift.

    For the window [t-W+1, t] we fit z = a + b * i on the *available* samples
    and report |b| * sqrt(Sxx), where Sxx = sum (i - i_bar)^2.  Under H0 with
    unit-variance innovations, |b| * sqrt(Sxx) is the absolute value of a
    standard normal deviate (the usual t-statistic with sigma known), so the
    same bootstrap calibration applies and no parametric threshold is assumed.

    Implementation is prefix-sum based: O(T) memory, no loop over windows, and
    exact in the presence of gaps.
    """
    Z = _as2d(Z)
    T, R = Z.shape
    out = np.full((T, R), np.nan)
    if T < W:
        return out
    idx = np.arange(T, dtype=float)[:, None] * np.ones((1, R))
    finite = np.isfinite(Z)
    C = _prefix_count(Z)
    S1 = _prefix_nan(Z)
    # the index sums must run over the *available* samples only: mixing a count
    # of valid days (n) with index sums taken over all calendar slots makes the
    # determinant n*sii - si^2 negative whenever a window contains gaps, which
    # silently turns the whole statistic into NaN.  (Caught by
    # tests/test_detectors.py::test_window_slope_matches_bruteforce_with_gaps.)
    Si = _prefix_nan(np.where(finite, idx, np.nan))
    Sii = _prefix_nan(np.where(finite, idx ** 2, np.nan))
    Siz = _prefix_nan(idx * Z)

    n = _window_diff(C, W, T)
    s1 = _window_diff(S1, W, T)
    si = _window_diff(Si, W, T)
    sii = _window_diff(Sii, W, T)
    siz = _window_diff(Siz, W, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        det = n * sii - si ** 2
        b = (n * siz - si * s1) / det
        sxx = (sii - si ** 2 / n)
        stat = np.abs(b) * np.sqrt(np.maximum(sxx, 0.0))
    good = (n >= 5) & (det > 1e-9)
    stat = np.where(good, stat, np.nan)
    out[W - 1:] = stat
    return out


# --------------------------------------------------------------------------- #
# D3: replay (self-similarity)
# --------------------------------------------------------------------------- #


def window_distinct_counts(W: np.ndarray, quant: float = 0.1) -> np.ndarray:
    """Number of distinct quantised values in each row of a window matrix."""
    Q = np.round(np.asarray(W, dtype=float) / quant)
    Qs = np.sort(np.where(np.isfinite(Q), Q, np.nan), axis=1)
    with np.errstate(invalid="ignore"):
        changes = np.diff(Qs, axis=1) != 0
    return 1 + np.nansum(changes, axis=1)


def ncc_gram(raw: np.ndarray, L: int, chunk: int = 1024,
             min_distinct: int = 1) -> np.ndarray:
    """Normalised cross-correlation between all length-L windows of ``raw``.

    Returns G with G[i, j] = corr(raw[i:i+L], raw[j:j+L]).  Computed as a BLAS
    matrix product of the windowed matrix with itself, which is O(n^2 L) FLOPs
    but with a very small constant, and one FFT-free pass; for n = 1826, L = 7
    this takes a few milliseconds.

    G[k, j] = (d_{kj} - L mu_k mu_j) / (L s_k s_j),  d = <w_k, w_j>
    """
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    if n < L + 1:
        return np.zeros((0, 0))
    W = np.lib.stride_tricks.sliding_window_view(raw, L)          # (n-L+1, L)
    if not np.all(np.isfinite(W)):
        W = W.copy()
        W[~np.isfinite(W)] = np.nan
    import warnings

    with warnings.catch_warnings():          # all-NaN windows are legitimate: they mean "no data"
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(W, axis=1)
        sd = np.nanstd(W, axis=1, ddof=0)
    Wc = W - mu[:, None]
    Wc = np.where(np.isfinite(Wc), Wc, 0.0)
    G = np.empty((len(W), len(W)))
    for i in range(0, len(W), chunk):
        j = min(i + chunk, len(W))
        d = Wc[i:j] @ Wc.T
        with np.errstate(invalid="ignore", divide="ignore"):
            G[i:j] = d / (L * sd[i:j, None] * sd[None, :])
    G[~np.isfinite(G)] = np.nan
    bad = sd <= 1e-12
    if min_distinct > 1:
        # A window with only one or two distinct quantised values carries no
        # *shape*: any two dry weeks correlate perfectly after centring.  This is
        # the sparse-window degeneracy that made the precipitation replay
        # detector fire 12.8 times per sensor-year on clean data.
        bad = bad | (window_distinct_counts(W) < int(min_distinct))
    G[bad, :] = np.nan
    G[:, bad] = np.nan
    return np.clip(G, -1.0, 1.0)


def replay_max_ncc(raw: np.ndarray, L: int = 7, min_source_gap: int = 2,
                   min_distinct: int = 3) -> np.ndarray:
    """SP5/D3 statistic: max NCC of the window ending at t against earlier windows.

    The per-row prefix maximum is obtained with ``np.maximum.accumulate`` along
    the lag axis, so the whole statistic series costs one BLAS product plus one
    O(n^2) pass instead of n Python-level reductions.  NaN entries (constant
    windows) are mapped to -inf before the accumulation so that they do not
    propagate.
    """
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    out = np.full(n, np.nan)
    G = ncc_gram(raw, L, min_distinct=min_distinct)
    if G.size == 0:
        return out
    nw = G.shape[0]
    Gf = np.where(np.isfinite(G), G, -np.inf)
    P = np.maximum.accumulate(Gf, axis=1)          # P[k, j] = max over j' <= j of G[k, j']
    k_idx = np.arange(nw)
    j_idx = k_idx - min_source_gap
    ok = j_idx >= 0
    vals = np.full(nw, -np.inf)
    vals[ok] = P[k_idx[ok], j_idx[ok]]
    vals[~np.isfinite(vals)] = np.nan
    # window index k corresponds to the window starting at k, ending at k+L-1
    out[k_idx + L - 1] = vals
    return out


def block_contrast(Z: np.ndarray, L: int) -> np.ndarray:
    """Level contrast between the last L days and the L days before them.

    This is the matched filter for a *block replacement* of known length -- the
    replay attack, and any injection that is switched on and off again.  A replay
    of L days shows up as a rectangular pulse in the residual: the inside window
    disagrees with the present context while the outside window does not.

    Statistic:  T_t = (mean_in - mean_out) / sqrt(1/n_in + 1/n_out)

    which is a standard normal deviate under H0 (z iid, unit scale) regardless of
    how many days are missing in each window.  It is far more powerful than
    "the largest normalised correlation over all past windows" for L = 7,
    because it makes one comparison, not N-hundreds of them: the maximum
    correlation over the whole history has a null distribution whose 99th
    percentile is already ~0.99 for L = 7 on any autocorrelated series (measured
    in results/tables/table_detectability.csv), so the shape-matching channel
    alone cannot reach the sensitivity that matters at this window length.

    The contrast statistic also separates a replay (finite duration, contrast
    returns to noise level after L days) from a step (contrast stays high), which
    L4 uses as evidence.
    """
    Z = _as2d(Z)
    T, R = Z.shape
    out = np.full((T, R), np.nan)
    if T < 2 * L:
        return out
    P1 = _prefix_nan(Z)
    C = _prefix_count(Z)
    a1 = _window_diff(P1, L, T)          # sum over [t-L+1, t]      (T-L+1, R)
    n1 = _window_diff(C, L, T)
    # sum over [t-2L+1, t-L]
    s_in = a1[L:]
    n_in = n1[L:]
    s_out = a1[:-L]
    n_out = n1[:-L]
    with np.errstate(invalid="ignore", divide="ignore"):
        m_in = s_in / n_in
        m_out = s_out / n_out
        stat = (m_in - m_out) / np.sqrt(1.0 / n_in + 1.0 / n_out)
    good = (n_in >= 3) & (n_out >= 3)
    stat = np.where(good, np.abs(stat), np.nan)
    out[2 * L - 1:] = stat
    return out


def replay_exact(raw: np.ndarray, L: int = 7, quant: float = 0.1, min_source_gap: int = 2,
                 min_distinct: int = 3) -> np.ndarray:
    """Exact-repetition channel: 1.0 when the window repeats an earlier window.

    Uses integer quantisation and a hash of the window (O(n) overall).  This is
    the high-specificity channel: an exact repeat of a 7-day weather sequence to
    the decile is not something nature does.

    ``min_distinct`` excludes **degenerate windows** -- windows with fewer than
    this many distinct quantised values (constant or two-valued runs).  This is
    not a refinement, it is what makes the channel usable at all: for
    precipitation, seven consecutive dry days is a perfectly ordinary event and
    the corresponding window repeats itself constantly, so without this rule the
    exact-repeat detector fires throughout every dry spell on honest data
    (measured: it dominated the precipitation false-alarm rate, 13-15 alarms per
    sensor-year, while contributing almost nothing for temperature).  A constant
    run is the *stuck-at* fault's signature and is D4's job; detecting it here
    double-counted it as a replay.
    """
    raw = np.asarray(raw, dtype=float)
    n = len(raw)
    out = np.zeros(n)
    q = np.round(raw / quant)
    seen: dict[tuple, list[int]] = {}
    for t in range(L - 1, n):
        win = q[t - L + 1: t + 1]
        if not np.all(np.isfinite(win)):
            continue
        if np.unique(win).size < int(min_distinct):
            continue
        key = tuple(int(v) for v in win)
        starts = seen.get(key, [])
        start = t - L + 1
        if any(start - s >= min_source_gap for s in starts):
            out[t] = 1.0
        starts.append(start)
        seen[key] = starts[-8:]          # keep memory bounded; only recent starts matter
    return out


# --------------------------------------------------------------------------- #
# D4: faults
# --------------------------------------------------------------------------- #


def stuck_run(raw: np.ndarray, quant: float = 0.1) -> np.ndarray:
    """Length in days of the current run of identical quantised readings."""
    raw = np.asarray(raw, dtype=float)
    q = np.round(raw / quant)
    n = len(q)
    out = np.zeros(n)
    run = 1
    for t in range(n):
        if t == 0 or not np.isfinite(q[t]) or q[t] != q[t - 1]:
            run = 1 if np.isfinite(q[t]) else 0
        else:
            run += 1
        out[t] = run
    return out


# --------------------------------------------------------------------------- #
# event counting
# --------------------------------------------------------------------------- #


def event_onsets(exceed: np.ndarray, cooldown: int = 14) -> np.ndarray:
    """Count alarm *events* rather than alarm *samples* (Schmitt-trigger rule).

    An event is counted when the statistic **crosses** its threshold from below,
    provided (a) at least ``cooldown`` days have passed since the previous event
    and (b) the statistic has returned below the threshold at least once since
    then.  Condition (b) is the documented reset rule: a latching excursion is one
    event however long it lasts, and a fresh excursion after a quiet period is a
    new event.

    Without (b) a single sustained bias is re-counted every ``cooldown`` days.
    On real station data that inflated the measured false-alarm rate by up to 7x
    (JENDOUBA / TMAX) before it was fixed.  "Events, not samples" is what the
    specification asks for; this is the rule that makes the count mean it.

    Accepts 1-D or 2-D boolean arrays; returns one count per replicate when 2-D.
    """
    E = np.asarray(exceed, dtype=bool)
    squeeze = E.ndim == 1
    if squeeze:
        E = E[:, None]
    T, R = E.shape
    counts = np.zeros(R, dtype=int)
    last = np.full(R, -10 ** 9, dtype=int)
    armed = np.ones(R, dtype=bool)
    for t in range(T):
        e = E[t]
        armed = np.where(~e, True, armed)              # re-arm only after a sub-threshold day
        fire = e & armed & ((t - last) > cooldown)
        counts += fire
        last = np.where(fire, t, last)
        armed = np.where(fire, False, armed)           # latch until it comes back down
    return counts[0] if squeeze else counts


class SchmittAlarm:
    """The one definition of an alarm event (used by every code path).

    An *event* starts when the exceedance indicator rises while the detector is
    **armed**; becoming armed again requires the indicator to fall below the
    threshold first (the Schmitt trigger), and consecutive events are separated by
    at least ``cooldown`` days.  Events are counted, not samples (spec section 8).

    This lives in its own class, rather than being written twice, because the
    vectorised evaluation and the streaming pipeline *must* agree; a version that
    only enforced the cooldown produced a different event count for a persistent
    exceedance (one event in batch, one every cooldown days online), which is
    exactly the kind of silent divergence that makes reported numbers describe a
    system nobody runs.
    """

    __slots__ = ("cooldown", "last", "armed")

    def __init__(self, cooldown: int = 14):
        self.cooldown = int(cooldown)
        self.last = -10 ** 9
        self.armed = True

    def step(self, exceeded: bool, t: int) -> bool:
        """Feed one day; return True if this day starts a counted event."""
        if not exceeded:
            self.armed = True
            return False
        if self.armed and (t - self.last) > self.cooldown:
            self.last = t
            self.armed = False
            return True
        return False


def event_onsets_mask(exceed: np.ndarray, cooldown: int = 14) -> np.ndarray:
    """Boolean mask of the first day of each counted alarm event.

    Same Schmitt-trigger rule as :func:`event_onsets`, returning *where* the
    events are (used by the delay measurement, the response ladder and the impact
    window).
    """
    E = np.asarray(exceed, dtype=bool)
    out = np.zeros(E.shape, dtype=bool)
    trig = SchmittAlarm(cooldown)
    for t in range(len(E)):
        out[t] = trig.step(bool(E[t]), t)
    return out


def events_per_year(exceed: np.ndarray, n_days: int, cooldown: int = 14) -> float:
    c = event_onsets(np.asarray(exceed, dtype=bool), cooldown)
    return float(np.sum(c) * 365.25 / max(n_days, 1))
