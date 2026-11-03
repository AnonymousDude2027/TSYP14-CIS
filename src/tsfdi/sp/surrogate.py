"""Surrogate data for calibrating the replay and fingerprint detectors.

Why a *different* null model is needed for D3 and D6.

For the sequential tests (D1, D2) the null hypothesis is "this series is a
realisation of the clean data-generating process".  A circular block bootstrap
is a good approximation of that null: it keeps 30-day stretches intact, so the
synoptic structure survives, and it produces a new trajectory for the CUSUM.

For the replay detector that null is *wrong*, and disastrously so.  A block
bootstrap draws blocks with replacement, so a block is often drawn twice, which
places an exact copy of a 30-day stretch elsewhere in the series.  The matched
filter then finds a genuine 1.000 correlation -- not because an attacker acted,
but because the resampling created one.  Calibrating the replay threshold on
such replicates pushes the threshold to ~1.0 and makes the detector blind: the
false-alarm rate is controlled by destroying the very sensitivity we want.

The correct null for "no repeated subsequence" is a series with the **same
second-order structure and no repetition**.  That is exactly what a
phase-randomised Fourier surrogate provides: keep the amplitude spectrum,
randomise the phases (uniform on [0, 2pi) for the interior bins, keeping the
DC and Nyquist coefficients real so that the inverse transform is real), and
the autocovariance is preserved in expectation while any phase-locked,
repeated structure is destroyed.

For a *pair* of series (sensor, reference) the same phase vector is applied to
both, which preserves the cross-spectrum and therefore the coherence -- the
feature D6 monitors -- while destroying the specific realisations.

References for the method (both standard): Theiler et al., "Testing for
nonlinearity in time series: the method of surrogate data", Physica D 58 (1992)
77-94; Schreiber & Schmitz, "Surrogate time series", Physica D 142 (2000)
346-382.
"""
from __future__ import annotations

import numpy as np


def phase_randomize(x: np.ndarray, rng: np.random.Generator, phases: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return (surrogate, phases) of ``x``; reuse ``phases`` to keep two series paired."""
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    filled = x.copy()
    if not ok.all():
        idx = np.arange(len(x))
        filled[~ok] = np.interp(idx[~ok], idx[ok], x[ok]) if ok.any() else 0.0
    n = len(filled)
    X = np.fft.rfft(filled)
    if phases is None:
        nb = len(X)
        phases = np.exp(2j * np.pi * rng.random(nb))
        phases[0] = 1.0                       # DC stays real
        if n % 2 == 0:
            phases[-1] = 1.0                  # Nyquist stays real
    Xs = np.abs(X) * phases
    out = np.fft.irfft(Xs, n=n)
    out = out - out.mean() + filled.mean()    # restore the mean level
    out[~ok] = np.nan
    return out, phases


def surrogate_pair(y: np.ndarray, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Joint surrogate of a sensor/reference pair (preserves the cross-spectrum)."""
    ys, ph = phase_randomize(y, rng)
    xs, _ = phase_randomize(x, rng, phases=ph)
    return ys, xs


def surrogate_replicates(values: np.ndarray, n_rep: int, rng: np.random.Generator,
                         paired: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Stack ``n_rep`` surrogates column-wise: shape (n, n_rep)."""
    cols = []
    cols2 = [] if paired is not None else None
    for _ in range(n_rep):
        if paired is None:
            s, _ = phase_randomize(values, rng)
        else:
            s, ph = phase_randomize(values, rng)
            if cols2 is not None:
                cols2.append(phase_randomize(paired, rng, phases=ph)[0])
        cols.append(s)
    return np.column_stack(cols), (np.column_stack(cols2) if cols2 else None)
