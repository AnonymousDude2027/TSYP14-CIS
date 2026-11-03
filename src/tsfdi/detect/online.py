"""Day-by-day detector state.

The evaluation pipeline works on whole series with vectorised code (it has to:
tens of thousands of injections).  The operational pipeline receives one record
per day, so it needs incremental state.  Having two implementations is a classic
source of divergence -- a detector that is calibrated one way and run another
way is a bug with a plausible-looking output -- so this module is the *single*
incremental implementation, and ``tests/test_online_vs_batch.py`` asserts that it
reproduces the batch statistics of ``detect/statistics.py`` to 1e-9 on the same
data.

Cost is O(1) per day except the two window tests whose window is at most 30 days.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .. import config


@dataclass
class OnlineSuite:
    """Incremental statistics for one (site, variable) pair."""

    k_step: float = 1.0
    k_drift: float = 0.5
    lam: float = 0.1
    windows: tuple[int, ...] = (3, 7)
    slope_windows: tuple[int, ...] = (14, 30)
    L: int = 7
    sp: float = 0.0
    sn: float = 0.0
    sp2: float = 0.0
    sn2: float = 0.0
    ew: float = 0.0
    hist: deque = field(default_factory=lambda: deque(maxlen=64))

    @classmethod
    def from_config(cls) -> "OnlineSuite":
        cfg = config.load("detectors.yaml")["detectors"]
        return cls(
            k_step=float(cfg["D1_step_cusum"]["k_sigma"]),
            k_drift=float(cfg["D2_drift_cusum"]["k_sigma"]),
            lam=float(cfg["D2_ewma"]["lam"]),
            windows=tuple(int(w) for w in cfg["D1_window_shift"]["windows"]),
            slope_windows=tuple(int(w) for w in cfg["D2_slope"]["windows"]),
            L=int(cfg["D3_replay"]["lengths"][0]),
        )

    # ------------------------------------------------------------------ step
    def step(self, z: float) -> dict[str, float]:
        """Update the state with today's z and return today's statistics."""
        out: dict[str, float] = {}
        if not np.isfinite(z):
            # A missing day freezes the sequential tests (their state is the
            # evidence accumulated so far) but it *is* a calendar day inside every
            # rolling window, so the window statistics are still computed -- this
            # keeps the streaming path identical to the vectorised one.  No alarm
            # can be raised on such a day: that gate is applied by the caller
            # (``SiteVariableModel.alarms`` and ``StreamingPipeline``).
            self.hist.append(np.nan)
            out.update(self._window_stats())
            out["D4_spike"] = np.nan
            return out
        self.sp = max(0.0, self.sp + z - self.k_step)
        self.sn = max(0.0, self.sn - z - self.k_step)
        self.sp2 = max(0.0, self.sp2 + z - self.k_drift)
        self.sn2 = max(0.0, self.sn2 - z - self.k_drift)
        self.ew = self.lam * z + (1 - self.lam) * self.ew
        self.hist.append(float(z))

        out["D1_step_cusum"] = max(self.sp, self.sn)
        out["D2_drift_cusum"] = max(self.sp2, self.sn2)
        out["D2_ewma"] = abs(self.ew) / np.sqrt(self.lam / (2 - self.lam))
        out["D4_spike"] = abs(z)

        out.update(self._window_stats())
        return out

    def _window_stats(self) -> dict[str, float]:
        """Rolling-window statistics over the last W calendar days."""
        out: dict[str, float] = {}
        H = np.fromiter(self.hist, dtype=float, count=len(self.hist))
        for W in self.windows:
            win = H[-W:]
            n = int(np.isfinite(win).sum())
            out[f"D1_window_shift_{W}"] = abs(np.nansum(win)) / np.sqrt(n) if n >= 2 else np.nan
        for W in self.slope_windows:
            win = H[-W:]
            ok = np.isfinite(win)
            if ok.sum() >= 5:
                i = np.arange(len(win), dtype=float)[ok]
                v = win[ok]
                n = len(v)
                si, sii, sz, siz = i.sum(), (i ** 2).sum(), v.sum(), (i * v).sum()
                det = n * sii - si ** 2
                if det > 1e-9:
                    b = (n * siz - si * sz) / det
                    sxx = sii - si ** 2 / n
                    out[f"D2_slope_{W}"] = abs(b) * np.sqrt(max(sxx, 0.0))
                else:
                    out[f"D2_slope_{W}"] = np.nan
            else:
                out[f"D2_slope_{W}"] = np.nan
        if len(H) >= 2 * self.L:
            a = H[-self.L:]
            b = H[-2 * self.L:-self.L]
            na, nb = int(np.isfinite(a).sum()), int(np.isfinite(b).sum())
            if na >= 3 and nb >= 3:
                out["D3_block_contrast"] = abs(np.nanmean(a) - np.nanmean(b)) / np.sqrt(1.0 / na + 1.0 / nb)
            else:
                out["D3_block_contrast"] = np.nan
        else:
            out["D3_block_contrast"] = np.nan
        return out

    # --------------------------------------------------------------- helpers
    def alarm(self, stats: dict[str, float], thresholds: dict[str, float]) -> dict[str, bool]:
        out = {}
        for k, v in stats.items():
            h = thresholds.get(k, np.inf)
            out[k] = bool(np.isfinite(v) and np.isfinite(h) and v > h)
        return out

    def clone(self) -> "OnlineSuite":
        s = OnlineSuite(**{k: getattr(self, k) for k in
                           ("k_step", "k_drift", "lam", "windows", "slope_windows", "L")})
        s.sp, s.sn, s.sp2, s.sn2, s.ew = self.sp, self.sn, self.sp2, self.sn2, self.ew
        s.hist = deque(self.hist, maxlen=self.hist.maxlen)
        return s
