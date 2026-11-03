"""D7: the availability channel (a *missing-data* fault, not a statistical detector).

The other detectors answer "does this value agree with an independent description
of the same day".  They are structurally blind to the failure mode in which the
value simply stops arriving: a dropout makes ``z`` NaN, so nothing exceeds any
threshold, and a silenced station looks exactly like a quiet one.  For a
false-data-injection exercise that is the worst possible blind spot, because
*withholding* data is a data-integrity attack with no statistics to catch it.

D7 therefore checks the reporting cadence itself.  The envelope is data-derived
in the same way as the stuck-at envelope (D4): it is 1.5x the longest gap
observed in the calibration period, so a station that routinely misses a week at
a time is not accused of an outage for missing a week, while a station that
reports every day is.  It needs no bootstrap, because there is no threshold to
calibrate on a continuous statistic -- the "statistic" is a run length of missing
days, and the false-alarm budget is expressed directly by the envelope.

The channel raises a *fault*, never an attack: an outage can be a dead battery,
a modem failure or a deliberate suppression, and the data cannot tell them apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIN_ENVELOPE_DAYS = 7


@dataclass
class AvailabilityMonitor:
    envelope_days: int = MIN_ENVELOPE_DAYS

    @classmethod
    def fit(cls, gap: np.ndarray, mask: np.ndarray | None = None,
            factor: float = 1.5, min_days: int = MIN_ENVELOPE_DAYS) -> "AvailabilityMonitor":
        g = np.asarray(gap, dtype=bool)
        if mask is not None:
            g = g & np.asarray(mask, dtype=bool)
        return cls(envelope_days=int(max(min_days, round(factor * _longest_run(g)))))

    def run_length_at(self, gap: np.ndarray) -> np.ndarray:
        """Length of the missing run ending at each day (0 on a reported day)."""
        g = np.asarray(gap, dtype=bool)
        out = np.zeros(len(g), dtype=np.int32)
        run = 0
        for i, missing in enumerate(g):
            run = run + 1 if missing else 0
            out[i] = run
        return out

    def exceed(self, gap: np.ndarray) -> np.ndarray:
        """True on the days where the missing run has passed the envelope."""
        return self.run_length_at(gap) > self.envelope_days

    def events(self, gap: np.ndarray, cooldown: int = 14) -> np.ndarray:
        """Alarm onsets (the first day the run passes the envelope), cooled down."""
        from .statistics import event_onsets_mask

        return event_onsets_mask(self.exceed(gap), cooldown)


def _longest_run(g: np.ndarray) -> int:
    best = run = 0
    for missing in g:
        run = run + 1 if missing else 0
        best = max(best, run)
    return int(best)
