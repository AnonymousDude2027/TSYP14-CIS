"""SP4 multi-scale analysis: the discrete wavelet transform of ``z``.

An injected step and a natural warm spell both raise the level of ``z``; they
differ in *where* in the scale hierarchy the energy appears.  The DWT (Daubechies-4,
5 levels on daily data) separates:

  * a step: detail coefficients at every level are large at the *same instant*,
    and the coarse approximation jumps by the step size;
  * a ramp: almost no fine-scale energy, growing coarse-scale energy;
  * a spike: fine-scale energy only, one coefficient;
  * a slow natural spell: coarse-scale energy spread over many coefficients,
    no fine-scale energy.

The features extracted here are *diagnostic* (L4), not detection statistics:
they are only evaluated once an alarm has fired.  They are still part of the
ablation, because the ablation has to justify their presence.

Gaps are handled by linear interpolation **inside this diagnostic stage only**
(a DWT needs a complete dyadic grid); the interpolated samples are flagged and
the affected windows are reported as "low confidence" rather than silently
trusted.  No interpolated sample is ever fed back into a detector.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_WAVELET = "db4"
DEFAULT_LEVEL = 5


def _fill_for_dwt(z: np.ndarray) -> tuple[np.ndarray, float]:
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(z)
    frac_missing = float(1.0 - ok.mean()) if len(z) else 0.0
    if ok.all():
        return z, frac_missing
    if ok.sum() < 4:
        return np.zeros_like(z), 1.0
    idx = np.arange(len(z))
    out = z.copy()
    out[~ok] = np.interp(idx[~ok], idx[ok], z[ok])
    return out, frac_missing


@dataclass
class WaveletFeatures:
    level_energy: np.ndarray      # relative energy in details D1..D5 (fraction of total)
    approx_energy: float
    step_score: float             # co-located energy across levels, in units of the calibration scale
    ramp_score: float             # coarse-only energy growth
    spike_score: float
    frac_missing: float
    onset_index: int | None       # index of the strongest co-located fine-scale event

    def as_dict(self) -> dict:
        d = {f"dwt_d{i + 1}_energy": float(v) for i, v in enumerate(self.level_energy)}
        d.update(
            dwt_approx_energy=float(self.approx_energy),
            dwt_step_score=float(self.step_score),
            dwt_ramp_score=float(self.ramp_score),
            dwt_spike_score=float(self.spike_score),
            dwt_frac_missing=float(self.frac_missing),
        )
        return d


class WaveletAnalyzer:
    """SP4: DWT of ``z`` and the shape descriptors derived from it."""

    def __init__(self, wavelet: str = DEFAULT_WAVELET, level: int = DEFAULT_LEVEL):
        self.wavelet = wavelet
        self.level = level
        self.ref_scale: dict[str, float] = {}

    def decompose(self, z: np.ndarray) -> dict[str, np.ndarray]:
        import pywt

        filled, _ = _fill_for_dwt(z)
        lvl = min(self.level, int(np.floor(np.log2(max(len(filled), 8)))) - 1)
        lvl = max(lvl, 1)
        coeffs = pywt.wavedec(filled, self.wavelet, level=lvl, mode="symmetric")
        out = {"approx": coeffs[0]}
        # coeffs[1] is the coarsest detail; we want D1 (finest) first
        for i, d in enumerate(coeffs[1:][::-1]):
            out[f"d{i + 1}"] = d
        return out

    def calibrate(self, z_cal: np.ndarray) -> "WaveletAnalyzer":
        """Robust scale of each descriptor on calibration data."""
        feats = self.features(z_cal)
        from .expectation import robust_scale

        self.ref_scale = {
            "step": robust_scale(np.array([feats.step_score])) or 1.0,
            "ramp": robust_scale(np.array([feats.ramp_score])) or 1.0,
            "spike": robust_scale(np.array([feats.spike_score])) or 1.0,
        }
        # single-window robust scales are degenerate; use a multi-window estimate
        seg = self._segment_scores(z_cal)
        for k, v in seg.items():
            self.ref_scale[k] = max(float(np.median(v)), 1e-9)
        return self

    def _segment_scores(self, z: np.ndarray, w: int = 60) -> dict[str, list[float]]:
        out = {"step": [], "ramp": [], "spike": []}
        for i in range(0, max(len(z) - w, 1), max(w // 2, 1)):
            f = self.features(z[i: i + w], normalise=False)
            out["step"].append(f.step_score)
            out["ramp"].append(f.ramp_score)
            out["spike"].append(f.spike_score)
        return out

    def features(self, z: np.ndarray, normalise: bool = True) -> WaveletFeatures:
        z = np.asarray(z, dtype=float)
        _, frac_missing = _fill_for_dwt(z)
        dec = self.decompose(z)
        details = [dec[f"d{i}"] for i in range(1, self.level + 1) if f"d{i}" in dec]
        energy = np.array([float(np.sum(d ** 2)) for d in details])
        total = float(energy.sum() + np.sum(dec["approx"] ** 2))
        rel = energy / total if total > 0 else np.zeros_like(energy)

        # --- step score: max over time of the product of |D1| and |D3| (both
        # must be large at the same instant for a genuine discontinuity)
        d1 = np.abs(details[0]) if details else np.zeros(1)
        d3 = np.abs(details[min(2, len(details) - 1)]) if details else np.zeros(1)
        m = min(len(d1), len(d3))
        if m < 2:
            step = 0.0
        else:
            a, b = d1[:m], d3[:m]
            step = float(np.max(np.sqrt(a * b)))
        onset = int(np.argmax(np.sqrt(d1[:m] * d3[:m]))) if m >= 2 else None

        # --- ramp score: coarse detail energy relative to fine detail energy
        fine = float(np.sum(energy[:2]))
        coarse = float(np.sum(energy[-2:])) if len(energy) >= 2 else 0.0
        ramp = coarse / (fine + 1e-12)
        # --- spike score: single-coefficient dominance at level 1
        spike = float(np.max(d1) / (np.sqrt(np.mean(d1 ** 2)) + 1e-12)) if d1.size else 0.0

        if normalise and self.ref_scale:
            step /= self.ref_scale.get("step", 1.0)
            ramp /= self.ref_scale.get("ramp", 1.0) if self.ref_scale.get("ramp") else 1.0
            spike /= self.ref_scale.get("spike", 1.0)
        return WaveletFeatures(
            level_energy=rel,
            approx_energy=float(np.sum(dec["approx"] ** 2) / total) if total > 0 else 0.0,
            step_score=float(step),
            ramp_score=float(ramp),
            spike_score=float(spike),
            frac_missing=frac_missing,
            onset_index=onset,
        )


def onset_shape(z: np.ndarray) -> str:
    """Classify the onset shape of an alarm window (diagnostic label)."""
    wa = WaveletAnalyzer()
    f = wa.features(z, normalise=False)
    rel = f.level_energy
    if len(rel) >= 3 and rel[:2].sum() > 0.5 * rel.sum() and f.spike_score > 6:
        return "spike"
    if len(rel) >= 3 and rel[-2:].sum() > rel[:2].sum() * 2:
        return "ramp"
    if f.step_score > 0:
        return "step"
    return "unclear"
