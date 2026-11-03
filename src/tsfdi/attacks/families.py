"""Attack, fault and natural-event generation (spec section 9).

Every generator has the same signature::

    inject(series, spec, rng) -> (corrupted_values, label_vector)

is fully seeded, never looks at the test period when it is being calibrated, and
returns a label vector so that the evaluation can score per day without any
ambiguity about where the truth is.

Magnitude conventions (the specification's percentages are ambiguous, section
18 item 3, so both are produced and both are reported):

* temperature: ``percent`` is taken over the site's *seasonal range* (the
  peak-to-trough amplitude of the fitted climatology), because "12 % of 31 degC"
  has no physical meaning; the sigma-equivalent is reported alongside.
* precipitation: ``percent`` is a multiplicative factor on the value, which is
  the only sensible reading for a non-negative, zero-inflated variable.

A random sign is used for the bias attacks (seeded), because a detector that only
finds warm biases is not a detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import config


@dataclass
class AttackSpec:
    family: str                  # step | ramp | replay | scaling | noise_shaped
                                 # stuck_at | dropout | spike | calibration_drift
    onset: int                   # index in the series
    duration: int = 120
    sign: int = +1
    magnitude_sigma: float = 3.0     # bias in units of the robust residual scale
    magnitude_percent: float = float("nan")
    # family specific
    rise_days: int = 90
    length: int = 7                  # replay length
    source_offset: int = 60          # replay source distance in days
    gain: float = 1.0
    noise_sigma: float = 0.0
    ar1: float = 0.7
    amplitude_sigma: float = 1.0
    label: str = ""

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class InjectionResult:
    values: np.ndarray            # corrupted sensor values (NaN where missing)
    labels: np.ndarray            # bool, True on days affected by the injection
    delta: np.ndarray             # signed bias actually added, in data units
    spec: AttackSpec
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# magnitude conversion
# --------------------------------------------------------------------------- #


def gain_per_sigma() -> float:
    """Fractional gain that one sigma of the grid axis means on a multiplicative variable."""
    try:
        return float(config.load("attacks.yaml")["conventions"]["multiplicative_gain_per_sigma"])
    except Exception:  # noqa: BLE001 - config is optional for library use
        return 0.2


def _nominal_sigma(spec: AttackSpec, kind: str, seasonal_range: float = np.nan,
                   sigma: float = np.nan) -> float:
    """The magnitude on the *nominal* axis, i.e. the point of the grid this spec is from.

    Reporting the nominal value (rather than re-measuring it from the realised delta)
    keeps the axis exact: a 90-day ramp reaches 1-1/rise of its nominal peak, so a
    measured value would split one grid point over two labels.  The realised value is
    reported separately as ``magnitude_units`` and checked in the tests.
    """
    if np.isfinite(spec.magnitude_sigma):
        return float(spec.magnitude_sigma)
    if np.isfinite(spec.magnitude_percent):
        if kind == "multiplicative":
            return float(spec.magnitude_percent) / 100.0 / gain_per_sigma()
        # a percentage of the seasonal range, expressed in sigma
        if np.isfinite(seasonal_range) and np.isfinite(sigma) and sigma:
            return float(spec.magnitude_percent / 100.0 * seasonal_range / sigma)
        return float("nan")
    if spec.gain != 1.0:
        return float(abs(spec.gain) / gain_per_sigma())
    return float("nan")


def magnitude_in_units(spec: AttackSpec, kind: str, seasonal_range: float,
                       sigma: float, values: np.ndarray | None = None) -> float:
    """Convert the requested magnitude into the units the family applies it in.

    Additive variables: data units, ``m * sigma`` (or ``percent/100 * seasonal_range``).
    Multiplicative variables: a *fractional gain*, ``m * gain_per_sigma`` -- see
    ``config/attacks.yaml`` for why the sigma axis is reinterpreted there.
    """
    if kind == "multiplicative":
        if np.isfinite(spec.magnitude_percent):
            return float(spec.magnitude_percent / 100.0)
        if np.isfinite(spec.magnitude_sigma):
            return float(spec.magnitude_sigma * gain_per_sigma())
        return float(spec.gain)
    if np.isfinite(spec.magnitude_percent) and not np.isfinite(spec.magnitude_sigma):
        return float(spec.magnitude_percent / 100.0 * seasonal_range)
    return float(spec.magnitude_sigma * sigma)


def to_working(v: np.ndarray, kind: str) -> np.ndarray:
    """SP1 coordinate (square root for a multiplicative variable), mirrored here.

    The injector has to speak the same coordinate as the detector, otherwise an
    "m-sigma bias" means a different physical change for rain than for
    temperature: one sigma of the precipitation residual is 0.19 mm^0.5, not
    0.19 mm, and adding the latter as a raw offset would be a ~0.04 sigma attack.
    """
    v = np.asarray(v, dtype=float)
    return np.sqrt(np.clip(v, 0.0, None)) if kind == "multiplicative" else v


def from_working(w: np.ndarray, kind: str) -> np.ndarray:
    w = np.asarray(w, dtype=float)
    return np.clip(w, 0.0, None) ** 2 if kind == "multiplicative" else w


def _realised_sigma(delta: np.ndarray, values: np.ndarray, a: int, b: int, kind: str,
                    sigma: float) -> float:
    """Effect size actually achieved, in units of sigma, in the SP1 coordinate.

    This is the number that makes families comparable: a 60 % gain on rain and a
    2 sigma offset are not the same thing, and only the realised effect says by
    how much they differ.
    """
    if b <= a or not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    dw = to_working(values[a:b] + delta[a:b], kind) - to_working(values[a:b], kind)
    dw = dw[np.isfinite(dw)]
    return float(np.abs(dw).mean() / sigma) if dw.size else float("nan")


def _nan_peak(delta: np.ndarray, a: int, b: int) -> float:
    """Largest |change| in the window, ignoring the missing days (which are NaN there)."""
    if b <= a:
        return 0.0
    w = np.abs(np.asarray(delta[a:b], dtype=float))
    w = w[np.isfinite(w)]
    return float(w.max()) if w.size else 0.0


def _nan_mean_abs(delta: np.ndarray, a: int, b: int) -> float:
    if b <= a:
        return 0.0
    w = np.abs(np.asarray(delta[a:b], dtype=float))
    w = w[np.isfinite(w)]
    return float(w.mean()) if w.size else 0.0


def percent_of_range_to_sigma(percent: float, seasonal_range: float, sigma: float) -> float:
    return float(percent / 100.0 * seasonal_range / sigma) if sigma else float("nan")


# --------------------------------------------------------------------------- #
# families
# --------------------------------------------------------------------------- #


def _window(spec: AttackSpec, n: int) -> tuple[int, int]:
    a = int(np.clip(spec.onset, 0, n))
    b = int(np.clip(spec.onset + spec.duration, 0, n))
    return a, b


def step(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
         sigma: float, rng: np.random.Generator) -> InjectionResult:
    a, b = _window(spec, len(values))
    mag = magnitude_in_units(spec, kind, seasonal_range, sigma)
    delta = np.zeros_like(values)
    if kind == "multiplicative" and np.isfinite(spec.magnitude_sigma):
        # grid axis: an offset of m sigma in the SP1 coordinate, which is a
        # wetness-dependent change in millimetres (see to_working)
        w = to_working(values, kind)
        shifted = w[a:b] + float(spec.magnitude_sigma) * sigma * spec.sign
        delta[a:b] = from_working(shifted, kind) - values[a:b]
    elif kind == "multiplicative":
        delta[a:b] = np.abs(values[a:b]) * mag * spec.sign     # a factor (profile)
    else:
        delta[a:b] = mag * spec.sign
    out = values + delta
    lab = np.zeros(len(values), bool)
    lab[a:b] = True
    # NB: ``abs(delta.max())`` is wrong when the sign is negative -- the untouched
    # days are 0, so the maximum of the delta is 0 and every negative-signed step
    # recorded a magnitude of zero (and a NaN whenever the window contained a
    # missing day).  Use the nan-safe maximum of |delta| over the window.
    return InjectionResult(out, lab, delta, spec,
                           {"magnitude_units": _nan_mean_abs(delta, a, b),
                            "magnitude_peak_units": _nan_peak(delta, a, b),
                            "magnitude_sigma": _nominal_sigma(spec, kind, seasonal_range, sigma),
                            "magnitude_sigma_realised": _realised_sigma(delta, values, a, b, kind, sigma),
                            "gain": float(mag) if kind == "multiplicative" and np.isfinite(spec.magnitude_percent) else np.nan})


def ramp(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
         sigma: float, rng: np.random.Generator) -> InjectionResult:
    a, b = _window(spec, len(values))
    mag = magnitude_in_units(spec, kind, seasonal_range, sigma)
    n = b - a
    prof = np.zeros(n)
    rise = min(spec.rise_days, n)
    prof[:rise] = np.linspace(0, 1, rise, endpoint=False)
    if n > rise:
        prof[rise:] = 1.0
    delta = np.zeros_like(values)
    if kind == "multiplicative":
        w = to_working(values, kind)
        if np.isfinite(spec.magnitude_sigma):
            off = float(spec.magnitude_sigma) * sigma
            delta[a:b] = from_working(w[a:b] + prof * off * spec.sign, kind) - values[a:b]
        else:
            delta[a:b] = prof * mag * spec.sign * np.abs(values[a:b])
    else:
        delta[a:b] = prof * mag * spec.sign
    lab = np.zeros(len(values), bool)
    lab[a:b] = prof > 0
    return InjectionResult(values + delta, lab, delta, spec,
                           {"magnitude_units": _nan_mean_abs(delta, a, b),
                            "magnitude_peak_units": _nan_peak(delta, a, b),
                            "magnitude_sigma": _nominal_sigma(spec, kind, seasonal_range, sigma),
                            "magnitude_sigma_realised": _realised_sigma(delta, values, a, b, kind, sigma),
                            "gain": float(mag) if kind == "multiplicative" and np.isfinite(spec.magnitude_percent) else np.nan,
                            "rise_days": int(rise)})


def replay(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
           sigma: float, rng: np.random.Generator) -> InjectionResult:
    """Replace L consecutive readings by L earlier readings of the same sensor."""
    n = len(values)
    a = int(np.clip(spec.onset, 0, n))
    L = int(spec.length)
    src = a - int(spec.source_offset)
    if src < 0 or a + L > n:
        raise ValueError("replay window or source out of range")
    block = np.array(values[src: src + L], dtype=float)
    if spec.gain != 1.0:
        block = block * spec.gain
    if spec.noise_sigma:
        block = block + rng.normal(0, spec.noise_sigma * sigma, L)
    out = values.copy()
    delta = np.zeros_like(values)
    out[a: a + L] = block
    delta[a: a + L] = block - values[a: a + L]
    lab = np.zeros(n, bool)
    lab[a: a + L] = True
    return InjectionResult(out, lab, delta, spec,
                           {"length": L, "source_offset": int(spec.source_offset),
                            "gain": spec.gain, "noise_sigma": spec.noise_sigma,
                            "magnitude_units": _nan_mean_abs(delta, a, a + L),
                            # the natural axis of a replay is its length, not a magnitude
                            "magnitude_sigma": np.nan})


def scaling(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
            sigma: float, rng: np.random.Generator) -> InjectionResult:
    """Multiplicative gain over the whole window (a miscalibration-shaped attack).

    On the sigma axis the gain is chosen *from the data* so that the realised change
    matches the nominal one.  A fixed 10 % gain is a 3 degC bias at a warm station
    and a 0.6 degC bias at a cold one, so a nominal grid expressed in sigma would
    otherwise hide a factor-of-five spread between sites.

    * additive variable: ``Delta = g y``, so ``mean|Delta| = |g| mean|y|`` and
      ``g = m sigma / mean|y|``;
    * multiplicative variable (square-root coordinate): ``Delta_w =
      sqrt(y(1+g)) - sqrt(y) = (sqrt(1+g) - 1) sqrt(y)``, so
      ``g = (1 + m sigma / mean sqrt(y))^2 - 1``.

    The percentage form is still available (``magnitude_percent``) and is what the
    organizers' profile uses.
    """
    a, b = _window(spec, len(values))
    if np.isfinite(spec.magnitude_sigma) and not np.isfinite(spec.magnitude_percent):
        if kind == "multiplicative":
            w = to_working(values, kind)[a:b]
            w = w[np.isfinite(w)]
            ref = float(w.mean()) if w.size else np.nan
            g = (float((1.0 + spec.magnitude_sigma * sigma / ref) ** 2 - 1.0)
                 if np.isfinite(ref) and ref > 0 else 0.0)
        else:
            v = np.abs(values[a:b]).astype(float)
            v = v[np.isfinite(v)]
            ref = float(v.mean()) if v.size else np.nan
            g = float(spec.magnitude_sigma * sigma / ref) if np.isfinite(ref) and ref > 0 else 0.0
    else:
        g = magnitude_in_units(spec, "multiplicative", seasonal_range, sigma)
    delta = np.zeros_like(values)
    delta[a:b] = values[a:b] * g * spec.sign
    lab = np.zeros(len(values), bool)
    lab[a:b] = True
    return InjectionResult(values + delta, lab, delta, spec,
                           {"gain": float(g * spec.sign),
                            "magnitude_units": _nan_mean_abs(delta, a, b),
                            "magnitude_sigma": _nominal_sigma(spec, kind, seasonal_range, sigma),
                            "magnitude_sigma_realised": _realised_sigma(delta, values, a, b, kind, sigma)})


def noise_shaped(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
                 sigma: float, rng: np.random.Generator) -> InjectionResult:
    """Add temporally correlated noise: preserves the marginal spectrum shape.

    The amplitude is the grid magnitude when one is given (``magnitude_sigma``), so
    that this family is evaluated on the same sigma axis as the others; the AR(1)
    structure makes the *instantaneous* bias small while the *sustained* excursion
    is what a drift detector can see, which is the point of the family.
    """
    a, b = _window(spec, len(values))
    n = b - a
    m = _nominal_sigma(spec, kind, seasonal_range, sigma)
    if not np.isfinite(m):
        m = float(spec.amplitude_sigma)
    e = np.zeros(n)
    for i in range(n):
        e[i] = spec.ar1 * (e[i - 1] if i else 0.0) + rng.normal(0, np.sqrt(1 - spec.ar1 ** 2))
    delta = np.zeros_like(values)
    if kind == "multiplicative":
        w = to_working(values, kind)
        delta[a:b] = from_working(w[a:b] + e * m * sigma, kind) - values[a:b]
    else:
        delta[a:b] = e * m * sigma
    lab = np.zeros(len(values), bool)
    lab[a:b] = True
    return InjectionResult(values + delta, lab, delta, spec,
                           {"ar1": spec.ar1, "amplitude_sigma": float(m),
                            "magnitude_units": _nan_mean_abs(delta, a, b),
                            "magnitude_sigma": _nominal_sigma(spec, kind, seasonal_range, sigma),
                            "magnitude_sigma_realised": _realised_sigma(delta, values, a, b, kind, sigma)})


def stuck_at(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
             sigma: float, rng: np.random.Generator) -> InjectionResult:
    a, b = _window(spec, len(values))
    out = values.copy()
    delta = np.zeros_like(values)
    if a > 0:
        out[a:b] = values[a - 1]
        delta[a:b] = out[a:b] - values[a:b]
    lab = np.zeros(len(values), bool)
    lab[a:b] = True
    return InjectionResult(out, lab, delta, spec, {})


def dropout(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
            sigma: float, rng: np.random.Generator) -> InjectionResult:
    a, b = _window(spec, len(values))
    out = values.copy()
    out[a:b] = np.nan
    delta = np.zeros_like(values)
    lab = np.zeros(len(values), bool)
    lab[a:b] = True
    return InjectionResult(out, lab, delta, spec, {})


def spike(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
          sigma: float, rng: np.random.Generator) -> InjectionResult:
    a, b = _window(spec, len(values))
    delta = np.zeros_like(values)
    delta[a:b] = spec.amplitude_sigma * sigma * spec.sign
    out = values + delta
    lab = np.zeros(len(values), bool)
    lab[a:b] = True
    return InjectionResult(out, lab, delta, spec,
                           {"amplitude_sigma": spec.amplitude_sigma,
                            "magnitude_units": _nan_peak(delta, a, b),
                            "magnitude_sigma": _nominal_sigma(spec, kind, seasonal_range, sigma)})


def calibration_drift(values: np.ndarray, spec: AttackSpec, kind: str, seasonal_range: float,
                      sigma: float, rng: np.random.Generator) -> InjectionResult:
    """Benign slow drift of the sensor calibration (labelled fault, not attack)."""
    n = len(values)
    a = int(np.clip(spec.onset, 0, n))
    b = int(np.clip(spec.onset + spec.rise_days, 0, n))
    delta = np.zeros_like(values)
    if b > a:
        delta[a:b] = np.linspace(0, spec.magnitude_sigma * sigma * spec.sign, b - a, endpoint=False)
    lab = np.zeros(n, bool)
    lab[a:b] = True
    return InjectionResult(values + delta, lab, delta, spec,
                           {"magnitude_units": _nan_peak(delta, a, b),
                            "magnitude_sigma": _nominal_sigma(spec, kind, seasonal_range, sigma)})


FAMILIES = {
    "step": step, "ramp": ramp, "replay": replay, "scaling": scaling,
    "noise_shaped": noise_shaped, "stuck_at": stuck_at, "dropout": dropout,
    "spike": spike, "calibration_drift": calibration_drift,
}


# --------------------------------------------------------------------------- #
# injector
# --------------------------------------------------------------------------- #


class Injector:
    """Build and apply injections for one (site, variable) pair."""

    def __init__(self, sigma: float, seasonal_range: float, kind: str, n: int,
                 test_start: int | None = None, seed: int | None = None):
        self.sigma = float(sigma)
        self.seasonal_range = float(seasonal_range)
        self.kind = kind
        self.n = int(n)
        self.seed = config.SEED if seed is None else int(seed)
        cfg = config.load("attacks.yaml")["onset"]
        self.left = int(cfg["left_margin"])
        self.right = int(cfg["right_margin"])
        self.min_duration = int(cfg["min_duration_days"])
        self.test_start = test_start if test_start is not None else 0

    def draw_onset(self, rng: np.random.Generator, duration: int) -> int:
        lo = self.test_start + self.left
        hi = self.n - self.right - duration
        if hi <= lo:
            hi = max(lo + 1, self.n - duration - 1)
        return int(rng.integers(lo, max(hi, lo + 1)))

    def make_spec(self, family: str, rng: np.random.Generator, *, onset: int | None = None,
                  duration: int | None = None, magnitude_sigma: float | None = None,
                  magnitude_percent: float | None = None, sign: int | None = None,
                  length: int = 7, source_offset: int | None = None,
                  gain: float = 1.0, noise_sigma: float = 0.0) -> AttackSpec:
        fcfg = config.load("attacks.yaml")["families"]
        fam_cfg = {**fcfg["attack"], **fcfg["fault"]}.get(family, {})
        dur = duration or fam_cfg.get("duration_days", 120)
        if family == "ramp":
            dur = duration or int(fam_cfg.get("rise_days", 90) + fam_cfg.get("hold_days", 60))
        if family == "replay":
            dur = length
        if onset is None:
            onset = self.draw_onset(rng, int(dur))
        if sign is None:
            s = fam_cfg.get("sign", "random")
            sign = 0 if s == "none" else int(rng.choice([-1, 1]))
        if source_offset is None:
            so = config.load("attacks.yaml")["families"]["attack"]["replay"]
            source_offset = int(rng.integers(so["source_offset_min"], so["source_offset_max"]))
        return AttackSpec(
            family=family, onset=int(onset), duration=int(dur), sign=int(sign),
            magnitude_sigma=float("nan") if magnitude_sigma is None else float(magnitude_sigma),
            magnitude_percent=float("nan") if magnitude_percent is None else float(magnitude_percent),
            rise_days=int(fam_cfg.get("rise_days", 90)), length=int(length),
            source_offset=int(source_offset), gain=float(gain), noise_sigma=float(noise_sigma),
            ar1=float(fam_cfg.get("ar1", 0.7)),
            # a family that is parameterised by an amplitude (spike, noise_shaped)
            # must take that amplitude from the caller, otherwise "inject a 5 sigma
            # spike" silently injects the configured default
            amplitude_sigma=(float(magnitude_sigma) if (magnitude_sigma is not None
                                                       and family in ("spike", "noise_shaped"))
                             else float(fam_cfg.get("amp_sigma", 1.0))),
            label=family,
        )

    def inject(self, values: np.ndarray, spec: AttackSpec) -> InjectionResult:
        fn = FAMILIES[spec.family]
        return fn(np.asarray(values, dtype=float), spec, self.kind, self.seasonal_range,
                  self.sigma, np.random.default_rng(self.seed + spec.onset))


# --------------------------------------------------------------------------- #
# natural events (labelled from the reference, never injected)
# --------------------------------------------------------------------------- #


def natural_event_mask(reference_values: np.ndarray, kind: str, percentile: float,
                       min_run: int) -> np.ndarray:
    """Boolean mask of days inside a natural extreme of the reference series.

    Used twice: to evaluate false alarms *inside* natural extremes (the primary
    false-alarm control of the design -- a heat wave appears in the sensor and in
    the reference, so z stays small) and as an evidence feature in L4.
    """
    v = np.asarray(reference_values, dtype=float)
    ok = np.isfinite(v)
    if ok.sum() < 50:
        return np.zeros(len(v), bool)
    thr = np.nanpercentile(v[ok], percentile)
    above = ok & (v >= thr) if percentile > 50 else ok & (v <= thr)
    out = np.zeros(len(v), bool)
    run = 0
    for i, flag in enumerate(above):
        run = run + 1 if flag else 0
        if run >= min_run:
            out[i - run + 1: i + 1] = True
    return out
