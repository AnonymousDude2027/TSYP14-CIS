"""Evidence extraction for L4.

An alarm without evidence is not actionable.  At the moment an alarm fires, the
pipeline has already computed a great deal of information about the *shape* of
the anomaly; this module turns it into a fixed-length evidence record that

* the rule table (Tier 1) reads directly,
* the logistic regression (Tier 2) consumes as features,
* and the explanation text (L5) quotes verbatim.

The features are the ones the specification lists: which detectors fired, the
onset shape and scale from SP4, the duration, the number of variables affected,
agreement with the reference, the fingerprint distance, the provenance verdict,
the fault-envelope status, and whether the reference itself shows an extreme
event (the natural-event confounder).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..sp.wavelet import WaveletAnalyzer, onset_shape

CLASSES = ("natural", "fault", "attack", "undetermined")


@dataclass
class Evidence:
    site: str
    variable: str
    t_alarm: int
    detectors_fired: list[str] = field(default_factory=list)
    detector_families: list[str] = field(default_factory=list)
    onset_shape: str = "unclear"
    z_scale_at_alarm: float = float("nan")
    run_length_days: int = 0
    n_variables_affected: int = 1
    mean_z_in_window: float = float("nan")
    reference_extreme: bool = False
    fingerprint_distance: float = float("nan")
    fingerprint_too_clean: bool = False
    provenance_violation: bool = False
    fault_envelope_exceeded: bool = False
    stuck_run_days: int = 0
    replay_exact_hit: bool = False
    block_contrast: float = float("nan")
    # a single sample far outside the envelope with quiet neighbours: a spike/glitch
    single_sample_excursion: bool = False
    # availability channel (D7): a missing-data fault, not a statistical alarm
    availability_run_days: int = 0
    availability_envelope_days: int = 0
    natural_event_mask: np.ndarray | None = None

    def feature_vector(self) -> np.ndarray:
        fam = set(self.detector_families)
        return np.array([
            1.0 if self.provenance_violation else 0.0,
            float(len(self.detectors_fired)),
            1.0 if "D1" in fam else 0.0,
            1.0 if "D2" in fam else 0.0,
            1.0 if "D3" in fam else 0.0,
            1.0 if "D4" in fam else 0.0,
            1.0 if "D6" in fam else 0.0,
            1.0 if self.onset_shape == "step" else 0.0,
            1.0 if self.onset_shape == "ramp" else 0.0,
            1.0 if self.onset_shape == "spike" else 0.0,
            np.log1p(max(self.run_length_days, 0)),
            float(self.n_variables_affected),
            np.nan_to_num(self.mean_z_in_window, nan=0.0),
            1.0 if self.reference_extreme else 0.0,
            np.nan_to_num(self.fingerprint_distance, nan=0.0),
            1.0 if self.fingerprint_too_clean else 0.0,
            1.0 if self.fault_envelope_exceeded else 0.0,
            1.0 if self.replay_exact_hit else 0.0,
            np.nan_to_num(self.block_contrast, nan=0.0),
            1.0 if self.single_sample_excursion else 0.0,
            1.0 if (self.availability_envelope_days
                    and self.availability_run_days > self.availability_envelope_days) else 0.0,
        ])

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "provenance_violation", "n_detectors", "fam_D1", "fam_D2", "fam_D3", "fam_D4",
            "fam_D6", "shape_step", "shape_ramp", "shape_spike", "log_run_length",
            "n_variables_affected", "mean_z", "reference_extreme", "fingerprint_distance",
            "fingerprint_too_clean", "fault_envelope_exceeded", "replay_exact_hit",
            "block_contrast", "single_sample_excursion", "availability_fault",
        ]

    def as_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "natural_event_mask"}
        d["feature_names"] = self.feature_names()
        return d


def extract(model, t_alarm: int, detectors_fired: list[str], *, provenance_violation: bool = False,
            n_variables_affected: int = 1, window: int = 14,
            reference_extreme: bool = False, natural_mask: np.ndarray | None = None) -> Evidence:
    """Build the evidence record at alarm time ``t_alarm``."""
    from ..detect.engine import DETECTORS

    fired = list(detectors_fired)
    fams = sorted({DETECTORS[f][0] for f in fired if f in DETECTORS})
    z = model.z
    lo = max(0, t_alarm - window + 1)
    seg = z[lo: t_alarm + 1]
    seg = seg[np.isfinite(seg)]

    # onset shape from SP4 on the alarm window plus the preceding month
    ctx = z[max(0, t_alarm - 60): t_alarm + 1]
    shape = onset_shape(np.nan_to_num(ctx, nan=0.0)) if len(ctx) > 30 else "unclear"

    run = 0
    for i in range(t_alarm, -1, -1):
        if np.isfinite(z[i]) and abs(z[i]) > 1.0:
            run += 1
        else:
            break

    fp = float("nan")
    too_clean = False
    if model.fingerprint is not None and model.fingerprint.inv_cov is not None:
        d = model.fingerprint.distance_series(z, model.conditioned.y, model.conditioned.x)
        fp = float(d[t_alarm]) if np.isfinite(d[t_alarm]) else float("nan")
        flags = model.fingerprint.too_clean_flags(z, model.conditioned.y, model.conditioned.x)
        too_clean = bool(flags[max(0, t_alarm - 7): t_alarm + 1].any())

    # is this a one-sample excursion? (count of large |z| days in a +-3 day
    # neighbourhood, so that a level shift -- many large days -- is excluded)
    nb = z[max(0, t_alarm - 3): t_alarm + 4]
    large = np.isfinite(nb) & (np.abs(nb) > 3.0)
    single = bool(large.sum() <= 1 and np.nanmax(np.abs(nb)) >= 6.0 if nb.size else False)

    stuck = 0
    q = np.round(model.conditioned.y / 0.1)
    for i in range(t_alarm, 0, -1):
        if np.isfinite(q[i]) and q[i] == q[i - 1]:
            stuck += 1
        else:
            break

    stats = model.statistics()
    # detector statistics are (T, 1) arrays -- squeeze before indexing so a single
    # day is a scalar (numpy >= 1.25 refuses to cast a size-1 array implicitly)
    bc = stats.get("D3_block_contrast")
    bc = None if bc is None else np.asarray(bc, dtype=float).reshape(-1)
    block_contrast = float(bc[t_alarm]) if bc is not None and np.isfinite(bc[t_alarm]) else float("nan")
    exact = stats.get("D3_replay_exact")
    exact = None if exact is None else np.asarray(exact, dtype=float).reshape(-1)
    exact_hit = bool(exact is not None and np.nansum(exact[max(0, t_alarm - 14): t_alarm + 1]) > 0)

    return Evidence(
        site=model.site, variable=model.variable, t_alarm=int(t_alarm),
        detectors_fired=fired, detector_families=fams, onset_shape=shape,
        z_scale_at_alarm=float(np.nanmax(np.abs(seg)) if seg.size else np.nan),
        run_length_days=int(run), n_variables_affected=int(n_variables_affected),
        mean_z_in_window=float(np.nanmean(seg) if seg.size else np.nan),
        reference_extreme=bool(reference_extreme),
        fingerprint_distance=fp, fingerprint_too_clean=too_clean,
        provenance_violation=bool(provenance_violation),
        fault_envelope_exceeded=bool(stuck >= model.stuck_threshold),
        stuck_run_days=int(stuck), replay_exact_hit=exact_hit, block_contrast=block_contrast,
        single_sample_excursion=single,
        natural_event_mask=natural_mask,
    )
