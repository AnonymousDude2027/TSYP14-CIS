# State of the art and the gap (S1)

The challenge sits at the intersection of three literatures. This note records
what each provides, what it assumes, and which assumption breaks for in-situ
climate sensors — the last column is the reason for each design choice in the
system.

| Approach | Representative work (line of research) | Contribution | Assumption that fails here | What we take / change |
| --- | --- | --- | --- | --- |
| **False-data injection in control systems** | Secure state estimation and stealthy-attack analysis for power and water networks (Liu, Ning, Mo, Sinopoli and the follow-on literature) | Formal adversary model; the notion of a *stealthy* attack that keeps every residual below its threshold; observability conditions for detection | The plant is a linear model with known inputs and full measurement redundancy; observations are Gaussian and complete | We adopt the adversary hierarchy (A0–A3) and the *damage-under-a-budget* framing; we replace the linear state estimator with a robust regression on an independent reference, and replace "detectable/unobservable" with a calibrated false-alarm budget |
| **Climate data quality control** | GHCN-Daily QC flags, BSRN quality tests, neighbour-comparison and spatial-consistency checks | Operational, decades-proven fault detection (range, internal consistency, persistence, spatial outliers) | The checks detect *implausible* values; a falsified value that mimics weather passes all of them | We keep a subset of them as D4 (single-sample excursions) and D7 (reporting-gap availability) but add the reference-anchored residual tests that a plausible falsification must violate |
| **Change-point and sequential detection** | Page's CUSUM, Roberts' EWMA, PELT/binary segmentation, matched filtering | Optimal (in the Lorden sense) detection of a shift for a given false-alarm rate; exact inference for change points | Independent, stationary, complete, Gaussian-ish residuals | We use CUSUM/EWMA/PELT as D1/D2/B2 after *whitening, degapping and bias/scale stabilisation*; every threshold comes from a block bootstrap (or a phase-randomised surrogate where the bootstrap fabricates the signal) |
| **Residual-based sensor validation with redundancy** | Analytical-redundancy literature, cross-sensor validation | Cheap and powerful when several sensors measure the same quantity | A single station measures the point; no second sensor exists at the same location | We use a *different kind* of redundancy: an independent gridded product (reanalysis) that is cheap, complete and public, at the cost of site-representativeness error |
| **Replay / adversarial-sequence detection** | Anti-spoofing for GPS and sensor networks; subsequence matching | Detection of copied sequences via cross-correlation and exact matching | The signal is high-entropy and not strongly autocorrelated | We match on the *anomaly* road (a level series repeats its own annual cycle), exclude degenerate windows (dry spells are self-similar), and pair the match with a reference-contrast test |
| **Provenance and integrity** | Merkle hash chains, digital signatures, trusted execution | Tamper evidence for data in transit and at rest | A signature says nothing about truthfulness | L0 implements the chain and *explicitly* limits its claim; A3 is the case it cannot cover |
| **Adaptive/evasive adversaries** | Urbina et al. on stealthy sensor attacks; game-theoretic detection | Quantifies what an adversary achieves under a detection constraint | Requires knowing the detector's objective and its budget | We implement A2 white-box and black-box, and report damage at a fixed budget rather than a single detection rate |

## The gap, stated precisely

No published pipeline, to our knowledge, (a) anchors a point sensor to a
**hash-pinned independent reanalysis reference**, (b) calibrates **every**
threshold to a false-alarm budget in the presence of serial correlation,
35–45 % missing days and a drifting station/reference relationship, and (c)
reports detectability, false alarms and **attacker damage** on the same footing,
with the failure modes of the statistics — not only the success modes — measured
on real archives. This report does all three, and pays for it by documenting the
places where the result is a limitation rather than a win.

## References (line of research, not a bibliography of one paper)

* Secure state estimation / stealthy attacks: Liu, Mo & Johansson; Pasqualetti,
  Dörfler & Bullo; Teixeira et al.; Urbina et al.
* Sequential detection: Page (1954); Roberts (1959); Lorden (1971); Basseville &
  Nikiforov; Killick, Fearnhead & Eckley (PELT); Truong, Oudre & Vayatis
  (`ruptures`).
* Robust regression and scale: Huber; Rousseeuw & Croux (MAD/SN).
* Spectral and surrogate methods: Welch; Theiler et al. (phase randomisation);
  Ledoit & Wolf (shrinkage covariance).
* Climate data records and QC: Menne et al. (GHCN-Daily); Dürr et al. / Ohmura
  et al. (BSRN); reanalysis as a reference product: Hersbach et al. (ERA5);
  POWER project documentation (NASA LaRC).
* Reproducibility: Sandve et al.; Piccolo & Frampton (offline, container-based
  reproduction); `pytest` and `make`-based pipelines.
