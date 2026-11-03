# Detecting falsified in-situ climate observations with a reference-anchored, false-alarm-calibrated pipeline

*Technical report for the Track 2 challenge (false-data injection on in-situ
sensors). Anonymised; the code, the data bundle and every number below are
reproducible offline with `make all`.*

**Abstract.** A falsified sensor reading is dangerous precisely when it looks
ordinary. We present a six-layer detection pipeline that treats an in-situ
station as untrusted and an independent reanalysis product as a hash-pinned
trusted reference, and that converts "detect falsification" into a sequence of
statistical tests whose thresholds are *calibrated to a false-alarm budget*
rather than tuned to a detection target. Layers L0 (Ed25519-signature and
SHA-256 hash-chain provenance), L1 (conditioning, robust expectation, whitening),
L2 (six detector families), L3 (spectral-fingerprint and self-similarity
consistency), L4 (rule-based diagnosis with a first-class "undetermined" class)
and L5 (an action ladder with a decision-impact estimate) are implemented, unit
tested against known ground truth, and evaluated on 12 GHCN-Daily stations in
nine countries over 2015–2024 with NASA POWER as the reference. The evaluation
uses an injection grid over three variables and twelve stations, names four baselines, runs
five ablations, and reports detection delay, false alarms per sensor-year
(including inside natural extremes), AUPRC without point adjustment, and the
damage an adaptive attacker achieves at a fixed budget. Along the way we
document the failure modes that only appear on real, gappy, non-stationary
station data — a threshold-calibration non-monotonicity, a non-stationary
sensor/reference bias, a degenerate matched filter on zero-inflated variables, a
replay detector that matches dry spells, an injector that quietly produced
identical magnitudes for opposite-signed attacks, an attribution rule that
diagnosed the wrong alarm, and a whole failure class (a silenced sensor) that no
statistic can see and that needed its own channel — each with the measurement that
exposed it and the fix that removed it. {{ABSTRACT_NUMBERS}}

## 1. Problem and threat model

{{THREAT_SUMMARY}} The design consequence is a separation of concerns that we
enforce in the code: **L0 answers "who sent this and has it been altered"; L1–L5
answer "does it agree with an independent description of the same day".** No
detector depends on L0, so a compromised signer (adversary A3) degrades the
system to a purely statistical one rather than to nothing.

The system must also be judged on a quantity the literature usually omits: the
**false-alarm rate inside natural extreme events**, because an honest heat wave
is statistically indistinguishable from an injected step unless the detector uses
the reference. We report it explicitly (Table 3).

## 2. State of the art and the gap

Three literatures meet here. (i) *Sensor-network false-data detection* provides
the adversary hierarchy and the "stealthy attack" formulation (bounded bias that
keeps every residual test below its threshold), usually on linear
state-estimation models with no notion of false-alarm calibration.
(ii) *Climate data quality control* provides physical-consistency and
neighbour-comparison checks (GHCN-Daily, BSRN), which are designed for *faults*
and cannot see a plausible, physically consistent falsification.
(iii) *Change-point and sequential-analysis theory* provides CUSUM/EWMA, matched
filtering and block bootstrap, but assumes (near-)stationary, complete,
Gaussian-ish residuals.

The gap is the composition: no published pipeline (a) treats a reanalysis product
as a *pinned reference* for a point sensor, (b) calibrates *every* threshold to a
false-alarm budget under serial correlation and 35–45 % missing data, and (c)
reports detectability and attacker damage on the same footing. Table
{{TAB_REFERENCE}} in the repository documentation compares the closest published
approaches along those axes.

## 3. Architecture and trust mechanism

{{ARCHITECTURE}} The trust chain is: a 32-byte seed → an Ed25519 key pair
(fingerprint in every result file) → per-record signatures over a canonical
fixed-point encoding → a SHA-256 chain → a signed manifest over the 24 CSV files
of the bundle. Two design choices are worth stating because they are where naive
implementations fail:

* **The canonical form is not JSON.** Key order, float formatting and Unicode
  escaping are implementation-defined; a signature is only meaningful if every
  implementation agrees on the bytes. We sign `v1|sensor|seq|ts|variable|value|prev_hash`
  with the value quantised to 10⁻³ and the timestamp in UTC.
* **A missing day is a cadence gap, not a chain break.** The sequence number
  counts records *actually sent*; a dropout therefore shows up as a skipped
  afternoon report, and the verifier distinguishes "the record is absent"
  (an L3 fault) from "the record is a forgery" (an L0 incident). Conflating the
  two turns every station outage into a security alarm.

L0 rejects a tampered value, an outsider-signed record, a replayed record, a
reordered record and a deleted record; all five are unit tested
(`tests/test_provenance.py`). Its stated limit — it cannot detect a lying sensor
whose key is intact — is what L1–L5 exist for, and the report measures that case
separately (A3).

## 4. Signal processing

{{SP}} Four additions to the textbook chain were forced by the data, and each is
ablated:

1. **Rank-Gauss transform for precipitation.** A daily-rain residual has a point
   mass at zero and a heavy right tail; a MAD-standardised residual gives a `z`
   series of empirical standard deviation 5–7 and 13–15 false alarms per
   sensor-year. Mapping the calibration residual through its own empirical CDF
   and then the normal quantile function fixes the marginal shape, is monotone,
   and is fitted on calibration data only.
2. **Causal trailing bias correction.** The station-minus-reanalysis bias is not
   stationary (measured: annual mean residual moves by ≈1σ over the record at
   several stations). Removing a trailing 365-day median (30-day margin) is legal
   in a stream and turns a two-year false-alarm excursion into a level series.
3. **Bounded causal scale guard.** A 4–16 % inflation of the robust scale of `z`
   between the calibration and test epochs multiplies a 99.9 %-quantile
   exceedance rate by roughly 3. Dividing by the trailing MAD (clipped to
   0.75–1.5) removes it without touching a single test-period day.
4. **Gap-aware AR order selection.** With 35 % missing days, the intersection of
   lag-validity masks collapses; the order is chosen on the longest order whose
   common sample still meets a floor, and every order is scored with the
   maximum-likelihood innovation scale (mixing a MAD for p = 0 with a standard
   deviation for p ≥ 1 made AIC systematically choose p = 0, leaving a lag-1
   autocorrelation of 0.29 in `z` and a permanently mis-calibrated CUSUM).

## 5. Detection and diagnosis

{{DETECTORS}} The suite is calibrated by block bootstrap (block 30 days) at
budgets {0.1, 0.5, 1, 2} alarms per sensor-year, with the budget split across
detectors (Bonferroni) and reported with its Monte-Carlo standard error. Two
detectors are calibrated on phase-randomised surrogates instead, because a block
bootstrap duplicates blocks and therefore fabricates exactly the exact-repetition
pattern the replay detector is meant to find — a subtle way to make a detector
look specific while it is not.

Alarm *events*, not alarm samples, are counted, using one shared Schmitt-trigger
definition (`SchmittAlarm`) instantiated by both the vectorised evaluation and the
streaming pipeline; the two paths are asserted to produce the same alarm days, so
the reported numbers describe the system that actually runs.

Diagnosis is a transparent rule table (Tier 1) with a multinomial logistic
regression trained on training sites and attack families (Tier 2). "Undetermined"
is a first-class output: a stuck-at fault and a constant replay are identical in
the data. A persistent level shift (≥180 days) is reported as a *fault* with the
recommendation to re-baseline, not as a 600-day attack.

## 6. Evaluation and evasion

{{EVAL}}

{{TABLE1}}

{{TABLE2}}

{{TABLE3}}

{{TABLE4}}

{{TABLE7}}

{{FIGURES}}

{{EVASION}}

{{BASELINES}}

{{ABLATION}}

{{CROSSSITE}}

## 7. Efficiency and reproducibility

{{EFFICIENCY}} Reproducibility is enforced rather than asserted: one seed, a
pinned data manifest, a model cache that is byte-stable for a given seed, and
`python -m tsfdi verify`, which compares the *regenerated* result files with the
committed `expected_results.json` and fails on any metric outside a per-metric
tolerance (absolute for rates, relative for continuous quantities); it reports
internal consistency on its own and reproducibility only as the last step of the
chain that regenerates the tables first. `make all` runs data-verify → gonogo →
calibrate → experiments → report-figures → report-text → verify → static, needs
no network access after the pinned bundle, and ends with `make report-pdf` for
the PDF artefact.

## 8. Limitations

{{LIMITATIONS}}

## 9. Run guide pointer

`docs/reproduce.md` gives the environment, the seven `make` targets, the expected
runtimes on a 2-vCPU/4 GB machine and the artefact of record for every number in
this report. `docs/threat_model.md` states the adversary capabilities and,
with equal care, what the system does not claim. `docs/design.md` records every
deviation from a **[Default]** in the specification together with the measurement
that motivated it.
