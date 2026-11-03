# Design decisions and deviations from the specification's defaults

The specification marks a number of choices **[Default]**. This document records
every place where the implementation differs from a default, or makes a choice
the specification left open, together with the measurement or argument that
forced it. Nothing here is a matter of taste: each entry was a bug, a measured
failure, or a cost that had to be paid somewhere.

## 1. Data: an untrusted station and an independent reanalysis reference

The task defines false-data injection on in-situ sensors. The instantiation used
here is:

* **Untrusted ("sensor") road** — GHCN-Daily station records (`TMAX`, `TMIN`,
  `PRCP`), quality-flag clean, 2015-01-01 to 2024-12-31. These are the values an
  attacker can falsify.
* **Trusted (reference) road** — NASA POWER daily reanalysis at the station
  coordinates (`T2M_MAX`, `T2M_MIN`, `PRECTOTCORR`). A different instrument,
  a different processing chain, and an independent provider, which is what makes
  it a reference rather than a copy.

Both roads are pinned in a signed manifest (`data/bundle/MANIFEST.sha256` signed
with Ed25519); the pipeline refuses to run if any hash differs. Cross-site
generalisation is possible because 12 stations in 6 countries and 4 climate
zones are bundled (Tunisia, Egypt, Spain, France, Germany, Netherlands, Finland,
Norway, United States), with two sites held out from all tuning.

**Measured justification** (`results/gonogo.md`): the sensor-reference
correlation is 0.93-0.99 for temperature, so the reference carries real
information about the day; for precipitation it is much weaker, which is the
reason for the restricted claims on PRCP (§6 below).

## 2. Completeness: design for gaps instead of hiding them

The specification's default asks for ≥ 95 % complete records. On GHCN-Daily this
holds for the European and North-American stations in the bundle but **not** for
the Tunisian, Egyptian and Saharan ones, where `TMAX` is reported on a 60-75 %
duty cycle (`TMIN` is denser). Dropping the North-African sites would have
removed exactly the region the challenge is motivated by, so the pipeline was
designed for gaps. Five consequences, all implemented and tested:

1. **Nothing is interpolated.** A missing day is a hole with a flag. Detector
   windows use the samples that exist (`n_eff`), and a window with fewer than two
   samples produces `NaN`, not a guess.
2. **The AR filter uses the state-space convention.** A missing residual enters
   the recursion at its conditional mean; the filter state continues. Skipping
   the recursion whenever a lag is missing would throw away 0.65^6 ≈ 7 % of the
   samples for an AR(5) on a 65 %-complete station.
3. **Prefix-sum statistics are gap-aware.** Every windowed statistic (mean
   shift, slope) is computed from sums over *available* pairs, verified against a
   brute-force implementation on series with 30-40 % missing days
   (`tests/test_detectors.py`).
4. **The AR order is chosen on a common sample.** With 35 % missing days the
   intersection of the lag-validity masks collapses (CARTHAGE `TMIN`: 1015
   usable calibration residuals, 554 rows with two consecutive valid days, 152
   with four). AIC is therefore computed on the longest order whose common
   sample still meets a floor (300 rows or 15 % of the pairs); otherwise the
   comparison is between likelihoods computed on different samples and is not a
   comparison at all. This is stated in the code and in `results/gonogo.md`.
5. **Alarms are gated on validity.** A missing day cannot alarm, but the
   statistic series keeps its calendar meaning (a 7-day window is still seven
   calendar days, not seven observations).

## 3. SP3 whitening: three stages, one of them not in the specification

The specification's SP3 is "AR(p) whitening plus a seasonally varying robust
sigma". Two problems appeared when it was applied to real data, and both were
fixed at the modelling level rather than by tuning the threshold:

* **AIC was not comparing like with like.** The order-0 model was scored with a
  MAD-based robust scale and the order-p models with a standard deviation. For a
  heavy-tailed residual, MAD is systematically smaller, so AIC always picked
  p = 0 and the autocorrelation went straight into CUSUM (measured: lag-1 ACF of
  z = 0.29 on CABAUW `TMAX`). Fixed by scoring every order on the same rows with
  the maximum-likelihood innovation scale.
* **Precipitation residuals are not Gaussian in any linear coordinate.** 70 % of
  days are dry, so the residual has a point mass at zero and a long right tail;
  a MAD-standardised residual gives a z series with empirical standard deviation
  5-7 and a false-alarm rate of 13-15 per sensor-year against a budget of 1. The
  fix is a **rank-Gauss (empirical probability-integral) transform**, fitted on
  the calibration period only, applied before the AR filter:
  `w = Phi^-1(F_cal(r))`, extrapolated linearly beyond the calibration range and
  capped at |w| = 8. It is monotone, so a positive bias remains a positive shift,
  and it is enabled per variable kind (`sp.rank_gauss_kinds: [multiplicative]`).
  After the transform the precipitation z series has unit scale and a false-alarm
  rate in line with temperature (see `docs/report.md`, Table 3).

The rank-Gauss transform is therefore **an addition to the specification's SP3**,
not a deviation from a default, and it is included in the ablation so that its
contribution is measured rather than asserted.

## 3b. Two things the specification's SP3 could not do, and what replaced them

* **The sensor/reference bias is not stationary.** Measured annual mean residuals
  on JENDOUBA `TMAX` move from +0.24 °C (2015) to −0.90 °C (2024) — about one σ
  over the record — because the station (instrument, exposure, local site) and the
  reanalysis both drift. A drift detector whose threshold was calibrated on
  2015-2019 sees that as a permanent bias and alarms for years: the measured
  false-alarm rate was 6 per sensor-year against a budget of 1, and the drift
  CUSUM alone contributed 4 per year, on several sites. No stationary threshold
  can fix a non-stationary statistic, so SP3 now removes a **causal trailing
  robust bias** before whitening: the median residual of the previous 365 days,
  ending 30 days before the current day. It is causal (legal in a stream), it
  cannot be dragged by an ongoing attack (median, and a margin), and its cost is
  explicit — a bias that develops slower than the window chases its own baseline.
  That is exactly the A2 black-box strategy, which is why the ramp family is
  evaluated down to 1 σ and Figure 2 reports damage against the false-alarm budget.
* **The fingerprint distance is not stationary either.** The Mahalanobis distance
  of the spectral fingerprint has a median of ~3.5 in 2015-2019 and ~5.5 in
  2020-2024 for the *same, honest* station: the relationship between a point
  station and a reanalysis grid cell changes over a decade. A threshold calibrated
  on the first period therefore fired 2.4 times per sensor-year on every site, for
  both variables, with nothing wrong. D6's statistic is now **self-normalised**,
  `d_t / median(d over the previous two years, excluding the last 30 days)`, which
  keeps the detector's purpose (a *change* in the fingerprint) and removes the
  stationarity assumption. This is a deviation from "robust Mahalanobis distance
  to the calibration distribution" as literally specified; the reason is measured,
  not aesthetic, and the limitation it introduces (a change slower than the
  two-year baseline is invisible to this channel) is stated in the code.

## 3c. Detectors that only work on the right kind of variable

The specification defines detectors per *target* (step, drift, replay), not per
variable, and two of them are simply degenerate on zero-inflated data. Both
degeneracies were found by measuring false alarms on clean data, and both are
fixed by restricting the channel to the variable kinds where it is identifiable —
not by raising a threshold until the number looks acceptable:

* **D3 matched filter (NCC).** A precipitating day is a spike; a window of seven
  days is mostly a *sparse spike pattern*. Two windows whose spikes fall on the
  same offsets correlate almost perfectly after centring, whatever the
  amplitudes, so the statistic saturates and fires 12-14 times per sensor-year on
  honest data. The NCC channel is therefore restricted to `temperature`
  (`detectors.D3_replay.ncc_kinds`), and precipitation keeps the two channels that
  remain identifiable: the exact-repeat test (a copied wet sequence, at a 14-day
  window instead of 7, because a sparse "which days were wet" pattern recurs by
  chance in five years) and the reference-contrast test `D3_block_contrast`.
* **Degenerate windows in the other channels.** A window with fewer than three
  distinct quantised values carries no shape, and it repeats itself constantly
  (seven dry days is an ordinary event). `replay_exact`, `replay_max_ncc` and the
  runtime `ReplayDetector` all skip such windows. A constant run is the
  *stuck-at* fault's signature and is D4's job; counting it as a replay
  double-counted it and inflated the precipitation false-alarm rate.

With these three changes (trailing bias, self-normalised fingerprint, degenerate-
window exclusion) the measured false-alarm rates moved from 13-15/yr (PRCP) and
2.2-6.4/yr (TMAX) to the values in `docs/report.md`, Table 3.

## 3d. One definition of an alarm event

The vectorised evaluation and the streaming pipeline originally counted events
with two slightly different rules: the evaluation used a Schmitt trigger (an event
requires the alarm to have *cleared*, so a persistent exceedance is one event),
the pipeline only enforced the cooldown (so a persistent exceedance produced one
event every 15 days). Both are defensible; having both is not. The rule now lives
in a single class, `detect.statistics.SchmittAlarm`, which both paths instantiate,
and `tests/test_pipeline.py` asserts that the two paths produce the *same set* of
alarm days on the same series. The specification's "an alarm latches until a
documented reset rule clears it" is exactly this rule.

## 3e. Three more guards that the data demanded

All three are causal, all three are ablated or measured, and all three were added
only after a measurement showed the unguarded version failing.

* **Gap reset after a long outage.** A 300-day outage followed by a report is not
  "yesterday was missing": the AR recursion would predict today's value from a
  state that is a year old. The next p+1 observations after a run of more than 7
  missing days therefore carry no `z`. Short gaps keep the state-space convention,
  which is what makes a 65 %-complete station usable at all. (The bundle contains a
  station — KHARGA — that reports 12 days in all of 2021, so this is not
  hypothetical.)
* **Bounded causal scale guard.** The robust scale of `z` is 4 % larger in the test
  epoch on the median pair and up to 16 % on some, and a 99.9 %-quantile threshold
  amplifies a 10 % scale inflation by roughly a factor of three. Dividing `z` by
  its trailing 365-day MAD (30-day margin, clipped to 0.75–1.5) removes the drift
  without using a single test-period day for fitting; the clip keeps a pathological
  window from collapsing the statistic.
* **Persistent shift → re-baseline, not re-alarm.** When the residual stays elevated
  for ≥ 180 days with no replay, fingerprint or provenance evidence, the rule table
  reports a *fault* (`code="persistent_shift"`) and the action ladder recommends a
  re-baseline (SUSPEND). Calling a two-year level shift a 600-day attack is
  technically possible and operationally useless.

## 4. SP5 matched filter: match the anomaly, not the level

The specification says "z-normalised cross-correlation of the last L samples
against every earlier window of the sensor's own history". Applied to the raw
level, this detector is **dead**: a strongly seasonal series repeats its own
annual cycle, so a 7-day window matches the same calendar window a year earlier
with NCC ≈ 1 and the detector alarms every day (measured on synthetic and real
data alike; the calibrated threshold saturates at 1.0, which is the signature of
a detector that cannot discriminate). The matched filter therefore runs on the
**anomaly road** `a_t = y_t - seasonal_baseline_t` (SP2 harmonics only, no
reference, no look-ahead), which removes the deterministic annual cycle and keeps
the day-to-day shape that a replay actually copies. The exact-repetition channel
still tests the *raw* values at the data quantisation (0.1 °C / 0.1 mm), because
that is what a replay attacker copies.

## 5. False-alarm calibration: what changed and why the numbers are what they are

* **Block bootstrap, as specified (block = 30 days), but 300 replicates instead
  of 2000**, because the calibration is run for 14 detectors × 34 (site,
  variable) pairs × 4 budgets on a 2-vCPU machine and 2000 replicates cost a
  factor 6.7 more for a reduction in Monte-Carlo error from 5.8 % to 2.2 % on
  the *rate* (1/√N). The Monte-Carlo error is reported with every budget rather
  than hidden.
* **The two replay detectors are calibrated on phase-randomised surrogates, not
  on the block bootstrap.** A circular block bootstrap duplicates blocks, so the
  surrogate series contains exact repeats that the real series does not: it
  fabricates precisely the pattern D3 exists to detect, and the calibrated
  threshold would be wrong in the *dangerous* direction. Phase randomisation
  preserves the power spectrum (hence the autocorrelation) and destroys
  subsequence repetition, which is the correct null. This is a documented
  deviation from the default and it is the reason D3 has a usable threshold.
* **The suite-level budget is split across detectors (Bonferroni)** instead of
  being granted to each detector. With 14 detectors and a 1-per-year budget,
  granting 1/year to each would give ≈14/year at the suite level. Splitting
  gives a suite-level guarantee that is slightly conservative because the
  detectors are correlated (they share z), which is the right direction to err.
* **The threshold search is restricted to the region above the null median.**
  Below it, the statistic is in alarm on most days, and with the latching event
  rule a permanently-alarmed detector counts as *one* event, so the "rate" loses
  monotonicity and bisection converges to ≈0 (a threshold of zero that alarms
  once and then stays latched). This was a real bug, found by a test asserting
  that the achieved rate is close to the budget.
* **Alarms are events, not samples.** An alarm latches and is reset by a
  documented rule (`calibration.cooldown_days`); the false-alarm rate is events
  per sensor-year. Counting alarmed *samples* inflated the measured false-alarm
  rate by up to 7× during development, and the latching rule is what makes the
  budget comparable across detectors with different memory.
* **The achieved rate is reported next to the requested one.** Where a budget is
  unreachable (the smallest attainable rate is still above it), the calibration
  returns the achieved rate instead of silently returning a threshold that
  satisfies nothing.

## 6. Per-variable budgets and where the claims stop

A single 1-per-year budget across all variables is not attainable for
precipitation with a reanalysis reference. Rather than tune until the number
looks good, the budget is per variable and the achieved rates are reported:

| Variable | Operating budget (alarms / sensor-year) | Rationale |
| --- | --- | --- |
| `TMAX`, `TMIN` | 1.0 | Residual is close to Gaussian after SP3; the reference explains most of the variance. |
| `PRCP` | see `docs/report.md`, Table 3 | Zero-inflated, weak point-gauge/reanalysis correlation, and a distribution shift between the calibration and test periods that the fixed calibration cannot absorb. Reported as a limitation with its own budget, and PRCP results are labelled as such wherever they appear. |

The go/no-go experiment (section 6 of the specification) is the gate: for each
pair it reports the minimum detectable bias at the operating budget, and a pair
whose minimum detectable bias exceeds 3 σ is reported as a documented limitation
rather than quietly dropped.

## 7. Detection granularity and the metrics

* **No point adjustment, anywhere.** AUPRC is computed over sample days with the
  score `max_d S_d(t)/h_d`, i.e. the actual statistic values divided by their
  thresholds, and detection is an *event* whose interval is
  `[onset, onset + duration + 30]`. Point-adjusted F1 rewards a detector for
  catching one point of a 30-day attack; the specification forbids it and the
  code has no code path that computes it.
* **Delay** is `t_alarm - onset` in days, reported as median and IQR, with the
  never-detected fraction as a separate column.
* **Damage** is counted in σ·sensor-days of undetected bias, and separately as
  the number of *decision flips* (frost, heat stress, drought) in the alarm
  window, which is the "for what purpose was it modified" answer.

## 8. Provenance: the honest scope

* Ed25519 over a canonical fixed-point encoding (`sp/../provenance/record.py`),
  SHA-256 chaining, signed manifest. The canonical form is *not* JSON: key order,
  float formatting and Unicode escaping are implementation-defined, and a
  signature is only meaningful if every implementation agrees on the bytes.
* Values are quantised to 1e-3 before signing, so two representations of the same
  measurement cannot produce two different signatures; the quantisation is
  documented and tested.
* The sequence number counts records *actually sent*, so a sensor outage is a
  cadence gap, not a chain break. A backwards timestamp is always rejected; a
  forward jump is rejected only beyond `max_gap_seconds` (40 days), because a
  daily series legitimately skips days.
* **Stated limit, repeated in the code and in the report:** provenance proves who
  sent what, in which order, unchanged. It proves nothing about whether the
  sensor measured truthfully. An A3 attacker who holds the key passes L0
  completely, which is why L1-L5 exist and why the evaluation reports A3
  separately.

## 9. Diagnosis: two confidence numbers, and a class for "I do not know"

* `confidence` is a float (the posterior of the chosen class: 0.6-0.97 from the
  rule table's documented priors, or the logistic-regression probability in
  Tier 2) and is what the alert JSON carries. `confidence_band` is the ordinal
  (low/medium/high) the rule table states from the number of agreeing evidence
  lines. Reporting a string where the specification asks for a confidence was a
  real defect caught by a test.
* **"Undetermined" is a first-class class.** A stuck-at fault and a constant
  replay are identical in the data; the system says so. Excluding it from the
  confusion matrix would inflate accuracy by removing the hard cases.

## 10. Reproducibility and cost

* Deterministic: one seed (`config.SEED`) drives every bootstrap, surrogate and
  injection; models are cached under `results/cache/models` and are byte-stable
  for a given seed and data manifest.
* Offline: the bundle (24 CSVs + signed manifest) is committed; no command in
  `make all` touches the network. `python -m tsfdi fetch` is the only networked
  command and is not part of the pipeline.
* Feasibility target: fresh Ubuntu VM, 4 GB RAM, no GPU. Measured runtime and
  peak memory are in `docs/report.md`, Table 5; the pipeline is pure NumPy /
  SciPy / pandas, so there is no compilation step and no GPU dependency.
* `make verify` re-runs the pipeline and compares every published number against
  `expected_results.json` (rate metrics with an absolute tolerance, because a
  rate of 0.02/yr cannot be compared with a 2 % relative tolerance).

## 3f. The magnitude axis, and why it had to be redefined

Every generalisation result is quoted in units of the residual scale σ that the
detector actually sees, because that is the only magnitude that is comparable
across sites, seasons and variables. Making that axis *exact* took three fixes,
all of which were found by looking at a raw grid file rather than at a summary.

**The nominal value is the axis.** The injector reports two numbers: the *nominal*
magnitude (the grid point the injection was drawn from) and the *realised* one
(largest absolute change actually applied). An early version reported only the
realised value, and that silently split cells: a 90-day ramp reaches
`1 − 1/rise` of its nominal peak, so "1.5 σ" arrived as 1.467 σ and formed a
second group in every table. `magnitude_sigma` is now the nominal grid point,
`magnitude_sigma_realised` the effect actually achieved on the same axis, and
`magnitude_units` the same thing in data units; the family tests check that the
nominal and realised values agree to rounding.

**A sign bug, not a data bug.** `step` recorded `abs(delta.max())`. For a
*downward* step every untouched day contributes a delta of exactly 0, so the
maximum was 0 and every negative-signed step was logged as a magnitude of zero —
150 phantom `0.0 σ` rows in one grid file. It now takes the nan-safe maximum of
`|Δ|` inside the window, which also fixes the all-NaN magnitudes that a single
missing day inside the window produced for precipitation.

**Multiplicative variables need a different axis.** For precipitation the
residual lives in square-root coordinates and is zero-inflated, so a bias in
millimetres is not a bias of a fixed number of sigmas: one sigma of the
precipitation residual is 0.19 mm^0.5, not 0.19 mm. Two consequences, both of
which were measured before they were fixed:

* an additive offset of `m sigma` is applied in the **SP1 coordinate**
  (`sqrt(y)`), so a "2 sigma step" is a wetness-dependent change in millimetres
  whose realised effect really is 2 sigma. Before this, a 3 sigma rain step
  realised only ~1 sigma and the precipitation frontier was shifted sideways by a
  factor of three;
* a *multiplicative* attack (scaling) has its gain computed from the data so that
  the realised change equals the nominal magnitude:
  `g = (1 + m sigma / mean sqrt(y))^2 - 1` in the square-root coordinate and
  `g = m sigma / mean |y|` for temperature. A fixed 10 % gain is a 3 °C bias at a
  warm station and a 0.6 °C bias at a cold one, which is why the first version of
  the scaling row showed 100 % detection at every magnitude: it was a 1.8-3 sigma
  attack mislabelled as 0.5 sigma.

For the organizers' *percentage* profile the reading is unchanged and is stated
wherever it is used: a percentage of the seasonal range for temperature, a
multiplicative factor for rain (so 12 % = 0.6 sigma there).

Noise-shaped attacks use the same nominal axis (they were previously drawn from
`amplitude_sigma`, a fixed 1.0, which made their detection rate flat in the
magnitude grid — a family that cannot vary cannot be measured).

## 3g. Two adaptive-attacker artefacts, and why the black-box row is empty no longer

The A2 experiments initially reported that a black-box adaptive attacker achieved
*zero* undetected bias on every site, i.e. that the detector suite had no
exploitable margin at all. That is not a result, it is a bug, and it took three
fixes to see it:

1. **Attribution.** The bisection counted *any* alarm event after the injection
   window as a detection, so a natural event three hundred days later capped the
   search at zero. Alarms are now counted only inside `[onset, onset+duration+30]`,
   the same window the rest of the evaluation uses.
2. **Pre-existing alarms.** Even inside the window, an alarm that the unmodified
   series produces anyway (a heat wave, a persistent shift, a false alarm of the
   drift detector) cannot be attributed to the injection. The clean-series event
   mask is subtracted from the attacked-series mask.
3. **Resolution.** A 0.05 σ bisection tolerance cannot distinguish "no margin"
   from "0.05 σ of margin", and the returned lower bound was silently 0 whenever
   the last step alarmed. The tolerance is now 0.01 σ.

With those three fixed the black-box attacker reaches 0.05–0.9 σ of undetected
bias depending on site (Table 6.4), which is the interpretation the theory
predicts: below the detectors' effective resolution an adversary is free, above it
it is not, and the width of the free band is site-dependent because the residual
scale, the AR order and the false-alarm transfer differ by site.

The evasion *curve* had a fourth artefact: a single injection window per budget
makes the curve non-monotone, because the achievable undetected bias depends on
which days the window covers (a calm spell hides more than a stormy one). The
curve now averages four windows × both signs per (site, budget), and both
monotonicity and the L3 gap survive that averaging.

## 3h. Coverage: pairs that cannot be modelled are named, not hidden

KHARGA precipitation has 13 valid days in eleven years, and one earlier run wrote
an `ERROR` row into a results table for it. A results table is not the place for
an exception message, and quietly dropping the row is worse: the reader cannot
tell a missing pair from a pair that failed. `results/excluded_pairs.csv` now
lists every excluded (site, variable) pair with the reason and is regenerated on
every evaluation run; the tables cover the 34 design pairs, which is all of them
on the current bundle. The exclusion path is exercised by
`tests/test_reports.py`.

## 3i. The blind spot no statistic can cover: D7, the availability channel

Every detector in L2 answers "does this value agree with an independent description
of the same day".  All of them are blind to the failure in which the value stops
arriving: a dropout makes the innovation NaN, so nothing exceeds a threshold, and
a silenced station is indistinguishable from a quiet one.  In a false-data-
injection exercise that is the most important gap of all, because withholding the
record is a data-integrity attack that has no statistics behind it.

D7 therefore tests the reporting cadence, with a data-derived envelope: a run of
missing days longer than 1.5x the longest gap observed in the calibration period
is an outage fault.  It is a rule rather than a calibrated test (there is no
continuous statistic whose tail defines a budget), and it is deliberately kept out
of the statistical alarm stream, so that a station with a broken modem cannot
consume the detectors' false-alarm budget.  The measured consequence in Table 7 is
that 12 % of the bundled days are missing and a quarter of the pairs have at least
one outage event in the test period — i.e. the channel fires on real gaps, not on
noise.

Two attribution bugs were fixed while making the diagnosis numbers honest, and
both are the same mistake in different places: **the first alarm of the record is
not the alarm of the injection.**  The diagnosis experiment used to take
`events[0]` over the whole series, so every row of the confusion matrix was a
diagnosis of some unrelated false alarm six months earlier — which is why the
matrix looked identical for every injected family.  It now diagnoses the first
exceedance inside `[onset, onset+duration+30]`, preferring the exceedance day over
the cooldown-collapsed event onset so that an injection landing within 14 days of
an earlier alarm still gets its own evidence.  The generator had the mirror-image
bug: `make_spec` ignored the requested magnitude for the families parameterised by
an amplitude, so "inject a 12 sigma spike" injected the configured 1 sigma
default, and the spike row of every experiment was measuring noise.  Both are
covered by tests.
