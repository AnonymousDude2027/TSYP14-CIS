# Compliance with the TSYP14 challenge brief (Track 2)

This page maps the brief's requirements to the artefacts in this repository, states
which public datasets are used and why they are treated as the reference, and
defines the "standard evaluation scenario" precisely enough that a reader can
reproduce it without reading the code.

Track: **Track 2 — False-Data Injection** (silent failure in in-situ sensing).
Segment: ground/in-situ sensing (surface stations) with a reanalysis reference.

## 1. The six official content deliverables

| # | Brief's deliverable | Where it is |
| --- | --- | --- |
| 1 | Threat model & segment analysis | `docs/threat_model.md` (assets, system boundary, A0–A3 adversary hierarchy, attacker knowledge, calibration/test split, and an explicit "what this system does not claim"); summarised in `docs/report.md` §1. |
| 2 | Platform / architecture / trust anchor | `docs/report.md` §3 and `docs/design.md`; the L0–L6 pipeline is `src/tsfdi/pipeline.py`; the trust anchor is `src/tsfdi/provenance/` (Ed25519 per-record signature + SHA-256 hash chain + signed component manifest, `data/bundle/MANIFEST.sig`). A software anchor is used, which the brief explicitly allows. |
| 3 | Signal / image processing | `src/tsfdi/sp/` — conditioning (SP1), robust seasonal expectation with an optional gradient-boosted correction (SP2), AR whitening (SP3), shape features (SP4), rank-Gauss transform for precipitation (SP5), spectral fingerprint (SP6); `docs/report.md` §4. |
| 4 | AI or statistical detection | `src/tsfdi/detect/` — D1 step/CUSUM, D2 drift/CUSUM/EWMA/slope, D3 replay (matched filter, exact repeat, block contrast), D4 single-sample excursion, D6 spectral fingerprint, D7 availability envelope; block-bootstrap threshold calibration at a chosen false-alarm budget; baselines B0–B3 in `src/tsfdi/eval/baselines.py`. |
| 5 | Adversarial validation & response | `src/tsfdi/attacks/` (A0–A3 lab, five attack families and four fault families) and `src/tsfdi/diagnose/` (evidence extraction, rule table, confidence band, action ladder, decision impact); `docs/report.md` §5–§6. |
| 6 | Documentation | `docs/report.md` + `docs/report.pdf` (report), `docs/report_appendix.md` + `docs/appendix.pdf` (full tables), `docs/reproduce.md` (every number, with the command that produces it), `docs/data_sources.md`, `docs/state_of_the_art.md`, `README.md`, and the run log in `results/experiments_meta.json`. |

## 2. Datasets, and why they are the reference

The brief requires "the official reference datasets" and "the common standard
evaluation scenario" but the 2-page call does not name either. We therefore use
two national/international *authoritative* archives, chosen so that the sensor and
the reference are produced by independent organisations — otherwise a reference
comparison is circular:

| Role | Source | Product / endpoint | Licence |
| --- | --- | --- | --- |
| Sensor (the data under attack) | NOAA NCEI GHCN-Daily via the Access Data Service | `daily-summaries`, `TMAX`, `TMIN`, `PRCP`, metric units, QC attributes retained | US Government open data (public domain) |
| Independent reference | NASA POWER (LARC) daily point API, community `AG` | `T2M`, `T2M_MAX`, `T2M_MIN`, `RH2M`, `WS2M`, `PRECTOTCORR`, `ALLSKY_SFC_SW_DWN`, `CLRSKY_SFC_SW_DWN` | NASA open data |

Twelve stations in nine countries (`config/sites.yaml`), 34 (site, variable) pairs,
2015–2019 for calibration and 2020–2024 for evaluation. The exact request URLs,
the missing-value conventions (`-999.0` for POWER, omitted days for GHCN) and the
QC-flag handling are in `docs/data_sources.md`; the fetched payload is committed
and hash-pinned (`data/bundle/MANIFEST.sha256` + `MANIFEST.sig`), so every number
in the report is reproducible offline.

**If the organisers name a different official dataset or a prescribed scenario,
that is a substitution of data, not of method:** `src/tsfdi/data_fetch.py` +
`config/sites.yaml` define the only two places where a source is named, and the
pipeline downstream reads `data/bundle/` only. Nothing in `sp/`, `detect/`,
`diagnose/` or `eval/` refers to a provider.

## 3. The standard evaluation scenario (the common scenario, defined)

* **Injection geometry.** Onset uniform in the evaluation period with a 120-day
  warm-up margin and 60 days of attack remaining; magnitudes in units of the
  robust residual scale σ of the clean series (so "1 σ" means the same thing for
  temperature in °C and rain in mm).
* **Standard profile** (the specification's 12 % / 15 % / 7-day profile, the
  headline of Table 1): a 12 % step (as a percentage of the site's seasonal range
  for temperature, multiplicative for precipitation), a 15 % ramp with a 90-day
  rise and 60-day hold, and a 7-day replay of a window 30–400 days earlier.
* **Families.** step, ramp, replay, scaling, noise-shaped (adversarial), plus
  stuck-at, dropout, spike and slow calibration drift (faults).
* **Adversary levels.** A0 (scripted, knows the variable but not the detector),
  A1 (knows the detector family and the calibration period), A2 (adaptive: probes
  the detector and optimises inside the false-alarm budget), A3 (key compromise /
  provenance bypass, analysed rather than assumed away).
* **Operating point.** Block-bootstrap thresholds at 1.0 alarm per sensor-year,
  with the 0.1 / 0.5 / 1.0 / 2.0 budgets also published (`results/calibration.csv`).
* **Held out from every tuning decision.** Stations MIAMI_INTL and NAVACERRADA;
  attack families scaling, noise-shaped and A2. Calibration uses 2015–2019 and
  evaluation 2020–2024, so the split is temporal as well as spatial.
* **Parameters** are not prose: they are `config/attacks.yaml` (families,
  magnitudes, onset rules, held-out list) and `config/detectors.yaml`
  (calibration, cooldown, budgets). Changing the scenario means changing those
  two files and re-running `make all`.

## 4. What the brief asks to be reported, and where each number is

| Asked | Reported in |
| --- | --- |
| Detection quality | Table 1 (standard profile), Table 2 (by family and magnitude), Table 2b (replay by length), Table 6 (cross-site); `docs/report.md` §5–§6. |
| False alarms | Table 3 (per pair, events per sensor-year, including natural extremes), Table 7 (availability), and the baseline comparison Table 6.5/6.6. |
| Latency | Table 5 (per-layer, per reading, streaming path) — 14.78 ms/reading end-to-end on 2 vCPU. |
| Memory | Table 5 (peak traced memory, 321 MB). |
| Security overhead | `results/bench_provenance.csv` — sign/verify µs per record, bytes added, throughput ratio; `docs/report.md` §3 and §7. |
| Feasibility, affordability, local execution | `docs/reproduce.md` (runtime table, 2-vCPU/4 GB target, no GPU, no network after the bundle), `docs/report.md` §7. |

## 5. Rules

* Student Branch only, one track, ≤ 5 members — organisational, not recorded here.
* Phase 1 is anonymous: no names, unit names, logos or identifying repository
  metadata. The repository is committed under a neutral identity, contains no
  personal paths, e-mail addresses or institutional names (checked by grep over
  every tracked text file), and the provenance keys under `data/keys/` are not
  committed while the public anchor is.
* Authentic public data only: both sources above are open and unmodified; the
  injections are synthetic and are generated from the clean series at run time,
  never stored as if they were observations.
