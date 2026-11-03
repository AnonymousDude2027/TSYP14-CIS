# TSFDI — falsified-data detection for in-situ climate sensors

Track 2 (false-data injection on in-situ sensor readings). The system treats a
weather station as **untrusted** and an independent reanalysis product as a
**hash-pinned trusted reference**, then decides — with thresholds calibrated to a
*false-alarm budget* — whether a day's reported value is consistent with the
reference, the sensor's own history and its spectral fingerprint.

Everything is offline-reproducible on a 4 GB / 2-vCPU machine with no GPU:

```bash
make setup          # pinned requirements in a venv
make all            # data-verify -> gonogo -> calibrate -> experiments -> report -> verify
make dashboard      # interactive demo (Streamlit); `make static` is the no-Streamlit fallback
```

`bash run.sh --quick` runs the whole chain in a few minutes with a reduced grid.

The three PDFs are the submitted artefacts. They are typeset from the Markdown by
`make report-pdf` (XeLaTeX), so every number in them is the generated one:

| PDF | Pages | Contents |
| --- | --- | --- |
| `docs/report.pdf` | 8 | the report: threat model, architecture and trust anchor, signal processing, detection, evaluation and evasion, efficiency, limitations |
| `docs/appendix.pdf` | 6 | full per-site and per-magnitude tables (A.1–A.15) |
| `docs/documentation.pdf` | 21 | README + compliance + threat model + design decisions + data sources + state of the art + reproducibility guide, bound together |

`docs/compliance.md` maps the six required deliverables and the four Phase-1
submission items to the files that satisfy them, names the two data archives and
defines the standard evaluation scenario in full.

## What is here

| Path | Contents |
| --- | --- |
| `src/tsfdi/provenance/` | L0: canonical fixed-point records, SHA-256 hash chain, Ed25519 signatures, signed data manifest, streaming verifier with typed errors |
| `src/tsfdi/sp/` | L1: SP1 conditioning (units, lag, gaps), SP2 robust expectation (Huber + harmonics), SP3 whitening (AR + seasonal scale + rank-Gauss + causal bias/scale guards), SP4 wavelets, SP5 matched filter, SP6 spectral fingerprint |
| `src/tsfdi/detect/` | L2: D1 step, D2 drift, D3 replay, D4 fault envelope, D6 fingerprint; block-bootstrap false-alarm calibration; the single `SchmittAlarm` event rule |
| `src/tsfdi/attacks/` | the injector (step, ramp, replay, scaling, noise-shaped, faults, natural events) and the adversary hierarchy A0–A3 |
| `src/tsfdi/diagnose/` | L4/L5: evidence, rule table, Tier-2 logistic regression, action ladder, decision-impact estimate, templated explanations |
| `src/tsfdi/eval/` | metrics (no point-adjusted F1), runner, baselines B0–B3, ablations, experiment orchestration |
| `src/tsfdi/pipeline.py` | the streaming L0→L5 path used by the CLI, the dashboard and the tests |
| `docs/` | report (`report.pdf`), appendix (`appendix.pdf`), documentation bundle (`documentation.pdf`) and their Markdown sources: threat model, design decisions, data sources, state of the art, reproducibility guide, compliance |
| `data/bundle/` | 24 CSVs + `data/bundle/FETCH_INFO.json` + signed `MANIFEST.sha256`/`MANIFEST.sig` |
| `results/` | tables, figures, raw per-injection CSVs, model cache, benchmark |

## Commands

```
python -m tsfdi fetch          # (network) rebuild the bundle; not part of make all
python -m tsfdi data-verify    # verify the signed manifest
python -m tsfdi sites          # re-measure station completeness
python -m tsfdi gonogo         # go/no-go gate (spec section 6) -- run before quoting anything
python -m tsfdi calibrate      # fit + calibrate 34 (site, variable) models
python -m tsfdi stream CARTHAGE TMAX --attack step --magnitude 3   # day-by-day L0-L5
python -m tsfdi experiments    # standard profile, grid, false alarms, baselines, ablations, evasion
python -m tsfdi report-figures # tables + figures from results/raw
python -m tsfdi report-text    # docs/report.md from the tables
python -m tsfdi bench          # provenance overhead over 100 000 records
python -m tsfdi verify         # compare against expected_results.json
python -m tsfdi selftest       # the unit-test suite
```

## Headline results (regenerate with `make all`)

See `docs/report.md` for the full evaluation. The short version, all measured on
the 2020–2024 test period with thresholds calibrated on 2015–2019 only:

* the organizers' standard profile (12 % step, 15 % ramp, 7-day replay) is
  detected on every station/variable pair;
* false alarms on clean data have a median of ~1.6 per sensor-year across 34
  pairs, reported per station and separately inside natural extremes;
* the damage a black-box adaptive attacker can insert at a 1-alarm/year budget is
  quantified in σ·sensor-days, and the evasion curve is reported against the budget;
* provenance costs ~0.2 ms per reading and adds ~200 bytes per record.

## Honesty notes

Four failure modes were found on real data that do not appear in synthetic
benchmarks (a non-monotone threshold calibration, a non-stationary
sensor/reference bias, a degenerate matched filter on zero-inflated precipitation,
and a replay detector that matched dry spells). Each is documented in
`docs/design.md` with the measurement that exposed it and the fix that removed it.
Where the system does not deliver — A1c and A2 adversaries, stations whose
relationship to the reference changes, precipitation as a restricted claim — the
limitation is stated in `docs/report.md` §8 and `docs/threat_model.md` §5 instead
of being tuned away.

## Anonymity

No names, affiliations, logos or identifying metadata in code, data or documents.

## What the numbers rest on

* **One magnitude axis.** Every detection rate in the report is quoted in units of
  the robust residual scale the detector actually sees, and every family is
  calibrated so that a nominal `m` sigma is a realised `m` sigma (additive
  families as an offset in the SP1 coordinate, multiplicative families as a gain
  computed from the data). `docs/design.md` §3f explains the three bugs that made
  the first version of the grid misleading, and the parity that the fix restored.
* **False alarms are measured, not assumed.** Table 3 reports alarm events per
  sensor-year out of sample, including inside natural extremes; the baselines get
  their own false-alarm column (Table baselines-b) so that a detection rate can be
  read against the rate it actually costs.
* **Every failure mode has a channel.** The detectors see values; a silenced
  sensor produces no value to see, so D7 tests the reporting cadence itself and
  reports an outage fault (Table 7). It is kept out of the statistical alarm
  stream on purpose: a broken modem must not consume the detectors' budget.
* **Coverage is auditable.** `results/excluded_pairs.csv` names every (site,
  variable) pair that cannot be modelled and why; no results table ever carries an
  error message in place of a result.
* **Every metric is checked.** `python -m tsfdi verify` re-reads
  `expected_results.json` (44 metrics) with per-metric tolerances; `pytest` covers
  the SP chain, the detectors, the injector, provenance, the streaming path, the
  diagnosis rules and the report builder.
