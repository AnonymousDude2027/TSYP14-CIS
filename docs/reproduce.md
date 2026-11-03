# Reproducing every number in the report

Everything needed is in this repository. The data bundle is committed and
hash-pinned, so no command in the default pipeline touches the network.

## 1. Environment

Target: a fresh Ubuntu 22.04/24.04 machine, 4 GB RAM, 2 vCPU, no GPU, no
compiler toolchain. Python ≥ 3.11.

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt        # pinned versions
```

`requirements.txt` pins every runtime dependency (numpy, pandas, scipy,
scikit-learn, PyWavelets, cryptography, matplotlib, PyYAML, ruptures, tabulate).
Three optional files add what the runtime does not need:

| File | Contents | Needed for |
| --- | --- | --- |
| `requirements-dev.txt` | `pytest` | `make test` |
| `requirements-app.txt` | `streamlit` | `make dashboard` |
| `requirements-report.txt` | `pypdf` | `make report-pdf` (plus XeLaTeX, below) |

`pyod` is not required: the Isolation-Forest baseline B3 uses `scikit-learn`'s
implementation when `pyod` is absent. The dashboard is not required either —
`results/static_report.html` is self-contained and needs no Python at all.
Nothing else is imported.

## 2. One command

```bash
make all          # data-verify -> gonogo -> calibrate -> experiments -> report -> verify -> static
```

`make all` is a wrapper over the CLI; each stage can be run alone and writes its
own artefacts:

| Target | Command | Output |
| --- | --- | --- |
| `make data-verify` | `python -m tsfdi data-verify` | verifies the signed manifest |
| `make gonogo` | `python -m tsfdi gonogo` | `results/gonogo.{csv,md}` |
| `make calibrate` | `python -m tsfdi calibrate` | `results/calibrate_all.log` (not committed), model cache |
| `make experiments` | `python -m tsfdi experiments` | `results/raw/*.csv` (tidy, one row per cell) |
| `make report` | `python -m tsfdi report-figures` then `report-text` | `results/tables/*.csv/.md`, `results/figures/*.png`, `docs/report.md` |
| `make report-pdf` | `python -m tsfdi report-pdf` | `docs/report.pdf`, `docs/appendix.pdf`, `docs/documentation.pdf` and their `.tex` sources (XeLaTeX) |
| `make verify` | `python -m tsfdi verify` | exit code 0/1 and the metric-by-metric diff on stdout |
| `make bench` | `python -m tsfdi bench` | `results/bench_provenance.{csv,json}` |
| `make test` | `python -m pytest tests/` | unit and integration tests (81, no network; needs `requirements-dev.txt`) |
| `make dashboard` | `streamlit run src/tsfdi/app/dashboard.py` | interactive demo |
| `make static` | `python -m tsfdi report-static` | `results/static_report.html` (no JS, self-contained) |

The PDFs need XeLaTeX and a handful of packages, none of which the pipeline
imports: `xelatex` with `booktabs`, `longtable`, `titlesec`, `enumitem`,
`caption`, `fvextra` and the `tex-gyre` fonts (`texlive-xetex` +
`texlive-latex-recommended` + `texlive-fonts-recommended` on Debian/Ubuntu, or
`tlmgr install booktabs titlesec enumitem caption fvextra tex-gyre microtype` on
a minimal TeX Live). The committed PDFs are the artefacts of record; re-running
the target is only needed if the Markdown changes.

`bash run.sh` does the same once the environment above is installed (it does not
create the virtual environment itself), and accepts `--quick` for a smoke run of
the whole pipeline in a few minutes.

## 3. Determinism

* One seed: `config.SEED` (`src/tsfdi/config.py`). Every bootstrap replicate,
  phase-randomised surrogate, injection and train/test split derives from it.
* The (site, variable) models are cached in `results/cache/models`; deleting the
  directory and re-running `make calibrate` reproduces the same thresholds to
  bit level on the same data manifest.
* `make verify` compares the metrics computed from the current `results/` tree
  with `expected_results.json` at the repository root. Because `make all`
  regenerates `results/` first, the comparison is a genuine reproduction check;
  run on its own against the committed results it only confirms internal
  consistency, which is why it is documented as the last step of the chain. Rate metrics are compared with
  an absolute tolerance (a rate of 0.02 alarms/sensor-year cannot be compared
  with a 2 % relative tolerance), proportions with a 0.02 absolute tolerance.

## 4. Data

`data/bundle/` contains 24 CSVs (12 stations × up to 3 variables) plus
`data/bundle/FETCH_INFO.json`, `MANIFEST.sha256` and its Ed25519 signature
`MANIFEST.sig`. The reference files came from NASA POWER
(`/api/temporal/daily/point`, community AG); the sensor files from the
GHCN-Daily access API (`dataset=daily-summaries`, metric units, quality flags
kept). `data/bundle/FETCH_INFO.json` records the exact request strings, the retrieval date,
and the row counts, so the bundle can be rebuilt from scratch if a network is
available:

```bash
python -m tsfdi fetch --force
```

Re-fetching is **not** part of `make all`: the providers revise their archives,
and a reproducibility claim that depends on a live API is not a reproducibility
claim. The pinned bundle is the artefact of record.

## 5. Runtimes on the reference machine

Measured on 2 vCPU / 4 GB (no GPU). Figures are wall-clock and exclude the
one-off data fetch.

| Stage | Runtime |
| --- | --- |
| `data-verify` | < 1 s |
| `gonogo` (34 pairs, incl. Monte-Carlo minimum detectable bias) | see `results/gonogo.md` header |
| `calibrate` (34 pairs, 14 detectors, 4 budgets) | ≈ 28 min |
| `experiments` (full grid) | ≈ 5.6 h single-core; the shipped run used the split below |
| `scripts/run_grid.py results/raw/grid_tmax.csv TMAX 20` | ≈ 1.3 h (one core) |
| `scripts/run_grid.py results/raw/grid_secondary.csv TMIN 10` then `PRCP` | ≈ 0.4 h + 0.2 h (second core) |
| `scripts/run_rest.py` (profile, FA, diagnosis, baselines, ablation, efficiency, evasion) | ≈ 0.9 h |
| `report-figures` | < 30 s |
| `bench` (100 000 records) | < 60 s |
| `tests` | see `make test` |

The generalisation grid is the only stage that does not fit comfortably in one
process on the reference machine, so it is split by variable and run two at a time:

```
python3 scripts/run_grid.py results/raw/grid_tmax.csv TMAX 20      # core 1
python3 scripts/run_grid.py results/raw/grid_secondary.csv TMIN 10 # core 2, then PRCP 10
python3 scripts/merge_grid.py            # -> results/raw/table2_grid_raw.csv
python3 scripts/run_rest.py              # everything except the grid
```

`scripts/README.md` documents this parallel path and the constraint behind it: a
change to how a family's magnitude is calibrated invalidates that family's rows
and nothing else, so the suite is re-run per family rather than wholesale.

The dominant cost is the bootstrap calibration, which is O(replicates × days ×
detectors) per pair and was deliberately traded against Monte-Carlo error (see
`docs/design.md` §5).

## 6. Repository layout

```
config/          YAML configuration: sites, detectors, attacks, profile, decisions
data/bundle/     pinned CSVs + signed manifest (the artefact of record)
docs/            threat model, design decisions, this file, report
results/         every generated artefact (raw, tables, figures, cache)
scripts/         optional parallel entry points (see scripts/README.md)
src/tsfdi/       the package (sp, detect, attacks, provenance, diagnose, eval, app)
tests/           unit and integration tests (pytest)
Makefile         the seven stages above
run.sh           the same, from a prepared shell
```
