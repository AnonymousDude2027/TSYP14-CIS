# scripts/

Thin, documented entry points for running the experiment suite **in parallel**.
They are not the canonical path — `make experiments` runs the same steps
serially through `python -m tsfdi experiments` — but the generalisation grid is
the expensive part (≈ 1.3 h on one core), so it is split by variable and merged:

```bash
python3 scripts/run_grid.py results/raw/grid_tmax.csv TMAX 20        # core 1
python3 scripts/run_grid.py results/raw/grid_secondary.csv TMIN 10   # core 2, then PRCP
python3 scripts/merge_grid.py        # -> results/raw/table2_grid_raw.csv
python3 scripts/run_rest.py          # every other step, cheapest-first
```

`calibrate_all.py` is the same job as `make calibrate` with an optional
`--resume` flag for interrupted runs. `make report` / `make report-figures` must
be re-run after any of these so that `results/tables/` matches `results/raw/`.
