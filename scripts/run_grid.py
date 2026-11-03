"""Run the generalisation grid for one variable group and write its own raw file.

Two invocations run concurrently on a 2-vCPU machine (one per core) and the
results are concatenated into ``results/raw/table2_grid_raw.csv`` by
``scripts/merge_grid.py``.  Splitting by variable keeps the files independent, so
an interrupted run still leaves usable partial results.

Usage: python scripts/run_grid.py <out_name> <variable> [n_per_cell] [sites...]
"""
import sys, time
sys.path.insert(0, "src")
import pandas as pd

from tsfdi import config
from tsfdi.eval import experiments as E
from tsfdi.eval.runner import generalization_grid

out = sys.argv[1]
variable = sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 30
sites = sys.argv[4:] or None
if sites is None:
    sites = {"TMAX": E.GRID_SITES_TMAX, "TMIN": E.GRID_SITES_TMIN,
             "PRCP": E.GRID_SITES_PRCP}[variable]
t0 = time.time()
df = generalization_grid(sites=sites, variables=(variable,), magnitudes=(0.5, 1.0, 1.5, 2.0, 3.0),
                         n_per_cell=n)
p = config.results_dir() / "raw" / out
p.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(p, index=False)
print(f"[grid] {variable}: {len(df)} rows in {time.time() - t0:.0f}s -> {p.name}", flush=True)
