"""Concatenate the per-variable grid files into results/raw/table2_grid_raw.csv."""
import sys
from pathlib import Path
sys.path.insert(0, "src")
import pandas as pd
from tsfdi import config

raw = config.results_dir() / "raw"
parts = [raw / n for n in ("grid_tmax.csv", "grid_secondary.csv")]   # secondary = TMIN+PRCP
frames = [pd.read_csv(p) for p in parts if p.is_file()]
if not frames:
    raise SystemExit("no grid files found")
df = pd.concat(frames, ignore_index=True)
out = raw / "table2_grid_raw.csv"
df.to_csv(out, index=False)
print(f"[grid] merged {sum(len(f) for f in frames)} rows -> {out.name}")
print(df.groupby("variable").size().to_string())
