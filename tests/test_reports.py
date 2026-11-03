"""The reporting and verification layer: tables come from data, tolerances are per metric."""
import json

import numpy as np
import pandas as pd
import pytest

from tsfdi import config, verify
from tsfdi.report_text import build as build_report


def test_report_builds_from_results(tmp_path):
    out = build_report()
    text = out.read_text(encoding="utf-8")
    assert "## 1. Problem and threat model" in text
    assert "{{" not in text.split("**Abstract.**")[1][:20000]   # no unfilled placeholder in the body
    assert len(text.split()) > 800


def test_tolerance_is_absolute_for_rates_and_relative_for_measurements():
    lo, hi = verify._tolerance_for("table2.step.1.0.detection_rate", 0.90, 0.02)
    # one injection out of a 20-injection cell is 0.05, so the band must cover a
    # single boundary flip and no more
    assert 0.06 <= hi - lo <= 0.12 + 1e-9
    lo2, hi2 = verify._tolerance_for("calibration.sigma_over_range_median", 0.06, 0.02)
    # a continuous quantity gets the 0.02 floor, i.e. a much tighter band than a rate
    assert hi2 - lo2 <= 0.05 < hi - lo
    lo, hi = verify._tolerance_for("bench.sign_us_per_record", 15.0, 0.02)
    assert hi - lo >= 4.0                       # timing is machine dependent


def test_compare_expected_round_trips(tmp_path, monkeypatch):
    p = config.repo_root() / verify.EXPECTED
    backup = p.read_text() if p.is_file() else None
    try:
        assert verify.compare_expected(regenerate=True)
        assert verify.compare_expected()
        # perturb one metric beyond its band and expect a failure
        d = json.loads(p.read_text())
        first = next(iter(d))
        d[first] = d[first] * 3 + 1
        p.write_text(json.dumps(d), encoding="utf-8")
        assert not verify.compare_expected()
    finally:
        if backup is not None:
            p.write_text(backup, encoding="utf-8")
        elif p.is_file():
            p.unlink()


def test_figures_are_written_when_tables_exist():
    from tsfdi import reporting

    reporting.build_all()
    figs = list((config.results_dir() / "figures").glob("*.png"))
    assert figs, "no figures were produced"


def test_no_point_adjusted_f1_anywhere():
    """The specification forbids it, so the string should not appear in the code."""
    import pathlib

    root = config.repo_root() / "src"
    hits = [p.name for p in root.rglob("*.py")
            if "point_adjust" in p.read_text(encoding="utf-8")
            and "forbid" not in p.read_text(encoding="utf-8")]
    assert not hits, hits


def test_excluded_pairs_file_exists_and_tables_never_contain_error_rows():
    """A pair that cannot be modelled must be named in excluded_pairs.csv, never
    written into a results table as an ERROR row."""
    from tsfdi.eval import runner

    runner._EXCL = None                      # noqa: SLF001 - force a re-scan
    pairs = runner.evaluable_pairs()
    assert len(pairs) == 34
    p = config.results_dir() / "excluded_pairs.csv"
    assert p.is_file()
    assert list(pd.read_csv(p).columns) == ["site", "variable", "reason"]
    for name in ("table1_standard_profile", "table2_grid_raw", "table3_false_alarms_raw"):
        f = config.results_dir() / "raw" / f"{name}.csv"
        if f.is_file():
            d = pd.read_csv(f)
            assert "family" not in d.columns or "ERROR" not in set(d["family"].dropna())


def test_variable_labels_in_the_abstract_keep_their_case():
    """str.capitalize() lowercases the rest of the string: it silently turned
    'PRCP 0.48' into 'prcp 0.48' in the abstract."""
    from tsfdi.report_text import block_abstract_numbers

    txt = block_abstract_numbers()
    if "PRCP" in txt or "prcp" in txt:
        assert "PRCP" in txt and "prcp" not in txt
