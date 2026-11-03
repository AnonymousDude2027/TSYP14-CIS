"""Provenance overhead benchmark (specification section 10).

Measures, over N synthetic records:

* signing time per record (microseconds);
* verification time per record (microseconds);
* bytes added per record by the provenance layer (the JSON record with signature
  and hash, minus the bare measurement);
* throughput with and without the provenance layer;
* peak traced memory.

The numbers in the report come from this script only.  Signing dominates because
Ed25519 on a software anchor is a curve operation; verification is the cheaper
direction, which is the one the operator pays for at scale.
"""
from __future__ import annotations

import json
import time
import tracemalloc

import numpy as np

from . import config
from .provenance.anchor import SoftwareAnchor
from .provenance.record import from_values, sign_record, verify_record
from .provenance.verify import ProvenanceVerifier


def run_benchmark(n_records: int = 100_000, seed: int = config.SEED) -> list[dict]:
    rng = np.random.default_rng(seed)
    # use the *same* anchor the pipeline and the data bundle use, so that the
    # fingerprint quoted in the benchmark is the fingerprint quoted everywhere
    anchor = SoftwareAnchor.load_or_create(config.path("data", "keys", "anchor.pem"))
    values = 20 + 10 * rng.random(n_records)
    dates = np.datetime64("2020-01-01") + np.arange(n_records).astype("timedelta64[m]")

    tracemalloc.start()
    # ---- signing ---------------------------------------------------------
    t0 = time.perf_counter()
    prev = "0" * 64
    signed = []
    for i in range(n_records):
        rec = from_values("BENCH0000001", i, dates[i], "TMAX", float(values[i]), prev_hash=prev)
        sign_record(rec, anchor.private_key)
        prev = rec.hash
        signed.append(rec)
    t_sign = time.perf_counter() - t0
    cur, peak = tracemalloc.get_traced_memory()

    # ---- verification ----------------------------------------------------
    t0 = time.perf_counter()
    verifier = ProvenanceVerifier(anchor.public_key(), skew_seconds=10 ** 9)
    accepted = sum(1 for rec in signed if verifier.check(rec))
    t_verify = time.perf_counter() - t0

    # ---- baseline: no provenance ----------------------------------------
    t0 = time.perf_counter()
    _ = [{"sensor_id": "BENCH0000001", "seq": i, "ts": str(dates[i]),
          "variable": "TMAX", "value": f"{values[i]:.3f}"} for i in range(n_records)]
    t_bare = time.perf_counter() - t0
    tracemalloc.stop()

    bare_bytes = len(json.dumps({"sensor_id": "BENCH0000001", "seq": 0,
                                 "ts": "2020-01-01T00:00:00Z", "variable": "TMAX",
                                 "value": "20.000"}).encode())
    full_bytes = len(json.dumps(signed[0].as_dict()).encode())

    rows = [
        {"metric": "n_records", "value": n_records, "unit": "records"},
        {"metric": "sign_us_per_record", "value": 1e6 * t_sign / n_records, "unit": "microseconds"},
        {"metric": "verify_us_per_record", "value": 1e6 * t_verify / n_records, "unit": "microseconds"},
        {"metric": "sign_verify_us_per_record", "value": 1e6 * (t_sign + t_verify) / n_records, "unit": "microseconds"},
        {"metric": "bytes_added_per_record", "value": full_bytes - bare_bytes, "unit": "bytes"},
        {"metric": "bytes_full_record", "value": full_bytes, "unit": "bytes"},
        {"metric": "throughput_with_provenance", "value": n_records / (t_sign + t_verify), "unit": "records/s"},
        {"metric": "throughput_without_provenance", "value": n_records / t_bare, "unit": "records/s"},
        {"metric": "security_overhead_ratio", "value": (t_sign + t_verify) / t_bare, "unit": "x"},
        {"metric": "peak_traced_mb", "value": peak / 1e6, "unit": "MB"},
        {"metric": "records_accepted", "value": accepted, "unit": "records"},
        {"metric": "anchor_fingerprint", "value": anchor.fingerprint(), "unit": "sha256-hex-32"},
    ]
    out = config.results_dir() / "bench_provenance.csv"
    import pandas as pd

    pd.DataFrame(rows).to_csv(out, index=False)
    (config.results_dir() / "bench_provenance.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[bench] wrote {out}")
    return rows
