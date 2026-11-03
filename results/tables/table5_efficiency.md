# Table 5 - efficiency: latency and memory

Latency is measured per reading through the real streaming path (L0 provenance, SP1-SP3, L2 detectors, L3 consistency). Memory is peak traced allocation of the streaming run plus the process RSS, i.e. an upper bound that includes the interpreter and the fitted model. The provenance-only cost of the signing layer is in results/bench_provenance.csv (sign + verify microseconds per record, bytes added per record).

| site                | variable   |   n_readings |   ms_per_reading |   peak_python_mb |   rss_mb |   L0_provenance_per_reading |   L2_detectors_per_reading |   L3_consistency_per_reading |   SP3_whitening_per_reading |
|:--------------------|:-----------|-------------:|-----------------:|-----------------:|---------:|----------------------------:|---------------------------:|-----------------------------:|----------------------------:|
| CARTHAGE            | TMAX       |         3653 |          10.499  |          321.23  |  504.652 |                      0.2601 |                     0.4951 |                       0.0147 |                      0.0156 |
| GAFSA               | TMAX       |         3653 |          20.8758 |          321.314 |  510.285 |                      0.3089 |                     0.5478 |                       0.0143 |                      0.0175 |
| NAVACERRADA         | TMAX       |         3653 |           9.5734 |          321.23  |  516.812 |                      0.3794 |                     0.5442 |                       0.0141 |                      0.0202 |
| FELDBERG            | TMAX       |         3653 |           7.9406 |          321.273 |  520.234 |                      0.3938 |                     0.5722 |                       0.0147 |                      0.021  |
| HELSINKI_KAISANIEMI | TMAX       |         3653 |          19.062  |          321.318 |  520.234 |                      0.3934 |                     0.5589 |                       0.0178 |                      0.0199 |
| MIAMI_INTL          | TMAX       |         3653 |          27.7951 |          321.365 |  520.234 |                      0.385  |                     0.5487 |                       0.0249 |                      0.0242 |
