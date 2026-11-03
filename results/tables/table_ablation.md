# Table ablation - what each component contributes

Each variant removes exactly one component and re-runs the same false-alarm calibration and the same injections. The columns are meant to be read together, because the components do not all pay in the same currency: removing whitening leaves detection power nearly intact and makes the alarms arrive sooner, but it raises clean-data false alarms (2.3 to 4.4 per sensor-year), so whitening buys false-alarm control at the cost of speed; replacing the reference expectation with a gradient-boosted model raises false alarms further still (5.3); and removing the reference model leaves detection unchanged on this grid, which is why the anchor's contribution is argued from the adversarial families in Section 6.4 and not from marginal steps. The grid is the specification's (TMAX, step and ramp, 1-3 sigma); a per-family ablation is the natural extension and is not claimed.

| variant         |   magnitude_sigma |   n |   n_detected |   delay_median |   delay_q1 |   delay_q3 |   fa_per_year |   build_seconds |   detection_rate |   never_detected_rate |
|:----------------|------------------:|----:|-------------:|---------------:|-----------:|-----------:|--------------:|----------------:|-----------------:|----------------------:|
| full            |               1   |  24 |           16 |          51    |     12     |     56     |        2.2991 |         44.433  |           0.6667 |                0.3333 |
| full            |               1.5 |  24 |           22 |          33.75 |      9.625 |     40.375 |        2.2991 |         44.433  |           0.9167 |                0.0833 |
| full            |               2   |  24 |           23 |          16    |      8     |     27.625 |        2.2991 |         44.433  |           0.9583 |                0.0417 |
| full            |               3   |  24 |           23 |           6    |      4.375 |      7.25  |        2.2991 |         44.433  |           0.9583 |                0.0417 |
| gbm_expectation |               1   |  24 |           21 |          39.25 |     25.625 |     51.375 |        5.2978 |         44.966  |           0.875  |                0.125  |
| gbm_expectation |               1.5 |  24 |           22 |          19.75 |     13.25  |     35.5   |        5.2978 |         44.966  |           0.9167 |                0.0833 |
| gbm_expectation |               2   |  24 |           23 |           7.75 |      6.625 |     16.5   |        5.2978 |         44.966  |           0.9583 |                0.0417 |
| gbm_expectation |               3   |  24 |           23 |           6.25 |      4.875 |      9.75  |        5.2978 |         44.966  |           0.9583 |                0.0417 |
| no_fingerprint  |               1   |  24 |           17 |          47    |     19.75  |     54.75  |        1.5993 |         28.3219 |           0.7083 |                0.2917 |
| no_fingerprint  |               1.5 |  24 |           20 |          28    |      9.125 |     40.125 |        1.5993 |         28.3219 |           0.8333 |                0.1667 |
| no_fingerprint  |               2   |  24 |           23 |          16    |      8     |     23.125 |        1.5993 |         28.3219 |           0.9583 |                0.0417 |
| no_fingerprint  |               3   |  24 |           23 |           6    |      4.375 |      7.25  |        1.5993 |         28.3219 |           0.9583 |                0.0417 |
| no_reference    |               1   |  24 |           20 |          32.5  |     29     |     45.5   |        2.1991 |         44.7058 |           0.8333 |                0.1667 |
| no_reference    |               1.5 |  24 |           23 |          30.25 |     23     |     41.125 |        2.1991 |         44.7058 |           0.9583 |                0.0417 |
| no_reference    |               2   |  24 |           23 |          20    |      6.75  |     27.75  |        2.1991 |         44.7058 |           0.9583 |                0.0417 |
| no_reference    |               3   |  24 |           24 |          14    |      9.125 |     17.25  |        2.1991 |         44.7058 |           1      |                0      |
| no_whitening    |               1   |  24 |           18 |          27.25 |     23.125 |     48.375 |        4.3982 |         44.5154 |           0.75   |                0.25   |
| no_whitening    |               1.5 |  24 |           21 |           7.5  |      5.25  |     22.5   |        4.3982 |         44.5154 |           0.875  |                0.125  |
| no_whitening    |               2   |  24 |           21 |           5.25 |      3.5   |     10.75  |        4.3982 |         44.5154 |           0.875  |                0.125  |
| no_whitening    |               3   |  24 |           21 |           3.25 |      2.25  |      4.25  |        4.3982 |         44.5154 |           0.875  |                0.125  |
