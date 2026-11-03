# Table baselines - B0-B3 on the same injections

B0 fixed |z|>h threshold (h calibrated to the same false-alarm budget); B1 CUSUM on the unwhitened residual; B2 offline ruptures/PELT (no threshold, so its row is the honest limit of a batch method); B3 Isolation Forest on 7-day window features.

| baseline                     | family   |   magnitude_sigma |   n |   detection_rate |   never_detected_rate |   delay_median |   delay_q1 |   delay_q3 |   auprc_median |
|:-----------------------------|:---------|------------------:|----:|-----------------:|----------------------:|---------------:|-----------:|-----------:|---------------:|
| B0_fixed_threshold           | ramp     |               1   |  20 |             0.95 |                  0.05 |           24   |       8    |      70    |       nan      |
| B0_fixed_threshold           | ramp     |               1.5 |  20 |             0.95 |                  0.05 |           27   |      11    |      58.5  |       nan      |
| B0_fixed_threshold           | ramp     |               2   |  20 |             0.95 |                  0.05 |           27   |      11    |      43.5  |       nan      |
| B0_fixed_threshold           | ramp     |               3   |  20 |             1    |                  0    |           31.5 |      11.5  |      45    |       nan      |
| B0_fixed_threshold           | ramp     |               5   |  20 |             1    |                  0    |           24   |      15    |      38.25 |       nan      |
| B0_fixed_threshold           | step     |               1   |  20 |             0.9  |                  0.1  |           22   |      13.75 |      44.5  |       nan      |
| B0_fixed_threshold           | step     |               1.5 |  20 |             0.95 |                  0.05 |           13   |       6    |      29.5  |       nan      |
| B0_fixed_threshold           | step     |               2   |  20 |             1    |                  0    |            7.5 |       4.75 |      15    |       nan      |
| B0_fixed_threshold           | step     |               3   |  20 |             1    |                  0    |            3   |       1    |      12.75 |       nan      |
| B0_fixed_threshold           | step     |               5   |  20 |             1    |                  0    |            2.5 |       0    |      10.5  |       nan      |
| B1_cusum_unwhitened          | ramp     |               1   |  20 |             1    |                  0    |           76.5 |      55.75 |     105    |       nan      |
| B1_cusum_unwhitened          | ramp     |               1.5 |  20 |             1    |                  0    |           69.5 |      43.75 |      80.25 |       nan      |
| B1_cusum_unwhitened          | ramp     |               2   |  20 |             1    |                  0    |           55.5 |      42.75 |      68    |       nan      |
| B1_cusum_unwhitened          | ramp     |               3   |  20 |             1    |                  0    |           42.5 |      32.25 |      50.75 |       nan      |
| B1_cusum_unwhitened          | ramp     |               5   |  20 |             1    |                  0    |           25.5 |      18.75 |      32.25 |       nan      |
| B1_cusum_unwhitened          | step     |               1   |  20 |             1    |                  0    |           27   |       7.75 |      44.75 |       nan      |
| B1_cusum_unwhitened          | step     |               1.5 |  20 |             1    |                  0    |            9.5 |       5    |      19    |       nan      |
| B1_cusum_unwhitened          | step     |               2   |  20 |             1    |                  0    |            5   |       3    |       8    |       nan      |
| B1_cusum_unwhitened          | step     |               3   |  20 |             1    |                  0    |            2.5 |       1.75 |       4    |       nan      |
| B1_cusum_unwhitened          | step     |               5   |  20 |             1    |                  0    |            1   |       1    |       1.25 |       nan      |
| B2_ruptures_pelt             | ramp     |               1   |  20 |             0.65 |                  0.35 |           42   |      27    |      62    |       nan      |
| B2_ruptures_pelt             | ramp     |               1.5 |  20 |             0.95 |                  0.05 |           42   |      35    |      62.5  |       nan      |
| B2_ruptures_pelt             | ramp     |               2   |  20 |             1    |                  0    |           46.5 |      37    |      62.25 |       nan      |
| B2_ruptures_pelt             | ramp     |               3   |  20 |             1    |                  0    |           40   |      30    |      57    |       nan      |
| B2_ruptures_pelt             | ramp     |               5   |  20 |             1    |                  0    |           26.5 |      20    |      33    |       nan      |
| B2_ruptures_pelt             | step     |               1   |  20 |             0.6  |                  0.4  |           12.5 |       2    |      88.5  |       nan      |
| B2_ruptures_pelt             | step     |               1.5 |  20 |             1    |                  0    |            3   |       1    |      22    |       nan      |
| B2_ruptures_pelt             | step     |               2   |  20 |             1    |                  0    |            2   |       1    |      88.5  |       nan      |
| B2_ruptures_pelt             | step     |               3   |  20 |             1    |                  0    |            2   |       1    |     119    |       nan      |
| B2_ruptures_pelt             | step     |               5   |  20 |             1    |                  0    |            2   |       1    |     119    |       nan      |
| B3_isolation_forest[sklearn] | ramp     |               1   |  20 |             0.25 |                  0.75 |           58   |      52    |      66    |       nan      |
| B3_isolation_forest[sklearn] | ramp     |               1.5 |  20 |             0.3  |                  0.7  |           70   |      55.5  |      88.25 |       nan      |
| B3_isolation_forest[sklearn] | ramp     |               2   |  20 |             0.3  |                  0.7  |           70   |      51    |      88.25 |       nan      |
| B3_isolation_forest[sklearn] | ramp     |               3   |  20 |             0.6  |                  0.4  |           94.5 |      72    |     107.75 |       nan      |
| B3_isolation_forest[sklearn] | ramp     |               5   |  20 |             0.9  |                  0.1  |           88   |      68    |      96    |       nan      |
| B3_isolation_forest[sklearn] | step     |               1   |  20 |             0.2  |                  0.8  |           45   |      30.25 |      60    |       nan      |
| B3_isolation_forest[sklearn] | step     |               1.5 |  20 |             0.35 |                  0.65 |           21   |      15    |      60.5  |       nan      |
| B3_isolation_forest[sklearn] | step     |               2   |  20 |             0.45 |                  0.55 |           21   |      15    |      55    |       nan      |
| B3_isolation_forest[sklearn] | step     |               3   |  20 |             0.75 |                  0.25 |           19   |       9.5  |      48    |       nan      |
| B3_isolation_forest[sklearn] | step     |               5   |  20 |             1    |                  0    |            8   |       1.75 |      15.5  |       nan      |
| PROPOSED                     | ramp     |               1   |  20 |             0.7  |                  0.3  |           90   |      65    |     116    |         0.0662 |
| PROPOSED                     | ramp     |               1.5 |  20 |             0.8  |                  0.2  |           70.5 |      58.75 |      83    |         0.255  |
| PROPOSED                     | ramp     |               2   |  20 |             0.95 |                  0.05 |           68   |      58.5  |      82.5  |         0.2702 |
| PROPOSED                     | ramp     |               3   |  20 |             1    |                  0    |           50.5 |      42.5  |      66.5  |         0.3091 |
| PROPOSED                     | ramp     |               5   |  20 |             1    |                  0    |           36   |      30.75 |      50.75 |         0.3018 |
| PROPOSED                     | step     |               1   |  20 |             0.75 |                  0.25 |           40   |      25    |      48    |         0.2196 |
| PROPOSED                     | step     |               1.5 |  20 |             0.9  |                  0.1  |           22   |      12.5  |      27    |         0.4461 |
| PROPOSED                     | step     |               2   |  20 |             1    |                  0    |           10   |       6.5  |      20    |         0.4749 |
| PROPOSED                     | step     |               3   |  20 |             1    |                  0    |            6   |       4    |       8    |         0.5101 |
| PROPOSED                     | step     |               5   |  20 |             1    |                  0    |            2   |       1    |       3    |         0.3719 |
