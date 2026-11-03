# Table baselines-b - false alarms of every baseline on the clean series

Alarm events per sensor-year on the untouched 2020-2024 test period, with the same event definition and 14-day cooldown as the proposed suite. B0's threshold is calibrated to the same 1/yr budget by the textbook quantile rule; B1 uses its published-style fixed 5-sigma CUSUM threshold (it has no calibration step); B2 has no threshold (every change point is an alarm); B3 is an isolation forest at its default 1 % contamination. This is the column that decides whether a detection rate means anything.

| baseline                     |   sites |   fa_median |     fa_max |
|:-----------------------------|--------:|------------:|-----------:|
| B3_isolation_forest[sklearn] |       4 |      0.2999 |     0.5998 |
| B2_ruptures_pelt             |       4 |      0.7997 |     1.3994 |
| PROPOSED                     |       4 |      2.1991 |     3.1987 |
| B0_fixed_threshold           |       4 |      6.4973 |     8.1966 |
| B1_cusum_unwhitened          |       4 |   8218.12   | 12783.8    |
