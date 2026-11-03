# Table 1 - the organizers' standard profile (12 % step, 15 % ramp, 7-sample replay)

Percentages are taken over each site's seasonal range (the peak-to-trough amplitude of the harmonic climatology, SP2); for precipitation they are multiplicative factors. The sigma-equivalent columns give the same attack in units of the robust residual scale, because a percentage of a temperature is not physically meaningful on its own. n = number of injections, pooled over the sites listed in the raw file.

| variable   | family   |   n |   detection_rate |   never_detected_rate |   delay_median |   delay_q1 |   delay_q3 |   auprc_median |   magnitude_sigma_median |   magnitude_sigma_min |   magnitude_sigma_max |
|:-----------|:---------|----:|-----------------:|----------------------:|---------------:|-----------:|-----------:|---------------:|-------------------------:|----------------------:|----------------------:|
| PRCP       | ramp     |  50 |           0.48   |                0.52   |           74.5 |       35.5 |     116.5  |         0.0476 |                   0.75   |                0.75   |                0.75   |
| PRCP       | replay   |  50 |           0.4    |                0.6    |            4.5 |        2   |      13    |         0.0048 |                 nan      |              nan      |              nan      |
| PRCP       | step     |  50 |           0.56   |                0.44   |           49   |       13.5 |      75    |         0.0453 |                   0.6    |                0.6    |                0.6    |
| TMAX       | ramp     |  60 |           0.8833 |                0.1167 |           54   |       37   |      90    |         0.2885 |                   2.315  |                1.1715 |                4.4976 |
| TMAX       | replay   |  60 |           0.8667 |                0.1333 |            2.5 |        0   |       6    |         0.1601 |                 nan      |              nan      |              nan      |
| TMAX       | step     |  60 |           0.8833 |                0.1167 |           12   |        5   |      35    |         0.4711 |                   1.852  |                0.9372 |                3.5981 |
| TMIN       | ramp     |  60 |           0.9333 |                0.0667 |           70.5 |       43   |      92.75 |         0.2134 |                   1.4546 |                0.895  |                2.2904 |
| TMIN       | replay   |  60 |           0.8167 |                0.1833 |            5   |        1   |       6    |         0.0122 |                 nan      |              nan      |              nan      |
| TMIN       | step     |  60 |           0.8333 |                0.1667 |           26   |       14   |      42.75 |         0.3568 |                   1.1637 |                0.716  |                1.8323 |
