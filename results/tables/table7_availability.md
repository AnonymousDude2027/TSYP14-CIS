# Table 7 - the availability channel (D7): how often the record goes silent

A run of missing days longer than 1.5x the longest gap in the calibration period is a fault: no statistical detector can see a silenced sensor, because there is no residual to threshold. The rate below is a property of the stations (real outages), not a false-alarm budget item, which is why these events are reported separately from Table 3 and never merged into the statistical alarm stream.

| variable   |   pairs |   missing_fraction |   events_per_year |   events_max |   longest_gap_days |
|:-----------|--------:|-------------------:|------------------:|-------------:|-------------------:|
| PRCP       |      10 |             0.0394 |                 0 |       1.5993 |                6   |
| TMAX       |      12 |             0.0541 |                 0 |       2.9988 |                9.5 |
| TMIN       |      12 |             0.0545 |                 0 |       0.1999 |                9.5 |
