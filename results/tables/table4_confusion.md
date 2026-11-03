# Table 4 - three-class (plus undetermined) diagnosis confusion matrix (accuracy 0.549)

Rows are the ground truth of the injection, columns are the rule-table verdict. 'undetermined' is a legitimate output: a stuck-at fault and a constant replay are indistinguishable in the data. Rows labelled 'clean_alarm' in the raw file (alarm with no injection) are excluded here because their truth is not a fault/attack label.

| truth        |   natural |   fault |   attack |   undetermined |   row_total |
|:-------------|----------:|--------:|---------:|---------------:|------------:|
| natural      |         4 |       0 |        0 |              0 |           4 |
| fault        |         0 |      63 |       18 |             27 |         108 |
| attack       |         0 |       0 |       45 |             63 |         108 |
| undetermined |         0 |       0 |        2 |             22 |          24 |
