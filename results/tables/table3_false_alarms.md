# Table 3 - false alarms on clean data, including natural extremes

events_per_year is measured on the 2020-2024 test period with thresholds calibrated on 2015-2019 only. natural_days counts days inside reference-labelled extremes (TMAX above its 95th percentile for >= 3 days, TMIN below its 5th, PRCP above its 99th); events_in_natural counts alarm events inside those windows. The gap between the calibration-period rate and the test-period rate is the *transfer* cost of calibrating on one period and operating on another; it is reported, not hidden.

| site                | variable   |   years |   events |   events_per_year |   events_in_natural |   natural_days |   events_calibration |   fa_target |   excess_over_budget |
|:--------------------|:-----------|--------:|---------:|------------------:|--------------------:|---------------:|---------------------:|------------:|---------------------:|
| CARTHAGE            | TMAX       |  5.0021 |        8 |            1.5993 |                   0 |             80 |                    2 |           1 |               0.5993 |
| CARTHAGE            | TMIN       |  5.0021 |        3 |            0.5998 |                   0 |             27 |                    1 |           1 |              -0.4002 |
| CARTHAGE            | PRCP       |  5.0021 |        4 |            0.7997 |                   0 |             16 |                    4 |           1 |              -0.2003 |
| JENDOUBA            | TMAX       |  5.0021 |       12 |            2.399  |                   1 |             92 |                    0 |           1 |               1.399  |
| JENDOUBA            | TMIN       |  5.0021 |       18 |            3.5985 |                   0 |             34 |                    2 |           1 |               2.5985 |
| JENDOUBA            | PRCP       |  5.0021 |       16 |            3.1987 |                   0 |             14 |                    4 |           1 |               2.1987 |
| GAFSA               | TMAX       |  5.0021 |       16 |            3.1987 |                   3 |            100 |                    3 |           1 |               2.1987 |
| GAFSA               | TMIN       |  5.0021 |        2 |            0.3998 |                   0 |             50 |                    2 |           1 |              -0.6002 |
| GAFSA               | PRCP       |  5.0021 |       31 |            6.1975 |                   0 |             16 |                    3 |           1 |               5.1975 |
| MERSA_MATRUH        | TMAX       |  5.0021 |       19 |            3.7984 |                   1 |             92 |                    5 |           1 |               2.7984 |
| MERSA_MATRUH        | TMIN       |  5.0021 |        7 |            1.3994 |                   0 |             45 |                    1 |           1 |               0.3994 |
| KHARGA              | TMAX       |  5.0021 |       31 |            6.1975 |                   1 |             46 |                    3 |           1 |               5.1975 |
| KHARGA              | TMIN       |  5.0021 |       27 |            5.3978 |                   0 |             63 |                    1 |           1 |               4.3978 |
| NAVACERRADA         | TMAX       |  5.0021 |        7 |            1.3994 |                   0 |             73 |                    1 |           1 |               0.3994 |
| NAVACERRADA         | TMIN       |  5.0021 |        5 |            0.9996 |                   0 |             46 |                    1 |           1 |              -0.0004 |
| NAVACERRADA         | PRCP       |  5.0021 |        7 |            1.3994 |                   1 |             22 |                    1 |           1 |               0.3994 |
| MONT_AIGOUAL        | TMAX       |  5.0021 |        5 |            0.9996 |                   0 |             88 |                    5 |           1 |              -0.0004 |
| MONT_AIGOUAL        | TMIN       |  5.0021 |       24 |            4.798  |                   0 |             42 |                    6 |           1 |               3.798  |
| MONT_AIGOUAL        | PRCP       |  5.0021 |        8 |            1.5993 |                   1 |             10 |                    0 |           1 |               0.5993 |
| FELDBERG            | TMAX       |  5.0021 |        4 |            0.7997 |                   0 |             55 |                    2 |           1 |              -0.2003 |
| FELDBERG            | TMIN       |  5.0021 |       10 |            1.9992 |                   0 |             52 |                    2 |           1 |               0.9992 |
| FELDBERG            | PRCP       |  5.0021 |        6 |            1.1995 |                   0 |             15 |                    0 |           1 |               0.1995 |
| CABAUW              | TMAX       |  5.0021 |        4 |            0.7997 |                   0 |             60 |                    0 |           1 |              -0.2003 |
| CABAUW              | TMIN       |  5.0021 |       14 |            2.7989 |                   2 |             46 |                    2 |           1 |               1.7989 |
| CABAUW              | PRCP       |  5.0021 |        8 |            1.5993 |                   0 |             18 |                    1 |           1 |               0.5993 |
| HELSINKI_KAISANIEMI | TMAX       |  5.0021 |       14 |            2.7989 |                   0 |             84 |                    3 |           1 |               1.7989 |
| HELSINKI_KAISANIEMI | TMIN       |  5.0021 |        8 |            1.5993 |                   1 |             70 |                    3 |           1 |               0.5993 |
| HELSINKI_KAISANIEMI | PRCP       |  5.0021 |        4 |            0.7997 |                   0 |             21 |                    0 |           1 |              -0.2003 |
| ALTA_LUFTHAVN       | TMAX       |  5.0021 |        3 |            0.5998 |                   0 |             77 |                    0 |           1 |              -0.4002 |
| ALTA_LUFTHAVN       | TMIN       |  5.0021 |        7 |            1.3994 |                   0 |             58 |                    2 |           1 |               0.3994 |
| ALTA_LUFTHAVN       | PRCP       |  5.0021 |        9 |            1.7993 |                   0 |             15 |                    1 |           1 |               0.7993 |
| MIAMI_INTL          | TMAX       |  5.0021 |       19 |            3.7984 |                   1 |             55 |                    7 |           1 |               2.7984 |
| MIAMI_INTL          | TMIN       |  5.0021 |       13 |            2.5989 |                   0 |             48 |                    4 |           1 |               1.5989 |
| MIAMI_INTL          | PRCP       |  5.0021 |       14 |            2.7989 |                   1 |             21 |                    1 |           1 |               1.7989 |
