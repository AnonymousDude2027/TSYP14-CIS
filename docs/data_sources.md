# Data sources, the **[Verify]** items, and what was actually checked

The specification marks several items **[Verify]** (section 18). This note records
each one, the documentation or probe that resolved it, and the coding consequence.
Everything here was checked against the live services in October 2026 while the
bundle was built; the bundle itself is committed so that a reader never needs the
network.

## 1. NCEI data-service parameter names — resolved

Endpoint actually used:

```
https://www.ncei.noaa.gov/access/services/data/v1
    ?dataset=daily-summaries
    &stations=<11-char GHCN id>          (comma-separated for several)
    &startDate=YYYY-MM-DD&endDate=YYYY-MM-DD
    &dataTypes=TMAX,TMIN,PRCP
    &format=csv&units=metric&includeAttributes=true
```

Verified behaviour (probed on CARTHAGE `TSM00060715`, 2015-2024):

* **Units**: `units=metric` is required for °C and mm; the default is US units.
* **QC flags**: with `includeAttributes=true` every variable gets a companion
  column `<VAR>_ATTRIBUTES` whose comma-separated fields are
  `source, qflag, ...`. We keep a value only when the `qflag` field is empty
  (GHCN convention: empty = passed all QC); flagged values are *not* silently
  repaired, they become missing days. This is the difference between "the archive
  says this value is doubtful" and "the sensor is lying", and the pipeline must not
  confuse the two.
* **Missing days are omitted from the CSV, not filled** — so the calendar has to
  be rebuilt explicitly (`io.load_pair` reindexes onto a complete daily calendar
  and sets a gap flag). Treating the CSV row order as the calendar is a classic and
  silent source of misalignment.
* **AWND is not populated** for the Mediterranean stations in this bundle, so the
  wind variable was dropped early (the specification allows a subset); only
  `TMAX`, `TMIN`, `PRCP` are used.
* Station identifiers are the 11-character GHCN ids (e.g. `TSM00060715`), obtained
  from `ghcnd-stations.txt` and cross-checked against `ghcnd-inventory.txt` for the
  first/last year of each element (that check is what selected the sites in
  `config/sites.yaml`, and it is re-runnable as `python -m tsfdi sites`).

## 2. NASA POWER missing-value marker and parameter names — resolved

Endpoint actually used:

```
https://power.larc.nasa.gov/api/temporal/daily/point
    ?parameters=T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PRECTOTCORR,ALLSKY_SFC_SW_DWN,CLRSKY_SFC_SW_DWN
    &community=AG&latitude=<lat>&longitude=<lon>
    &start=YYYYMMDD&end=YYYYMMDD&format=JSON
```

* **Missing marker is `-999.0`** (fill value); it is mapped to `NaN` at load time
  and *never* to zero — for precipitation that distinction is the whole game.
* Values are a nested dictionary `properties.parameter.<NAME>.<YYYYMMDD>`, with
  string keys, so the date index must be parsed rather than positional (another
  silent-misalignment trap: a POWER response is not a rectangular array).
* **Timestamps are local solar time (LST)**, not UTC. GHCN-Daily days are local
  calendar days, so both roads are treated as local days and the *lag* between
  them is estimated on the calibration period instead of being assumed to be zero.
  Measured lags are 0 or ±1 day for the bundled stations, consistent with LST
  versus local-standard-time reporting.
* `PRECTOTCORR` is a corrected precipitation flux; it is the closest daily
  analogue to a gauge total, and its correlation with a point gauge is much weaker
  than for temperature — the reason precipitation carries a restricted claim in
  the report.

## 3. Hourly station data — available, deliberately not used

An hourly source does exist and was checked: NCEI's **Integrated Surface Database
(ISD / "Global Hourly")** is being replaced by **GHCNh (Global Historical
Climatology Network hourly)**, with GHCNh using the same station identifiers as
GHCN-Daily ([NCEI announcement](https://www.ncei.noaa.gov/news/next-generation-climate-dataset-built-seamless-integration),
[service-change notice](https://www.nesdis.noaa.gov/news/global-historical-climate-network-hourly-integrated-surface-data-global-hourly)).
The transition is date-dependent in the NCEI data-access API: `global-hourly` for
dates before 2025-01-01 and `global-historical-climatology-network-hourly` from
2025-01-01 onwards, with **UTC timestamps** for hourly queries (unlike the daily
service).

Decision: **daily data is used for the evaluation**, and the architecture is
hourly-capable (the streaming pipeline is O(1)-state per reading and provenance
costs ~0.2 ms/record). The reasons are stated rather than implied: (i) the
challenge's attack magnitudes are defined in units of the seasonal range and the
sigma of the *daily* residual, so an hourly evaluation would have needed a second
calibration protocol; (ii) GHCNh coverage for the North-African stations in this
bundle is sparse, which would have silently changed the site set; (iii) the
reference product is daily, so an hourly evaluation would have required an
interpolation of the reference that would then dominate the residual. This is a
documented scope decision, not an oversight.

## 4. The 12 % / 15 % / 7-sample profile and how a percent is defined — resolved by documentation

The specification leaves the meaning of "12 %" open. The implementation uses, and
every table states:

* **temperature**: percent of the site's *seasonal range* (max − min of the fitted
  seasonal baseline). Percent of a Celsius value is meaningless — 12 % of 5 °C is
  0.6 K and 12 % of 30 °C is 3.6 K for the same station in the same winter — so the
  range is the only scale that is stable across sites and seasons.
* **precipitation and wind**: a multiplicative factor (12 % more rain), because
  the working coordinate is already a variance-stabilised transform of a
  non-negative quantity.

Every table also reports the **σ-equivalent** of each percent, computed from the
site's own residual scale, so a reader who prefers σ units never has to convert.

## 5. Jury VM specification — unknown, therefore assumed worst case

Unknown at the time of writing. Consequences that are implemented, not promised:

* no network at run time (bundle committed; `make all` never fetches);
* no GPU and no compiler (pure NumPy/SciPy/pandas wheels, pinned in
  `requirements.txt`);
* the dashboard is optional: `python -m tsfdi report-static` writes a
  self-contained HTML file with inline SVG and no JavaScript, and the CLI reports
  everything the dashboard shows;
* the model cache is deterministic for a given seed and manifest, so a re-run
  either reproduces `expected_results.json` within tolerance or fails loudly.

## 6. Reproducing the bundle

`data/bundle/FETCH_INFO.json` records the exact request strings, retrieval timestamps and row
counts per file. `python -m tsfdi fetch --force` rebuilds it. The archive providers
revise their holdings, so a fresh fetch may legitimately change a handful of
values; that is precisely why the bundle is pinned and signed rather than fetched
on demand.
