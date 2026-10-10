# Registered forecasting challenge: August-October 2025

The July-selected boosting forecast has **MAE 5.4020 versus 5.6848** for the July-selected weekday-mean comparator over **17,226 zero-inclusive zone-days**. The 4.97% lower pooled MAE is a limited forecasting result under an **assumed daily operator feed**. Boosting loses on 21/66 days; ridge has lower overall RMSE and lower MAE in the busiest zone group. No dispatch saving, real-time TLC availability or frontier model is established.

## Why this comparison

The earlier July study compared the original six-month mean with zero. That established later-period information, but zero alone is insufficient for a strong forecasting claim. [Simple seasonal forecasts](https://otexts.com/fpp3/simple-methods.html) and [strictly earlier-observation evaluation](https://otexts.com/fpp3/tscv.html) motivate the stronger comparison here. A weekday mean captures weekly seasonality; recent means and medians check adaptation and outlier robustness; ridge and absolute-error boosting test whether pooled historical features add value. A complex sequence/graph model would require a broader tuning budget and additional uninspected evidence, not automatic entitlement to a stronger claim.

Eight transparent baselines, three regularised regressions and three boosting configurations were registered. **July is inspected development/selection data**, not a new holdout. August, September and October were fixed as one contiguous later evaluation before locally downloading any of their files. These months add 12,253,805 raw / 11,590,039 cleaned trips, 66 weekdays and 3,144 zero-activity zone-days. All holidays remain included. Every date and model result is retained.

## Two public freezes, then collection

1. [Protocol/code registration](https://github.com/Prasanna-K-123/urban-mobility-pyspark-optimization/commit/43216c7de47c66369fc9fad3e12cf50b202fad33) preceded development-source collection and all new model fitting. Protocol SHA256 `b4c4fc9e9130c487eee772f04b15f09404f275f603a2f2384d5adb75ca94e24a`.
2. [July selection lock](https://github.com/Prasanna-K-123/urban-mobility-pyspark-optimization/commit/01c5421152194e620f1c6c0d6196f92e288cc75f) preceded local August-October downloads and outcome reconstruction. Selection-lock SHA256 `93ad914e67a86dab21cef4973c98082632aa93167eeebe9699efd311baa45a37`.

The official website's three evaluation links were resolved before registration; the web reader rejected their binary content and supplied no records/counts/scores. That unsuccessful link-resolution attempt is disclosed in the protocol. This is retrospective research performed in October 2026, not a claim of a prospectively conducted 2025 experiment or an external preregistration service.

January is warm-up; February 3-June 30 supplies **27,666 identical supervised rows** to every candidate. July supplies 6,003 selection rows. Each model family's best July MAE is selected, then the best of those two is fixed as the primary challenger. The nonzero comparator is independently selected from seven simple baselines using July MAE. Exact ties use names. Chosen ridge alpha=100 and boosting max_leaf_nodes=15 are each refit once on **33,669 February-July rows**; parameters remain fixed throughout evaluation. No other four candidate configurations are scored on August-October. No choice, feature, loss, window, partition or metric was changed after those outcomes were inspected.

All six candidates receive the same earlier-day features, training rows and three-candidate-per-family budget. Ridge uses training-only scaling and an SVD solve. Boosting uses absolute-error loss, 120 iterations, learning rate 0.08, minimum leaf size 40, L2=1 and no random-validation early stopping. [Full protocol](../reference/forecast_challenge/PROTOCOL.json) and [all July trials](../reference/forecast_challenge/selection_trials.csv).

## Data and availability contract

Official [TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) are provider-supplied and their accuracy is not guaranteed. The [dictionary](https://www.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf) describes meter timestamps and TLC zones; it does not observe idle taxis or unmet requests. We preserve the original 261-zone universe and zero-fill every calendar weekday. January-June raw reconstruction exactly recovers the original 24,083,384 / 22,974,942 counts and zone means; July reconstruction exactly recovers its preserved panel.

Cleaning retains valid IDs 1-263, 0<distance<100, positive elapsed duration and floored-second durations of 1-240 minutes. January-June files use the original joint pickup-period filter; later files use their own pickup month. The four-hour duration bound rules out a prior-month pickup contributing a next-month 18:00 dropoff. Local wall-clock timestamp interpretation matches the original study. No evaluated 18:00 activity is outside the frozen zone universe.

At a target date's midnight origin, the complete previous weekday's 18:00 panel is **assumed available from an operator feed**. Feature construction uses only dates strictly before the target. Lagged flows, past means/medians/std, weekday and known day-of-year features enter the models. The numeric zone ID, same-day outcomes, fares and future weather/event realizations do not enter X. Training aggregates are constructed from earlier dates for each training row, rather than using future-in-training target means as features.

**TLC public files are published monthly, typically with a two-month delay. They cannot supply this daily feed.** Original-vintage reporting/arrival times are not available. The simulation therefore does not establish a forecast deployable from public TLC releases as they appeared in 2025. That separate availability-constrained evaluation and a real operator feed remain unexecuted. Earlier evaluation observations may update rolling history only after their date; they never refit model parameters or select configurations.

Raw file hashes, bytes and collection times appear in [development receipts](../reference/forecast_challenge/development_sources.json) and [evaluation receipts](../reference/forecast_challenge/evaluation_sources.json). Raw trip records are fetched from TLC; compact lossless aggregate panels are committed.

## Every final comparator

MAE/RMSE/bias are in signed net-trip-flow units per zone at 18:00. Bias is prediction minus observation. The primary comparison was selected on July, before the final scores.

| Forecast | MAE | RMSE | Signed bias |
|---|---:|---:|---:|
| Zero sanity comparator | 12.8452 | 35.8168 | -1.2932 |
| Original frozen Jan-Jun zone mean | 6.1339 | 16.1578 | -0.0220 |
| **Frozen Jan-Jun weekday mean: selected primary baseline** | **5.6848** | 14.6114 | -0.0312 |
| Previous same weekday (five weekday observations back) | 7.1357 | 17.5923 | -0.0046 |
| Previous 20-weekday mean | 5.8806 | 15.0792 | 0.0006 |
| Previous 20-weekday median | 5.9141 | 15.3042 | -0.1683 |
| Previous four same-weekday mean | 5.8248 | 14.1826 | -0.0222 |
| Previous four same-weekday median | 5.8060 | 14.1379 | -0.1162 |
| Selected ridge, alpha=100 | 5.4743 | **13.1142** | -0.7574 |
| **Selected boosting, 15 leaves: primary challenger** | **5.4020** | 13.5252 | -0.2961 |

Primary boosting-minus-baseline MAE **-0.282788**. Two thousand paired ISO-week-cluster draws give interval **[-0.484646, -0.132541]** across 14 blocks. A separate paired zone-cluster sensitivity gives **[-0.543194, -0.092005]** across 261 zones. Whole weeks retain all zones/dates; whole zones retain all dates. These are distinct conditional fixed-fit assumptions, **not joint spatial/calendar dependence, model-refit or July-selection uncertainty**, and not independent-year evidence.

| Month | Weekdays | Selected weekday mean MAE | Ridge MAE | Boosting MAE |
|---|---:|---:|---:|---:|
| August | 21 | 5.5431 | **4.9315** | 4.9525 |
| September | 22 | 6.0126 | 5.9364 | **5.8623** |
| October | 23 | 5.5006 | 5.5279 | **5.3722** |

## Adverse findings and practical limits

- Boosting loses to the selected simple baseline on **21 of 66 days**, winning on 45. Its worst relative day, September 8, has MAE 5.6354 versus 4.5921. No loss day or holiday is dropped. [Every daily score](../reference/forecast_challenge/daily_metrics.csv).
- Ridge has lower pooled RMSE (13.1142 versus 13.5252), lower August MAE and lower high-activity-zone MAE. The fixed top ceil(20%) group contains 53 zones chosen solely from Jan-Jun pickup+dropoff activity: ridge MAE 19.0375, boosting 19.5134, weekday baseline 20.6478. In the remaining 208 zones, boosting MAE is 1.8063 versus ridge 2.0183 and baseline 1.8721. Pooled superiority in one metric does not establish universal dominance.
- Both regressions have more negative pooled bias than the selected simple baseline. Median/mean smoothing, holiday/calendar effects, availability and fitting uncertainty remain substantive further questions. No claim that more complex models have been exhausted.
- A lower net-flow forecast error does not demonstrate an improved dispatch policy. Trip flow does not identify idle-fleet supply, observed travel/driver constraints or realised costs. The original 3.76% lower **static proxy vehicle-miles** remains a separate optimization result, unchanged.
- One city/type/year and three contiguous months limit generalisation. Full-panel forecasts and every model metric are visible; a modest result is retained without inflated operational claims.

## Inspect and reproduce

[Frozen selection lock](../reference/forecast_challenge/selection_lock.json) · [Summary and strata](../reference/forecast_challenge/summary.json) · [Every final prediction](../reference/forecast_challenge/evaluation_predictions.csv.gz) · [Development panel](../reference/forecast_challenge/development_panel.csv.gz) · [Final panel](../reference/forecast_challenge/evaluation_panel.csv.gz) · [Release hashes](../reference/forecast_challenge/artifact_manifest.json) · [Raw/refit verifier](../verify_forecast_challenge.py) · [Workflow](../.github/workflows/forecast-challenge.yml)

```bash
python -m pip install -r requirements-forecast.txt
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest -q
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python verify_forecast_challenge.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python verify_forecast_challenge.py --raw
```

The first verifier hashes the release, recomputes causal features, redoes all July candidate selection, refits both chosen models and rechecks all final predictions, scores, strata and paired intervals. `--raw` also fetches all ten pinned original Parquets and reconstructs every development/evaluation count. These checks passed locally; maximum prediction error against committed CSV precision is **5.12e-13**. All **20 tests** pass, including hand-constructed cleaning/count oracles and future-outcome perturbation tests. Recorded environment: Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, sklearn 1.8.0, DuckDB 1.4.1.

Two older workflows now install the shared test dependencies so adding the new independent tests does not break their discovery. This is workflow packaging, not a change to frozen forecasting methods. Source-freeze workflow runs precede completed data and must not be cited as full reconstruction evidence. Successful final exact-head runs are linked in the profile evidence guide after their actual completion. Preparation used AI assistance; no external adoption, endorsement or review is claimed.
