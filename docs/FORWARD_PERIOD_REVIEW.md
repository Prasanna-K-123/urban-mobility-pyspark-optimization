# Frozen later-month flow forecast: July 2025

The original January–June 2025 PySpark study produced a **129-weekday, 261-zone zero-inclusive panel** and its mean zone net flow. That aggregate was frozen before this follow-on evaluation. The [protocol](../reference/forward_panel/PROTOCOL.json) chose the immediately following month before downloading/scoring July. It is a retrospective study executed in October 2026, not a claim that this experiment was prospectively run in 2025.

## Question and sources

How well does the unchanged six-month mean predict next-month 6 PM net trip flow? The comparator predicts **zero** in every zone. This is a transparent sanity baseline, **not the strongest seasonal or rolling forecasting competitor**. No July fit, threshold, feature or parameter search was performed. The study checks whether historical means carry any later-period information; it does not claim a frontier demand model.

Official [NYC TLC trip records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page); [July 2025 Yellow Taxi Parquet](https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-07.parquet). TLC documents provider-supplied data and does not guarantee accuracy. Raw SHA256 `1efa6a0458dc60341f7f81dcbb5015af731f42d1a109f4e0d594dd46970fd4ef`; training aggregate SHA256 `a3301e203e89c65951dbfd2e743a6c051517ae0a9d6adf85977d9cfd3856efeb`. The raw Parquet is fetched from the official source rather than redistributed.

## Identical cleaning, complete held-out panel

July pickup timestamps; valid TLC IDs 1–263; 0<distance<100; dropoff after pickup; floored elapsed seconds/60 between 1 and 240. 6 PM uses local New York wall-clock timestamps in the source, with no UTC conversion. The frozen 261-zone universe is retained. All 23 calendar weekdays are scored, including the July 4 holiday, with no holiday exclusion chosen after seeing performance.

**3,898,963 raw / 3,676,278 cleaned** trips; **23 weekdays × 261 zones = 6,003** zone-days, including **1,197 zero-activity rows**. No 6 PM pickups/dropoffs fell outside the frozen zone universe. Out-of-universe activity is explicitly checked and reported rather than silently fitted into new zones. The follow-on aggregator uses DuckDB; the original 24.08 million-trip pipeline remains PySpark.

## Results

| Metric | Frozen Jan–Jun mean | Always-zero comparator |
|---|---:|---:|
| Mean absolute net-flow error |5.787926|12.403298|
| Root mean squared error |15.180173|34.981371|
| Signed prediction-minus-observation bias |0.172522|-1.098617|

Mean-minus-zero MAE **-6.615372**; 2,000 paired ISO-week-block resamples, interval **[-7.746767,-4.964824]**. Whole weeks retain all zones/days to respect spatial/within-week dependence. There are only **five calendar-week blocks**, so this is limited one-month uncertainty conditional on the frozen training means. It is not evidence across independent years or a proof of drift robustness. Mean forecasts beat zero on 22/23 days and lose on one; every [daily result](../reference/forward_panel/daily_metrics.csv) is retained.

## What this does not establish

Trip net flow does not observe idle-fleet supply or unmet demand. This does not execute the transport allocation on live taxis, establish dynamic routing, demonstrate causal savings, or convert the original 3.76% lower **proxy distance** into an out-of-period saving. The strongest remaining forecasting challenge is an independently frozen seasonal/rolling benchmark and multiple periods; it is not reported as completed. The existing transport optimum/greedy comparison remains a separate static proxy study.

## Inspect and reproduce

- [Every zero-inclusive zone-day, prediction and error](../reference/forward_panel/panel.csv)
- [Summary and hashes](../reference/forward_panel/summary.json)
- [Raw source identity](../reference/forward_panel/source.json)
- [Executable follow-on](../benchmark_forward.py) and [workflow](../.github/workflows/forward.yml)

```bash
python -m pip install -r requirements-forward.txt
python -m pytest -q
python benchmark_forward.py --offline
python benchmark_forward.py --download --verify
```

Offline replay re-scores committed panel bytes. Raw reconstruction actually fetches the pinned source and rebuilds every held-out row; the two checks are distinct. Both passed locally, with all seven project tests passing, including a hand-constructed raw-record oracle for month, distance, microsecond duration and hour boundaries. Remote CI state is reported only after its actual result. Environment: Python 3.12.14, DuckDB 1.4.1, NumPy 2.3.5, pandas 2.2.3. Preparation used AI assistance; no external adoption/review is claimed.
