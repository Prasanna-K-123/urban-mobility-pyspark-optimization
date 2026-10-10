# Urban Mobility Operations & Fleet Rebalancing Optimization with PySpark

## Registered seasonal/rolling challenge — October 10, 2026

On **17,226 August-October 2025 zone-days**, July-selected boosting has MAE **5.4020 versus 5.6848** for the locked weekday-mean comparator. Eight simple baselines and equal-budget ridge/boosting candidates were registered before the new period was collected. Every later prediction/day is retained: boosting loses on 21/66 days; ridge has lower RMSE and lower error in the busiest zone group. Rolling inputs require an **assumed daily operator feed**; monthly TLC publications cannot supply it. [Full comparison, public freezes, uncertainty and raw/refit reproduction](docs/FORECAST_CHALLENGE_REVIEW.md).

The 3.76% static proxy-distance result below is separate from this forecasting study; no realised dispatch saving is claimed. The July release remains an unchanged earlier result and is development context for the new comparison.

## Frozen later-period check — October 10, 2026

Unchanged Jan–Jun zone-flow means were evaluated on **6,003 July zone-days** from 3.90 million additional raw trips. MAE **5.7879** versus **12.4033** for a zero-flow sanity comparator; 22/23 days improve, with a paired week-block uncertainty check. This is a later-period flow forecast, **not realized dispatch savings**. [Full protocol, every prediction, baseline limits and raw reconstruction](docs/FORWARD_PERIOD_REVIEW.md).

## Evidence release — 2026-10-09

[100-case transport verification](results/transport_verification.json) · [Dependency-light solver](src/transport_core.py) · [Full reproduction workflow](.github/workflows/reproduce-tlc.yml)

Forty small networks are checked against exhaustive enumeration, with another sixty checked for capacity/conservation and greedy dominance. These are correctness fixtures, not additional TLC performance measurements. The full public-data pipeline was independently rerun in a pinned Spark/Python environment on 2026-10-09: [successful reproduction](https://github.com/Prasanna-K-123/urban-mobility-pyspark-optimization/actions/runs/37963818458).

The [261-zone aggregate inputs](results/zone_imbalance_weekday_6pm.csv), [optimized flow plan](results/optimized_rebalancing_plan.csv), and [regenerated metrics](results/project5_metrics.csv) are committed so the final allocation can be inspected directly.

[Research scope and next extension](docs/EVIDENCE_REVIEW.md)

---

Independent portfolio project using public **NYC Taxi & Limousine Commission (TLC)** Yellow Taxi trip records.

## Objective

Estimate recurring weekday 6 PM taxi-zone vehicle-flow imbalances from large-scale trip data, then solve a constrained fleet-rebalancing problem that covers the modeled deficit with minimum proxy deadhead distance.

## Final results

- **24,083,384** raw trips processed
- **22,974,942** cleaned trips retained (**95.40%**)
- Complete **129 weekdays × 261 active zones = 33,669** row 6 PM panel
- Modeled surplus: **1,713 vehicles across 184 zones**
- Modeled deficit: **1,382 vehicles across 24 zones**
- Integer optimizer covered **1,382 / 1,382 modeled deficit vehicles (100%)**
- Distance-aware greedy heuristic: **3,382.38 proxy vehicle-miles**
- Global integer optimum: **3,255.24 proxy vehicle-miles**
- **3.76% reduction** in proxy deadhead distance versus the distance-aware greedy heuristic

## Why PySpark

The project processes more than 24 million trip records using Spark DataFrames for filtering, temporal feature engineering, pickup/drop-off aggregation, and construction of the zone-hour operations table.

## Methodology

### 1. Data quality and cleaning

The pipeline removes null timestamps, restricts the analysis to Jan–Jun 2025 after the audit exposed a small number of out-of-period timestamps, keeps valid TLC location IDs, removes non-positive/extreme trip distances, requires drop-off after pickup, calculates duration with Spark timestamp functions, and retains trips between 1 and 240 minutes.

### 2. Complete weekday 6 PM panel

A naive average over only observed zone-hours can overstate activity in sparse zones because zero-activity periods disappear. The final method therefore creates the full **129 weekdays × 261 active zones** panel, left-joins observed 6 PM activity, fills missing observations with zero, and estimates:

`net vehicle flow = drop-offs - pickups`

Positive flow is treated as a modeled **surplus proxy** and negative flow as a modeled **deficit proxy**. TLC trip records do not directly observe idle taxis, so this is explicitly a proxy.

### 3. Geospatial mapping

Official TLC taxi-zone polygons are projected into **EPSG:2263** and represented by zone centroids. Straight-line centroid distance is used as a **deadhead-distance proxy**, not road-network distance.

### 4. Integer optimization

Decision variables represent integer vehicle movements from surplus zones to deficit zones. Constraints prevent sending more vehicles than modeled surplus or receiving more than modeled deficit. The model first forces maximum internally coverable deficit and then minimizes total proxy vehicle-miles. CBC returned an **Optimal** solution.

## Benchmark

The optimized allocation is compared with a **distance-aware greedy heuristic**, not an intentionally weak baseline. The heuristic repeatedly chooses the closest currently feasible surplus–deficit pair and moves as many vehicles as possible.

| Metric | Distance-aware greedy | Global integer optimum |
|---|---:|---:|
| Vehicles moved | 1,382 | 1,382 |
| Modeled deficit coverage | 100% | 100% |
| Proxy vehicle-miles | 3,382.38 | **3,255.24** |
| Avg proxy miles / vehicle | 2.447 | **2.355** |

**Optimization reduced proxy deadhead distance by 3.76%** while preserving full modeled deficit coverage.

## Tech stack

- Python
- PySpark / Spark DataFrames
- Pandas / NumPy
- PuLP / CBC integer optimization
- GeoPandas
- Matplotlib
- NYC TLC Parquet + official taxi-zone geospatial data

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── src/
│   └── analysis_pipeline.py
└── results/
    └── project5_metrics.csv
```

The original executed Colab notebook is retained separately as part of the project working files; this repository exposes the cleaned reproducible pipeline and frozen metrics.


## Reproduce

1. Install Java, Python, and the packages in `requirements.txt`.
2. Run `python src/analysis_pipeline.py` from the repository root.
3. The script downloads the public Jan-Jun 2025 TLC inputs, rebuilds the complete 6 PM panel, solves the optimization model, and writes regenerated outputs to `results/`.
4. Confirm that the optimized proxy vehicle-miles do not exceed the distance-aware greedy benchmark and that 1,382 modeled deficit vehicles are covered on the published data snapshot.

## Limitations

- Pickup/drop-off flow is a proxy for vehicle availability; TLC records do not directly observe idle taxis.
- Zone-centroid straight-line distance is a proxy, not road-network travel distance.
- The model uses recurring weekday 6 PM averages rather than real-time demand.
- It does not model traffic, driver behavior, labor constraints, operating cost, or road-network feasibility.
- Integer supply/deficit quantities are rounded from historical average flows.
- This is a decision-support prototype, not a production dispatch system.

## What this project demonstrates

- large-scale data processing with PySpark;
- careful data-quality auditing;
- complete-panel construction to avoid zero-activity omission bias;
- operations analytics;
- geospatial feature engineering;
- integer optimization;
- strong-baseline benchmarking;
- quantitative error analysis;
- translation of analytical outputs into an operational decision problem.
