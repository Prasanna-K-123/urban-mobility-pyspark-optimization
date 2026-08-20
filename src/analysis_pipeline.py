"""Urban mobility operations and fleet-rebalancing optimization.

Reproducible pipeline for NYC TLC Yellow Taxi records, Jan-Jun 2025.
CPU runtime is sufficient. Requires Java for PySpark.
"""

from pathlib import Path
from math import sqrt
import zipfile
import shutil

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
import pulp
from pyspark.sql import SparkSession, functions as F


DATA_DIR = Path("tlc_project")
OUTPUT_DIR = Path("project5_outputs")
MONTHS = [f"{m:02d}" for m in range(1, 7)]

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-{month}.parquet"
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_SHAPE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def get_spark() -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("NYC_Taxi_Fleet_Rebalancing")
        .config("spark.sql.shuffle.partitions", "48")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_and_clean(spark: SparkSession):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for month in MONTHS:
        download(TRIP_URL.format(month=month), DATA_DIR / f"yellow_tripdata_2025-{month}.parquet")
    download(ZONE_LOOKUP_URL, DATA_DIR / "taxi_zone_lookup.csv")
    download(ZONE_SHAPE_URL, DATA_DIR / "taxi_zones.zip")

    paths = [str(DATA_DIR / f"yellow_tripdata_2025-{m}.parquet") for m in MONTHS]
    raw = spark.read.parquet(*paths)

    trips = raw.select(
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "PULocationID",
        "DOLocationID",
        "trip_distance",
        "total_amount",
    )

    raw_rows = trips.count()

    clean = (
        trips
        .filter(F.col("tpep_pickup_datetime").isNotNull())
        .filter(F.col("tpep_dropoff_datetime").isNotNull())
        .filter((F.year("tpep_pickup_datetime") == 2025) & F.month("tpep_pickup_datetime").between(1, 6))
        .filter(F.col("PULocationID").between(1, 263))
        .filter(F.col("DOLocationID").between(1, 263))
        .filter(F.col("trip_distance") > 0)
        .filter(F.col("trip_distance") < 100)
        .filter(F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime"))
        .withColumn(
            "trip_minutes",
            F.expr("timestampdiff(SECOND, tpep_pickup_datetime, tpep_dropoff_datetime)") / 60.0,
        )
        .filter(F.col("trip_minutes").between(1, 240))
    )

    clean_rows = clean.count()
    print(f"Raw rows: {raw_rows:,}")
    print(f"Clean rows: {clean_rows:,}")
    print(f"Retained: {clean_rows / raw_rows:.2%}")
    return clean, raw_rows, clean_rows


def build_complete_six_pm_panel(clean):
    pickups = (
        clean
        .withColumn("service_date", F.to_date("tpep_pickup_datetime"))
        .withColumn("hour", F.hour("tpep_pickup_datetime"))
        .groupBy("service_date", "hour", F.col("PULocationID").alias("LocationID"))
        .agg(
            F.count("*").alias("pickups"),
            F.sum("total_amount").alias("pickup_revenue"),
        )
    )

    dropoffs = (
        clean
        .withColumn("service_date", F.to_date("tpep_dropoff_datetime"))
        .withColumn("hour", F.hour("tpep_dropoff_datetime"))
        .groupBy("service_date", "hour", F.col("DOLocationID").alias("LocationID"))
        .agg(F.count("*").alias("dropoffs"))
    )

    zone_hour = (
        pickups
        .join(dropoffs, on=["service_date", "hour", "LocationID"], how="full")
        .fillna({"pickups": 0, "dropoffs": 0, "pickup_revenue": 0.0})
        .withColumn("net_vehicle_flow", F.col("dropoffs") - F.col("pickups"))
        .withColumn("is_weekday", F.dayofweek("service_date").between(2, 6))
    )

    weekday_dates = (
        clean
        .select(F.to_date("tpep_pickup_datetime").alias("service_date"))
        .distinct()
        .filter(F.dayofweek("service_date").between(2, 6))
    )

    active_zones = (
        clean.select(F.col("PULocationID").alias("LocationID"))
        .union(clean.select(F.col("DOLocationID").alias("LocationID")))
        .distinct()
    )

    grid = weekday_dates.crossJoin(active_zones).withColumn("hour", F.lit(18))

    observed = (
        zone_hour
        .filter((F.col("is_weekday") == True) & (F.col("hour") == 18))
        .select("service_date", "hour", "LocationID", "pickups", "dropoffs", "pickup_revenue", "net_vehicle_flow")
    )

    panel = (
        grid
        .join(observed, on=["service_date", "hour", "LocationID"], how="left")
        .fillna({"pickups": 0, "dropoffs": 0, "pickup_revenue": 0.0, "net_vehicle_flow": 0})
    )

    imbalance = (
        panel
        .groupBy("LocationID")
        .agg(
            F.avg("pickups").alias("avg_pickups"),
            F.avg("dropoffs").alias("avg_dropoffs"),
            F.avg("net_vehicle_flow").alias("avg_net_vehicle_flow"),
            F.avg("pickup_revenue").alias("avg_pickup_revenue"),
            F.count("*").alias("weekday_6pm_hours"),
        )
        .orderBy(F.desc(F.abs(F.col("avg_net_vehicle_flow"))))
        .toPandas()
    )

    return imbalance


def load_zone_geometry() -> pd.DataFrame:
    zone_zip = DATA_DIR / "taxi_zones.zip"
    extract_dir = DATA_DIR / "taxi_zones_extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zone_zip, "r") as z:
        z.extractall(extract_dir)

    shp_files = list(extract_dir.rglob("*.shp"))
    if not shp_files:
        raise RuntimeError("No TLC taxi-zone shapefile found after extraction")

    zones = gpd.read_file(shp_files[0])
    if "LocationID" not in zones.columns:
        if "OBJECTID" not in zones.columns:
            raise RuntimeError(f"Could not identify taxi-zone ID column: {zones.columns.tolist()}")
        zones = zones.rename(columns={"OBJECTID": "LocationID"})

    zones["LocationID"] = zones["LocationID"].astype(int)
    zones = zones.to_crs(epsg=2263)
    zones["centroid"] = zones.geometry.centroid
    zones["x_ft"] = zones["centroid"].x
    zones["y_ft"] = zones["centroid"].y
    return zones[["LocationID", "borough", "zone", "x_ft", "y_ft"]].copy()


def prepare_optimization_inputs(imbalance_pd: pd.DataFrame, zone_meta: pd.DataFrame) -> pd.DataFrame:
    imbalance = imbalance_pd.merge(zone_meta, on="LocationID", how="left")
    if imbalance["x_ft"].isna().any():
        raise RuntimeError("At least one active zone lacks TLC geometry")

    imbalance["expected_surplus"] = np.maximum(np.rint(imbalance["avg_net_vehicle_flow"]), 0).astype(int)
    imbalance["expected_deficit"] = np.maximum(np.rint(-imbalance["avg_net_vehicle_flow"]), 0).astype(int)
    return imbalance


def solve_rebalancing(imbalance: pd.DataFrame):
    surplus = imbalance.loc[
        imbalance["expected_surplus"] > 0,
        ["LocationID", "borough", "zone", "expected_surplus", "x_ft", "y_ft"],
    ].copy()
    deficit = imbalance.loc[
        imbalance["expected_deficit"] > 0,
        ["LocationID", "borough", "zone", "expected_deficit", "x_ft", "y_ft"],
    ].copy()

    total_surplus = int(surplus["expected_surplus"].sum())
    total_deficit = int(deficit["expected_deficit"].sum())
    max_moves = min(total_surplus, total_deficit)

    distance = {}
    for _, s in surplus.iterrows():
        for _, d in deficit.iterrows():
            miles = sqrt((s["x_ft"] - d["x_ft"]) ** 2 + (s["y_ft"] - d["y_ft"]) ** 2) / 5280.0
            distance[(int(s["LocationID"]), int(d["LocationID"]))] = float(miles)

    model = pulp.LpProblem("NYC_Taxi_Internal_Rebalancing", pulp.LpMinimize)
    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat="Integer")
        for i in surplus["LocationID"].astype(int)
        for j in deficit["LocationID"].astype(int)
    }

    model += pulp.lpSum(distance[(i, j)] * x[(i, j)] for i, j in x)

    for _, row in surplus.iterrows():
        i = int(row["LocationID"])
        model += pulp.lpSum(x[(i, j)] for j in deficit["LocationID"].astype(int)) <= int(row["expected_surplus"])

    for _, row in deficit.iterrows():
        j = int(row["LocationID"])
        model += pulp.lpSum(x[(i, j)] for i in surplus["LocationID"].astype(int)) <= int(row["expected_deficit"])

    model += pulp.lpSum(x.values()) == max_moves
    status = model.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Solver status: {pulp.LpStatus[status]}")

    supply_meta = surplus.set_index("LocationID")
    demand_meta = deficit.set_index("LocationID")
    rows = []
    for (i, j), var in x.items():
        moved = int(round(var.value() or 0))
        if moved <= 0:
            continue
        rows.append({
            "from_LocationID": i,
            "from_zone": supply_meta.loc[i, "zone"],
            "to_LocationID": j,
            "to_zone": demand_meta.loc[j, "zone"],
            "vehicles": moved,
            "distance_miles_proxy": distance[(i, j)],
            "vehicle_miles_proxy": moved * distance[(i, j)],
        })

    plan = pd.DataFrame(rows)
    return plan, surplus, deficit, distance, total_surplus, total_deficit, max_moves


def greedy_nearest_pair(surplus, deficit, distance, max_moves):
    supply_left = {int(r.LocationID): int(r.expected_surplus) for r in surplus.itertuples()}
    demand_left = {int(r.LocationID): int(r.expected_deficit) for r in deficit.itertuples()}
    rows = []
    moved = 0

    while moved < max_moves:
        feasible = [
            (distance[(i, j)], i, j)
            for i, s in supply_left.items() if s > 0
            for j, d in demand_left.items() if d > 0
        ]
        if not feasible:
            break
        dist, i, j = min(feasible, key=lambda t: (t[0], t[1], t[2]))
        qty = min(supply_left[i], demand_left[j], max_moves - moved)
        rows.append({"from_LocationID": i, "to_LocationID": j, "vehicles": qty, "vehicle_miles_proxy": qty * dist})
        supply_left[i] -= qty
        demand_left[j] -= qty
        moved += qty

    return pd.DataFrame(rows)


def main():
    spark = get_spark()
    clean, raw_rows, clean_rows = load_and_clean(spark)
    imbalance_pd = build_complete_six_pm_panel(clean)
    zone_meta = load_zone_geometry()
    imbalance = prepare_optimization_inputs(imbalance_pd, zone_meta)

    plan, surplus, deficit, distance, total_surplus, total_deficit, max_moves = solve_rebalancing(imbalance)
    greedy = greedy_nearest_pair(surplus, deficit, distance, max_moves)

    optimal_miles = float(plan["vehicle_miles_proxy"].sum())
    greedy_miles = float(greedy["vehicle_miles_proxy"].sum())
    improvement = 1 - optimal_miles / greedy_miles

    assert int(plan["vehicles"].sum()) == max_moves
    assert int(greedy["vehicles"].sum()) == max_moves
    assert optimal_miles <= greedy_miles + 1e-6

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUTPUT_DIR / "optimized_rebalancing_plan.csv", index=False)
    imbalance.to_csv(OUTPUT_DIR / "zone_imbalance_weekday_6pm.csv", index=False)

    metrics = pd.DataFrame({
        "metric": [
            "raw_rows", "clean_rows", "zones_in_complete_panel",
            "total_expected_surplus", "total_expected_deficit", "vehicles_rebalanced",
            "internal_deficit_coverage", "greedy_total_proxy_vehicle_miles",
            "optimized_total_proxy_vehicle_miles", "distance_reduction_vs_greedy",
        ],
        "value": [
            raw_rows, clean_rows, len(imbalance), total_surplus, total_deficit, max_moves,
            max_moves / total_deficit if total_deficit else np.nan,
            greedy_miles, optimal_miles, improvement,
        ],
    })
    metrics.to_csv(OUTPUT_DIR / "project5_metrics.csv", index=False)

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
