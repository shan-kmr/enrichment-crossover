#!/usr/bin/env python3
"""Preprocess NYC TLC taxi data for both tasks (ranking + duration).

Steps:
  1. Load zone centroids from shapefile → zone_centroids.json, zone_to_cell.json
  2. Load & filter trip parquet files (train + test periods)
  3. Compute train-only aggregates → train_aggregates.json
  4. Build ranking task datasets (origin-based, 20 candidates each)
  5. Build duration task datasets (subsampled trips)

Outputs written to data/processed/nyc_taxi/.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

random.seed(42)
np.random.seed(42)


def geocell(lat: float, lon: float, resolution: int = 3) -> str:
    return f"{round(lat, resolution)},{round(lon, resolution)}"


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_zone_centroids(cfg):
    zones_dir = ROOT / cfg["dataset"]["zones_dir"]
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("geopandas required: pip install geopandas")

    shp_files = list(zones_dir.rglob("*.shp"))
    if not shp_files:
        sys.exit(f"No .shp in {zones_dir}")

    gdf = gpd.read_file(shp_files[0]).to_crs(epsg=4326)
    centroids = {}
    for _, row in gdf.iterrows():
        loc_id = int(row.get("LocationID", row.get("OBJECTID", 0)))
        if loc_id == 0:
            continue
        c = row.geometry.centroid
        centroids[loc_id] = (round(c.y, 6), round(c.x, 6))
    return centroids


def load_zone_lookup(cfg):
    import csv
    lookup = {}
    path = ROOT / cfg["dataset"]["zone_lookup"]
    with open(path) as f:
        for row in csv.DictReader(f):
            lookup[int(row["LocationID"])] = {
                "borough": row.get("Borough", ""),
                "zone_name": row.get("Zone", ""),
            }
    return lookup


def load_trips(cfg, months: list[str]) -> pd.DataFrame:
    raw_dir = ROOT / cfg["dataset"]["raw_dir"]
    dfs = []
    for m in months:
        path = raw_dir / f"yellow_tripdata_{m}.parquet"
        if not path.exists():
            print(f"  WARNING: {path.name} not found, skipping")
            continue
        print(f"  Loading {path.name} ...")
        df = pd.read_parquet(path, columns=[
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "PULocationID", "DOLocationID", "trip_distance",
        ])
        dfs.append(df)
    if not dfs:
        sys.exit("No parquet files found!")
    return pd.concat(dfs, ignore_index=True)


def filter_trips(df: pd.DataFrame, cfg) -> pd.DataFrame:
    ds = cfg["dataset"]
    excluded = set(ds["excluded_zones"])
    n0 = len(df)
    df = df[~df["PULocationID"].isin(excluded) & ~df["DOLocationID"].isin(excluded)].copy()
    df["duration_min"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60
    df = df[(df["duration_min"] >= ds["min_duration_min"]) & (df["duration_min"] <= ds["max_duration_min"])]
    df = df[df["trip_distance"] > 0]
    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["pickup_dow"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date.astype(str)
    print(f"  Filtered: {n0:,} → {len(df):,} trips")
    return df


def compute_aggregates(train_df: pd.DataFrame, centroids: dict, lookup: dict) -> dict:
    """Compute all train-only aggregates needed for both tasks."""
    print("Computing train aggregates ...")

    zone_pu_count = train_df.groupby("PULocationID").size().to_dict()
    zone_do_count = train_df.groupby("DOLocationID").size().to_dict()

    od_groups = train_df.groupby(["PULocationID", "DOLocationID"])
    od_count = od_groups.size().to_dict()
    od_mean_dur = od_groups["duration_min"].mean().to_dict()

    zone_mean_dur_pu = train_df.groupby("PULocationID")["duration_min"].mean().to_dict()
    zone_mean_dur_do = train_df.groupby("DOLocationID")["duration_min"].mean().to_dict()

    origin_unique_do = train_df.groupby("PULocationID")["DOLocationID"].nunique().to_dict()

    origin_active_days = train_df.groupby("PULocationID")["pickup_date"].nunique().to_dict()

    # Origin geo-spread: std of distances from origin centroid to all dropoff centroids
    origin_geo_spread = {}
    for pu_zone, group in train_df.groupby("PULocationID"):
        if pu_zone not in centroids:
            continue
        pu_lat, pu_lon = centroids[pu_zone]
        do_zones = group["DOLocationID"].unique()
        dists = []
        for dz in do_zones:
            if dz in centroids:
                dz_lat, dz_lon = centroids[dz]
                dists.append(haversine_km(pu_lat, pu_lon, dz_lat, dz_lon))
        origin_geo_spread[pu_zone] = float(np.std(dists)) if len(dists) > 1 else 0.0

    # Recent 5 trips per origin (last 5 by time)
    train_sorted = train_df.sort_values("tpep_pickup_datetime")
    recent5 = {}
    for pu_zone, group in train_sorted.groupby("PULocationID"):
        last5 = group.tail(5)
        recent5[pu_zone] = [
            {"DOLocationID": int(r["DOLocationID"]),
             "duration_min": round(float(r["duration_min"]), 2),
             "pickup_hour": int(r["pickup_hour"])}
            for _, r in last5.iterrows()
        ]

    # Hourly duration stats per origin
    hourly_dur = {}
    for (pu, hr), group in train_df.groupby(["PULocationID", "pickup_hour"]):
        key = f"{pu}_{hr}"
        hourly_dur[key] = {
            "mean": round(float(group["duration_min"].mean()), 2),
            "std": round(float(group["duration_min"].std()), 2)
            if len(group) > 1 else 0.0,
        }

    # Mean dropoff hour per origin
    mean_do_hour = {}
    std_do_hour = {}
    for pu_zone, group in train_df.groupby("PULocationID"):
        hours = group["pickup_hour"].values.astype(float)
        mean_do_hour[pu_zone] = round(float(hours.mean()), 2)
        std_do_hour[pu_zone] = round(float(hours.std()), 2) if len(hours) > 1 else 0.0

    # Avg hour at candidate (OD pair)
    avg_hour_od = {}
    for (pu, do), group in train_df.groupby(["PULocationID", "DOLocationID"]):
        avg_hour_od[f"{pu}_{do}"] = round(float(group["pickup_hour"].mean()), 2)

    # Zone borough
    zone_borough = {z: info.get("borough", "") for z, info in lookup.items()}

    agg = {
        "zone_pu_count": {str(k): v for k, v in zone_pu_count.items()},
        "zone_do_count": {str(k): v for k, v in zone_do_count.items()},
        "od_count": {f"{k[0]}_{k[1]}": v for k, v in od_count.items()},
        "od_mean_dur": {f"{k[0]}_{k[1]}": round(v, 2) for k, v in od_mean_dur.items()},
        "zone_mean_dur_pu": {str(k): round(v, 2) for k, v in zone_mean_dur_pu.items()},
        "zone_mean_dur_do": {str(k): round(v, 2) for k, v in zone_mean_dur_do.items()},
        "origin_unique_do": {str(k): v for k, v in origin_unique_do.items()},
        "origin_active_days": {str(k): v for k, v in origin_active_days.items()},
        "origin_geo_spread": {str(k): round(v, 4) for k, v in origin_geo_spread.items()},
        "recent5": {str(k): v for k, v in recent5.items()},
        "hourly_dur": hourly_dur,
        "mean_do_hour": {str(k): v for k, v in mean_do_hour.items()},
        "std_do_hour": {str(k): v for k, v in std_do_hour.items()},
        "avg_hour_od": avg_hour_od,
        "zone_borough": {str(k): v for k, v in zone_borough.items()},
    }
    return agg


def build_ranking_cases(train_df, test_df, cfg, centroids, lookup, agg):
    """Build ranking datasets: origin → 20 candidates (1 true + 19 neg)."""
    rl = cfg["ranking"]
    n_cands = rl["n_candidates"]
    n_train_origins = rl["n_train_origins"]
    n_test_origins = rl["n_test_origins"]

    zone_borough = {z: info.get("borough", "") for z, info in lookup.items()}
    all_zones = sorted(set(centroids.keys()) - set(cfg["dataset"]["excluded_zones"]))
    borough_zones = defaultdict(list)
    for z in all_zones:
        b = zone_borough.get(z, "")
        if b:
            borough_zones[b].append(z)

    def _make_cases(df, n_queries, label_prefix):
        # We need n_queries ranking instances (each = 1 origin + 1 gt trip + 20 candidates = 20 pointwise rows).
        # Build all (origin, gt_trip) pairs from the dataframe, then sample n_queries without replacement
        # so we get no duplicate ranking instances (same origin + same correct dropoff).
        origin_trips = {
            int(pu_zone): group.to_dict("records")
            for pu_zone, group in df.groupby("PULocationID")
        }

        all_pairs = []
        for origin_zone, trips in origin_trips.items():
            for t in trips:
                all_pairs.append((origin_zone, t))

        if len(all_pairs) >= n_queries:
            chosen = random.sample(all_pairs, n_queries)
        else:
            chosen = random.choices(all_pairs, k=n_queries)  # fallback with replacement if very few trips

        cases = []
        for origin_zone, gt_trip in chosen:
            gt_do = int(gt_trip["DOLocationID"])

            borough = zone_borough.get(origin_zone, "")
            pool = [z for z in borough_zones.get(borough, all_zones)
                    if z != gt_do and z != origin_zone]
            if len(pool) < n_cands - 1:
                pool = [z for z in all_zones if z != gt_do and z != origin_zone]

            negs = random.sample(pool, min(n_cands - 1, len(pool)))

            for cand_zone in [gt_do] + negs:
                label = 1 if cand_zone == gt_do else 0
                cases.append({
                    "origin_zone": origin_zone,
                    "candidate_zone": cand_zone,
                    "label": label,
                    "pickup_hour": int(gt_trip["pickup_hour"]),
                    "pickup_dow": int(gt_trip["pickup_dow"]),
                })
        return cases

    print("Building ranking train cases ...")
    train_cases = _make_cases(train_df, n_train_origins, "train")
    print(f"  {len(train_cases)} pointwise cases ({n_train_origins} queries × {n_cands} candidates)")

    print("Building ranking test cases ...")
    test_cases = _make_cases(test_df, n_test_origins, "test")
    print(f"  {len(test_cases)} pointwise cases ({n_test_origins} queries × {n_cands} candidates)")

    return train_cases, test_cases


def build_duration_cases(train_df, test_df, cfg):
    """Subsample trips for duration prediction."""
    dur = cfg["duration"]

    def _sample(df, n):
        if len(df) <= n:
            return df
        return df.sample(n=n, random_state=42)

    train_sample = _sample(train_df, dur["n_train_samples"])
    test_sample = _sample(test_df, dur["n_test_samples"])

    def _to_records(df):
        return [{
            "PULocationID": int(r["PULocationID"]),
            "DOLocationID": int(r["DOLocationID"]),
            "duration_min": round(float(r["duration_min"]), 2),
            "trip_distance": round(float(r["trip_distance"]), 3),
            "pickup_hour": int(r["pickup_hour"]),
            "pickup_dow": int(r["pickup_dow"]),
        } for _, r in df.iterrows()]

    print(f"Duration: {len(train_sample)} train, {len(test_sample)} test")
    return _to_records(train_sample), _to_records(test_sample)


def main():
    cfg = yaml.safe_load((ROOT / "nyc_taxi_config.yaml").read_text())
    ds = cfg["dataset"]
    out_dir = ROOT / ds["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Zone centroids
    print("=== Zone centroids ===")
    centroids = load_zone_centroids(cfg)
    lookup = load_zone_lookup(cfg)

    centroids_path = out_dir / "zone_centroids.json"
    centroids_path.write_text(json.dumps(
        {str(k): list(v) for k, v in centroids.items()}, indent=2))
    print(f"  Saved {len(centroids)} centroids → {centroids_path}")

    zone_to_cell = {str(z): geocell(lat, lon) for z, (lat, lon) in centroids.items()}
    (out_dir / "zone_to_cell.json").write_text(json.dumps(zone_to_cell, indent=2))
    print(f"  Saved zone_to_cell ({len(zone_to_cell)} zones)")

    # 2. Load trips
    print("\n=== Loading train trips ===")
    train_df = load_trips(cfg, ds["train_months"])
    train_df = filter_trips(train_df, cfg)

    print("\n=== Loading test trips ===")
    test_df = load_trips(cfg, ds["test_months"])
    test_df = filter_trips(test_df, cfg)

    # 3. Train aggregates
    print("\n=== Train aggregates ===")
    agg = compute_aggregates(train_df, centroids, lookup)
    agg_path = out_dir / "train_aggregates.json"
    agg_path.write_text(json.dumps(agg))
    print(f"  Saved → {agg_path}")

    # Also compute cell-level aggregates for latlng tier (vectorized)
    cell_series = train_df["DOLocationID"].astype(int).astype(str).map(zone_to_cell)
    cell_do_count = cell_series.dropna().value_counts().to_dict()
    (out_dir / "cell_popularity.json").write_text(
        json.dumps(cell_do_count))
    print(f"  cell_popularity: {len(cell_do_count)} cells")

    # 4. Ranking task
    print("\n=== Ranking task ===")
    ranking_dir = out_dir / "ranking"
    ranking_dir.mkdir(exist_ok=True)
    rank_train, rank_test = build_ranking_cases(
        train_df, test_df, cfg, centroids, lookup, agg)
    (ranking_dir / "train_cases.json").write_text(json.dumps(rank_train))
    (ranking_dir / "test_cases.json").write_text(json.dumps(rank_test))

    # 5. Duration task
    print("\n=== Duration task ===")
    dur_dir = out_dir / "duration"
    dur_dir.mkdir(exist_ok=True)
    dur_train, dur_test = build_duration_cases(train_df, test_df, cfg)
    (dur_dir / "train_cases.json").write_text(json.dumps(dur_train))
    (dur_dir / "test_cases.json").write_text(json.dumps(dur_test))

    print("\n=== Done! ===")
    print(f"All outputs in {out_dir}/")


if __name__ == "__main__":
    main()
