#!/usr/bin/env python3
"""Enrich NYC TLC taxi zones with Overture Maps POI categories.

Reuses the existing Overture US places parquet (data/raw/gowalla/overture_us_places.parquet).
For each of the 263 TLC zones, finds POIs within 1 km of the zone centroid and assigns
a primary category (most common) plus the full set of unique categories.

Output: data/processed/nyc_taxi/zone_categories.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OVERTURE_PARQUET = ROOT / "data" / "raw" / "gowalla" / "overture_us_places.parquet"
ZONE_SHAPEFILE_DIR = ROOT / "data" / "raw" / "nyc_taxi" / "zones"
ZONE_LOOKUP_CSV = ZONE_SHAPEFILE_DIR / "taxi_zone_lookup.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "nyc_taxi"
OUTPUT_PATH = OUTPUT_DIR / "zone_categories.json"

MATCH_RADIUS_M = 1000
EARTH_RADIUS_M = 6_371_000


def load_zone_centroids() -> dict[int, tuple[float, float]]:
    """Load zone centroids from the TLC taxi zone shapefile."""
    try:
        import geopandas as gpd
    except ImportError:
        sys.exit("geopandas required: pip install geopandas")

    shp_files = list(ZONE_SHAPEFILE_DIR.rglob("*.shp"))
    if not shp_files:
        sys.exit(f"No .shp file found in {ZONE_SHAPEFILE_DIR}")

    gdf = gpd.read_file(shp_files[0])
    gdf = gdf.to_crs(epsg=4326)

    centroids: dict[int, tuple[float, float]] = {}
    for _, row in gdf.iterrows():
        loc_id = int(row.get("LocationID", row.get("OBJECTID", 0)))
        if loc_id == 0:
            continue
        c = row.geometry.centroid
        centroids[loc_id] = (c.y, c.x)

    print(f"Loaded {len(centroids)} zone centroids from shapefile")
    return centroids


def load_zone_lookup() -> dict[int, dict]:
    """Load borough/zone name from the TLC lookup CSV."""
    import csv
    lookup: dict[int, dict] = {}
    with open(ZONE_LOOKUP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            loc_id = int(row["LocationID"])
            lookup[loc_id] = {
                "borough": row.get("Borough", ""),
                "zone_name": row.get("Zone", ""),
                "service_zone": row.get("service_zone", ""),
            }
    return lookup


def main():
    if OUTPUT_PATH.exists():
        print(f"Output already exists: {OUTPUT_PATH}")
        print("Delete it to re-run enrichment.")
        return

    if not OVERTURE_PARQUET.exists():
        sys.exit(
            f"Overture parquet not found: {OVERTURE_PARQUET}\n"
            "Run gowalla_enrich_overture.py phase1 first."
        )

    centroids = load_zone_centroids()
    lookup = load_zone_lookup()

    # Load Overture POIs (NYC area only for speed)
    import duckdb
    from sklearn.neighbors import BallTree

    print("Loading Overture POIs (NYC bounding box) ...")
    NYC_BBOX = (-74.3, 40.4, -73.7, 40.95)
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    df = con.execute(f"""
        SELECT name, category, lat, lon
        FROM read_parquet('{OVERTURE_PARQUET}')
        WHERE category IS NOT NULL
          AND lon BETWEEN {NYC_BBOX[0]} AND {NYC_BBOX[2]}
          AND lat BETWEEN {NYC_BBOX[1]} AND {NYC_BBOX[3]}
    """).df()
    con.close()
    print(f"  {len(df):,} NYC-area POIs with category")

    overture_coords = df[["lat", "lon"]].values.astype(np.float64)
    overture_rad = np.radians(overture_coords)
    print("Building BallTree ...")
    tree = BallTree(overture_rad, metric="haversine")

    zone_ids = sorted(centroids.keys())
    zone_coords = np.array([centroids[z] for z in zone_ids], dtype=np.float64)
    zone_rad = np.radians(zone_coords)

    radius_rad = MATCH_RADIUS_M / EARTH_RADIUS_M

    print(f"Querying {len(zone_ids)} zone centroids (radius={MATCH_RADIUS_M}m) ...")
    start = time.time()
    indices_list = tree.query_radius(zone_rad, r=radius_rad)
    elapsed = time.time() - start
    print(f"  Query done in {elapsed:.1f}s")

    mapping: dict[str, dict] = {}
    for i, zone_id in enumerate(zone_ids):
        matched_indices = indices_list[i]
        cats = [str(df.iloc[idx]["category"]) for idx in matched_indices]
        counter = Counter(cats)
        primary = counter.most_common(1)[0][0] if counter else "unknown"
        all_cats = sorted(set(cats)) if cats else []

        info = lookup.get(zone_id, {})
        mapping[str(zone_id)] = {
            "primary_category": primary,
            "all_categories": all_cats,
            "poi_count": len(matched_indices),
            "borough": info.get("borough", ""),
            "zone_name": info.get("zone_name", ""),
        }

    # Stats
    matched_zones = sum(1 for v in mapping.values() if v["poi_count"] > 0)
    total_pois = sum(v["poi_count"] for v in mapping.values())
    all_primaries = Counter(v["primary_category"] for v in mapping.values())
    print(f"\nEnrichment results:")
    print(f"  Zones with POIs: {matched_zones}/{len(zone_ids)}")
    print(f"  Total matched POIs: {total_pois:,}")
    print(f"  Top 15 primary categories:")
    for cat, cnt in all_primaries.most_common(15):
        print(f"    {cat}: {cnt}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(mapping, indent=2))
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
