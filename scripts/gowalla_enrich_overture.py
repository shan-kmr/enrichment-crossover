#!/usr/bin/env python3
"""Download Overture Maps US places and match to Gowalla check-in locations.

Phase 1: Download Overture places from S3 via DuckDB → local parquet
Phase 2: Build BallTree spatial index, match Gowalla locations within 50m
Phase 3: Save location_id → category mapping + match statistics

Dependencies: pip install duckdb scikit-learn
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

from src.gowalla.data_loader import load_checkins, load_test_cases, save_test_cases
from src.gowalla.enrichment import get_category

OVERTURE_RELEASE = "2026-02-18.0"
MIN_KNOWN_RATIO = 0.80  # both users need ≥80% known categories
MATCH_RADIUS_M = 50
EARTH_RADIUS_M = 6_371_000
US_BBOX = (-130, 24, -60, 50)  # west, south, east, north

RAW_DIR = ROOT / "data" / "raw" / "gowalla"
PROCESSED_DIR = ROOT / "data" / "processed" / "gowalla"
OVERTURE_PARQUET = RAW_DIR / "overture_us_places.parquet"
CATEGORIES_PATH = PROCESSED_DIR / "overture_categories.json"


def phase1_download():
    """Download Overture US places via DuckDB S3 query → local parquet."""
    if OVERTURE_PARQUET.exists():
        print(f"Phase 1: {OVERTURE_PARQUET.name} exists, skipping download")
        return

    import duckdb

    print("Phase 1: Downloading Overture Maps US places...")
    print(f"  Release: {OVERTURE_RELEASE}")
    print(f"  Bounding box: {US_BBOX}")
    print("  This may take 30-60 minutes depending on your connection...")

    con = duckdb.connect()
    con.execute("INSTALL spatial; INSTALL httpfs; LOAD spatial; LOAD httpfs;")
    con.execute("SET s3_region = 'us-west-2'")

    OVERTURE_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    s3_path = (
        f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}"
        f"/theme=places/type=place/*"
    )

    start = time.time()
    con.execute(f"""
        COPY (
            SELECT
                names.primary as name,
                categories.primary as category,
                ST_Y(geometry) as lat,
                ST_X(geometry) as lon
            FROM read_parquet('{s3_path}', filename=true, hive_partitioning=1)
            WHERE bbox.xmin BETWEEN {US_BBOX[0]} AND {US_BBOX[2]}
              AND bbox.ymin BETWEEN {US_BBOX[1]} AND {US_BBOX[3]}
        ) TO '{OVERTURE_PARQUET}' (FORMAT PARQUET)
    """)
    elapsed = time.time() - start
    print(f"  Downloaded in {elapsed:.0f}s")

    count = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{OVERTURE_PARQUET}')"
    ).fetchone()[0]
    print(f"  Total US POIs: {count:,}")
    con.close()


def phase2_match():
    """Match Gowalla locations to nearest Overture POI within 50m."""
    if CATEGORIES_PATH.exists():
        print(f"\nPhase 2: {CATEGORIES_PATH.name} exists, skipping matching")
        return

    from sklearn.neighbors import BallTree
    import duckdb

    print("\nPhase 2: Matching Gowalla locations to Overture POIs...")

    # --- Get canonical (lat, lon) per Gowalla location_id ---
    print("  Loading Gowalla check-ins...")
    checkins_path = RAW_DIR / "loc-gowalla_totalCheckins.txt"
    user_checkins = load_checkins(checkins_path)

    loc_coords: dict[int, list[tuple[float, float]]] = {}
    for cis in user_checkins.values():
        for c in cis:
            loc_coords.setdefault(c.location_id, []).append(
                (c.latitude, c.longitude)
            )
    print(f"  Unique Gowalla location IDs: {len(loc_coords):,}")

    loc_canonical: dict[int, tuple[float, float]] = {}
    for loc_id, coords in loc_coords.items():
        lats, lons = zip(*coords)
        loc_canonical[loc_id] = (float(np.median(lats)), float(np.median(lons)))

    loc_ids = list(loc_canonical.keys())
    gowalla_coords = np.array([loc_canonical[lid] for lid in loc_ids])

    # --- Load Overture POIs ---
    print("  Loading Overture places from local parquet...")
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    df = con.execute(f"""
        SELECT name, category, lat, lon
        FROM read_parquet('{OVERTURE_PARQUET}')
        WHERE category IS NOT NULL
    """).df()
    con.close()
    print(f"  Overture POIs with category: {len(df):,}")

    overture_coords = df[["lat", "lon"]].values.astype(np.float64)

    # --- Build BallTree ---
    print("  Building spatial index (BallTree with haversine)...")
    overture_rad = np.radians(overture_coords)
    tree = BallTree(overture_rad, metric="haversine")

    # --- Query ---
    print(f"  Querying {len(gowalla_coords):,} locations (radius={MATCH_RADIUS_M}m)...")
    gowalla_rad = np.radians(gowalla_coords)
    distances, indices = tree.query(gowalla_rad, k=1)
    distances_m = distances.flatten() * EARTH_RADIUS_M
    indices = indices.flatten()

    # --- Build mapping ---
    mapping: dict[str, dict] = {}
    matched = 0
    for i, loc_id in enumerate(loc_ids):
        if distances_m[i] <= MATCH_RADIUS_M:
            idx = int(indices[i])
            mapping[str(loc_id)] = {
                "category": str(df.iloc[idx]["category"]),
                "name": str(df.iloc[idx]["name"]) if df.iloc[idx]["name"] else None,
                "distance_m": round(float(distances_m[i]), 1),
            }
            matched += 1
        else:
            mapping[str(loc_id)] = {
                "category": "unknown",
                "name": None,
                "distance_m": round(float(distances_m[i]), 1),
            }

    # --- Save ---
    CATEGORIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    CATEGORIES_PATH.write_text(json.dumps(mapping))

    match_rate = matched / len(loc_ids) * 100
    print(f"\n  Match statistics:")
    print(f"    Total locations:          {len(loc_ids):,}")
    print(f"    Matched (≤{MATCH_RADIUS_M}m):         {matched:,} ({match_rate:.1f}%)")
    print(f"    Unmatched (→ 'unknown'):  {len(loc_ids) - matched:,} ({100 - match_rate:.1f}%)")

    cats = Counter(
        v["category"] for v in mapping.values() if v["category"] != "unknown"
    )
    print(f"\n  Top 20 matched categories:")
    for cat, count in cats.most_common(20):
        print(f"    {cat}: {count:,}")

    print(f"\n  Saved to {CATEGORIES_PATH}")


def _user_known_ratio(profile, cat_map: dict) -> float:
    if not profile.checkins:
        return 0.0
    known = sum(1 for c in profile.checkins if get_category(c.location_id, cat_map) != "unknown")
    return known / len(profile.checkins)


def phase3_filter():
    """Filter test/train cases to pairs where both users have ≥80% known categories."""
    filt_dir = PROCESSED_DIR / "enriched"
    filt_dir.mkdir(parents=True, exist_ok=True)

    cat_map = json.loads(CATEGORIES_PATH.read_text())

    case_files = [
        ("test_cases.json", "test_cases.json"),
        ("train_cases.json", "train_cases.json"),
        ("ml_test_cases.json", "ml_test_cases.json"),
        ("ml_train_cases.json", "ml_train_cases.json"),
    ]

    print(f"\nPhase 3: Filtering to pairs where both users have ≥{MIN_KNOWN_RATIO*100:.0f}% known categories...")

    for src_name, dst_name in case_files:
        src_path = PROCESSED_DIR / src_name
        dst_path = filt_dir / dst_name

        if not src_path.exists():
            print(f"  {src_name}: not found, skipping")
            continue

        cases = load_test_cases(src_path)
        filtered = []
        for tc in cases:
            ra = _user_known_ratio(tc.user_a, cat_map)
            rb = _user_known_ratio(tc.user_b, cat_map)
            if ra >= MIN_KNOWN_RATIO and rb >= MIN_KNOWN_RATIO:
                filtered.append(tc)

        save_test_cases(filtered, dst_path)
        pos = sum(1 for tc in filtered if tc.label == 1)
        print(f"  {src_name}: {len(cases)} → {len(filtered)} pairs "
              f"({pos} pos, {len(filtered)-pos} neg)")

    print(f"\n  Filtered cases saved to {filt_dir}/")


def main():
    phase1_download()
    phase2_match()
    phase3_filter()
    print("\nDone! Next: run enriched experiments on filtered data.")


if __name__ == "__main__":
    main()
