"""NYC taxi feature computation for both tasks, all 3 tiers, G0-G4.

Tier parameter controls place logic:
  latlng   - geo-cell (~100 m) matching via zone centroids
  zone_id  - exact TLC LocationID matching
  enriched - zone_id + Overture category features

Mirrors the structure of src/gowalla/nextloc_features.py.
"""

from __future__ import annotations

import math
from collections import Counter


def _geocell(lat: float, lon: float, resolution: int = 3) -> str:
    return f"{round(lat, resolution)},{round(lon, resolution)}"


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


# ─── Ranking features ───────────────────────────────────────────────


def compute_ranking_features(
    case: dict,
    granularity: str,
    tier: str,
    agg: dict,
    centroids: dict,
    zone_to_cell: dict,
    zone_cats: dict | None = None,
    cell_pop: dict | None = None,
) -> list[float]:
    """Compute features for a ranking case (origin → candidate dropoff zone)."""
    origin = case["origin_zone"]
    cand = case["candidate_zone"]
    origin_s, cand_s = str(origin), str(cand)

    # Tier-specific place resolution
    if tier == "latlng":
        origin_cell = zone_to_cell.get(origin_s, "")
        cand_cell = zone_to_cell.get(cand_s, "")
        # Cell-based unique dropoffs for this origin
        recent5 = agg.get("recent5", {}).get(origin_s, [])
        origin_do_cells = set()
        for r in recent5:
            do_z = str(r["DOLocationID"])
            c = zone_to_cell.get(do_z, "")
            if c:
                origin_do_cells.add(c)
        unique_dropoffs = len(origin_do_cells) if origin_do_cells else \
            int(agg.get("origin_unique_do", {}).get(origin_s, 0))
        popularity = (cell_pop or {}).get(cand_cell, 0)
        visited = cand_cell in origin_do_cells
        od_key = f"{origin_cell}_{cand_cell}"
    else:
        unique_dropoffs = int(agg.get("origin_unique_do", {}).get(origin_s, 0))
        popularity = int(agg.get("zone_do_count", {}).get(cand_s, 0))
        od_key = f"{origin}_{cand}"
        visited = int(agg.get("od_count", {}).get(od_key, 0)) > 0

    origin_total = int(agg.get("zone_pu_count", {}).get(origin_s, 0))
    trips_to_cand = int(agg.get("od_count", {}).get(od_key, 0))

    # Category info (enriched G1+)
    origin_cat_info = (zone_cats or {}).get(origin_s, {})
    cand_cat_info = (zone_cats or {}).get(cand_s, {})
    origin_cats_set = set(origin_cat_info.get("all_categories", []))
    cand_primary = cand_cat_info.get("primary_category", "unknown")

    feats: list[float] = []

    # G0 (3 features)
    feats.extend([
        float(origin_total),
        float(unique_dropoffs),
        float(popularity),
    ])

    # G1 (+2 base, +3 enriched)
    if granularity in ("G1", "G2", "G3", "G4"):
        origin_borough = agg.get("zone_borough", {}).get(origin_s, "")
        cand_borough = agg.get("zone_borough", {}).get(cand_s, "")
        same_borough = float(origin_borough == cand_borough and origin_borough != "")

        o_lat, o_lon = centroids.get(origin, (0, 0))
        c_lat, c_lon = centroids.get(cand, (0, 0))
        dist_km = _haversine_km(o_lat, o_lon, c_lat, c_lon) if o_lat else 0.0

        feats.extend([same_borough, dist_km])

        if tier == "enriched" and zone_cats:
            origin_primary = origin_cat_info.get("primary_category", "unknown")
            feats.extend([
                float(origin_primary != "unknown"),
                float(cand_primary != "unknown"),
                float(cand_primary in origin_cats_set and cand_primary != "unknown"),
            ])

    # G2 (+5 base, +3 enriched)
    if granularity in ("G2", "G3", "G4"):
        geo_spread = float(agg.get("origin_geo_spread", {}).get(origin_s, 0))
        active_days = float(agg.get("origin_active_days", {}).get(origin_s, 0))

        recent5 = agg.get("recent5", {}).get(origin_s, [])
        if recent5:
            last_do = recent5[-1]["DOLocationID"]
            ld_lat, ld_lon = centroids.get(last_do, (0, 0))
            dist_last = _haversine_km(ld_lat, ld_lon, c_lat, c_lon) if ld_lat else 0.0
        else:
            dist_last = 0.0

        feats.extend([
            geo_spread,
            active_days,
            float(visited),
            float(trips_to_cand),
            dist_last,
        ])

        if tier == "enriched" and zone_cats:
            origin_all_cats = origin_cat_info.get("all_categories", [])
            cat_counter = Counter(origin_all_cats)
            feats.extend([
                _entropy(cat_counter),
                float(cat_counter.get(cand_primary, 0)),
                cat_counter.get(cand_primary, 0) / max(sum(cat_counter.values()), 1),
            ])

    # G3 (+3 base, +1 enriched)
    if granularity in ("G3", "G4"):
        recent5 = agg.get("recent5", {}).get(origin_s, [])
        dists = []
        for r in recent5:
            do_z = r["DOLocationID"]
            dl, dlo = centroids.get(do_z, (0, 0))
            if dl:
                dists.append(_haversine_km(dl, dlo, c_lat, c_lon))

        feats.extend([
            min(dists) if dists else 0.0,
            sum(dists) / len(dists) if dists else 0.0,
            dists[-1] if dists else 0.0,
        ])

        if tier == "enriched" and zone_cats:
            same_cat_count = sum(
                1 for r in recent5
                if (zone_cats or {}).get(str(r["DOLocationID"]), {})
                .get("primary_category", "") == cand_primary
                and cand_primary != "unknown"
            )
            feats.append(float(same_cat_count))

    # G4 (+3)
    if granularity == "G4":
        mean_h = float(agg.get("mean_do_hour", {}).get(origin_s, 12.0))
        std_h = float(agg.get("std_do_hour", {}).get(origin_s, 0.0))
        avg_h_cand = float(agg.get("avg_hour_od", {}).get(
            f"{origin}_{cand}", -1.0))
        feats.extend([mean_h, std_h, avg_h_cand])

    return feats


# ─── Duration features ──────────────────────────────────────────────


def compute_duration_features(
    case: dict,
    granularity: str,
    tier: str,
    agg: dict,
    centroids: dict,
    zone_to_cell: dict,
    zone_cats: dict | None = None,
    cell_pop: dict | None = None,
) -> list[float]:
    """Compute features for a duration prediction case (one trip)."""
    origin = case["PULocationID"]
    dest = case["DOLocationID"]
    origin_s, dest_s = str(origin), str(dest)

    if tier == "latlng":
        origin_cell = zone_to_cell.get(origin_s, "")
        dest_cell = zone_to_cell.get(dest_s, "")
        origin_total = (cell_pop or {}).get(origin_cell, 0)
        dest_total = (cell_pop or {}).get(dest_cell, 0)
        od_key_agg = f"{origin_cell}_{dest_cell}"
    else:
        origin_total = int(agg.get("zone_pu_count", {}).get(origin_s, 0))
        dest_total = int(agg.get("zone_do_count", {}).get(dest_s, 0))
        od_key_agg = f"{origin}_{dest}"

    mean_dur_origin = float(agg.get("zone_mean_dur_pu", {}).get(origin_s, 0))
    mean_dur_dest = float(agg.get("zone_mean_dur_do", {}).get(dest_s, 0))

    origin_cat_info = (zone_cats or {}).get(origin_s, {})
    dest_cat_info = (zone_cats or {}).get(dest_s, {})

    feats: list[float] = []

    # G0 (4 features)
    feats.extend([
        float(origin_total),
        float(dest_total),
        mean_dur_origin,
        mean_dur_dest,
    ])

    # G1 (+4 base, +3 enriched)
    if granularity in ("G1", "G2", "G3", "G4"):
        origin_borough = agg.get("zone_borough", {}).get(origin_s, "")
        dest_borough = agg.get("zone_borough", {}).get(dest_s, "")
        same_borough = float(origin_borough == dest_borough and origin_borough != "")

        o_lat, o_lon = centroids.get(origin, (0, 0))
        d_lat, d_lon = centroids.get(dest, (0, 0))
        dist_km = _haversine_km(o_lat, o_lon, d_lat, d_lon) if o_lat else 0.0

        feats.extend([
            float(origin),
            float(dest),
            same_borough,
            dist_km,
        ])

        if tier == "enriched" and zone_cats:
            o_primary = origin_cat_info.get("primary_category", "unknown")
            d_primary = dest_cat_info.get("primary_category", "unknown")
            feats.extend([
                float(o_primary != "unknown"),
                float(d_primary != "unknown"),
                float(o_primary == d_primary and o_primary != "unknown"),
            ])

    # G2 (+5 base, +3 enriched)
    if granularity in ("G2", "G3", "G4"):
        geo_spread = float(agg.get("origin_geo_spread", {}).get(origin_s, 0))
        active_days = float(agg.get("origin_active_days", {}).get(origin_s, 0))
        trips_od = float(agg.get("od_count", {}).get(od_key_agg, 0))
        mean_dur_od = float(agg.get("od_mean_dur", {}).get(od_key_agg, 0))
        trip_dist = case.get("trip_distance", 0) * 1.60934  # miles to km

        feats.extend([geo_spread, active_days, trips_od, mean_dur_od, trip_dist])

        if tier == "enriched" and zone_cats:
            o_cats = Counter(origin_cat_info.get("all_categories", []))
            d_cats = Counter(dest_cat_info.get("all_categories", []))
            feats.extend([
                _entropy(o_cats),
                _entropy(d_cats),
                float(len(set(o_cats) & set(d_cats))),
            ])

    # G3 (+3 base, +1 enriched)
    if granularity in ("G3", "G4"):
        recent5 = agg.get("recent5", {}).get(origin_s, [])
        durs = [r["duration_min"] for r in recent5]
        feats.extend([
            min(durs) if durs else 0.0,
            sum(durs) / len(durs) if durs else 0.0,
            durs[-1] if durs else 0.0,
        ])

        if tier == "enriched" and zone_cats:
            d_primary = dest_cat_info.get("primary_category", "unknown")
            same_cat_durs = [
                r["duration_min"] for r in recent5
                if (zone_cats or {}).get(str(r["DOLocationID"]), {})
                .get("primary_category", "") == d_primary
                and d_primary != "unknown"
            ]
            feats.append(
                sum(same_cat_durs) / len(same_cat_durs) if same_cat_durs else 0.0
            )

    # G4 (+4)
    if granularity == "G4":
        pickup_hour = float(case.get("pickup_hour", 12))
        pickup_dow = float(case.get("pickup_dow", 0))
        hourly_key = f"{origin}_{int(pickup_hour)}"
        hourly_info = agg.get("hourly_dur", {}).get(hourly_key, {})
        feats.extend([
            pickup_hour,
            pickup_dow,
            float(hourly_info.get("mean", 0)),
            float(hourly_info.get("std", 0)),
        ])

    return feats
