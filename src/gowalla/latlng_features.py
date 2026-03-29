"""Pure lat/lng feature computation -- no venue IDs, no categories.

Uses geo-cell grid (rounding to `resolution` decimal places) to approximate
co-visitation from raw coordinates only.  This simulates the common scenario
where a practitioner has GPS traces but no places database.
"""

from __future__ import annotations

from src.gowalla.data_loader import FriendshipTestCase, UserProfile


def _geocell(lat: float, lon: float, resolution: int = 3) -> tuple:
    return (round(lat, resolution), round(lon, resolution))


def _user_cells(profile: UserProfile, resolution: int = 3) -> set[tuple]:
    return {_geocell(c.latitude, c.longitude, resolution) for c in profile.checkins}


def compute_latlng_pair_features(
    tc: FriendshipTestCase, resolution: int = 3
) -> dict:
    cells_a = _user_cells(tc.user_a, resolution)
    cells_b = _user_cells(tc.user_b, resolution)
    shared = len(cells_a & cells_b)
    union = len(cells_a | cells_b)
    jaccard = shared / union if union else 0.0

    a_by_cell: dict[tuple, list[str]] = {}
    for c in tc.user_a.checkins:
        cell = _geocell(c.latitude, c.longitude, resolution)
        a_by_cell.setdefault(cell, []).append(c.timestamp)

    co_occ = 0
    for c in tc.user_b.checkins:
        cell = _geocell(c.latitude, c.longitude, resolution)
        if cell in a_by_cell:
            for ts_a in a_by_cell[cell]:
                if ts_a[:13] == c.timestamp[:13]:
                    co_occ += 1
                    break

    return {
        "unique_cells_a": len(cells_a),
        "unique_cells_b": len(cells_b),
        "shared_cells": shared,
        "jaccard_cells": round(jaccard, 4),
        "temporal_co_cells": co_occ,
    }


def compute_latlng_handcrafted(
    tc: FriendshipTestCase, granularity: str, resolution: int = 3
) -> list[float]:
    a, b = tc.user_a, tc.user_b
    pair = compute_latlng_pair_features(tc, resolution)

    feats = [
        float(a.total_checkins),
        float(b.total_checkins),
        float(a.total_checkins + b.total_checkins),
    ]

    if granularity in ("G1", "G2", "G3", "G4"):
        feats.extend([
            float(tc.same_region),
            tc.centroid_distance_km,
            float(a.primary_region == b.primary_region),
        ])

    if granularity in ("G2", "G3", "G4"):
        feats.extend([
            float(pair["unique_cells_a"]),
            float(pair["unique_cells_b"]),
            float(pair["shared_cells"]),
            pair["jaccard_cells"],
            a.geo_spread_km,
            b.geo_spread_km,
        ])

    if granularity in ("G3", "G4"):
        feats.extend([
            float(a.active_days),
            float(b.active_days),
        ])

    if granularity == "G4":
        feats.extend([
            float(pair["temporal_co_cells"]),
            tc.centroid_distance_km / max(a.geo_spread_km + b.geo_spread_km, 0.01),
            pair["jaccard_cells"] * float(pair["temporal_co_cells"] + 1),
        ])

    return feats
