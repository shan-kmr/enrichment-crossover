"""Next-location prediction features for all 3 tiers.

Tier parameter controls co-visitation logic:
  latlng   – geo-cell (~100 m) matching, no venue IDs
  venue_id – exact location_id matching
  enriched – venue_id + Overture category features
"""

from __future__ import annotations

from collections import Counter

from src.gowalla.data_loader import (
    NextLocTestCase, haversine_km, _region_key,
)
from src.gowalla.latlng_features import _geocell
from src.gowalla.enrichment import get_category, user_category_profile

RESOLUTION = 3


def compute_nextloc_handcrafted(
    tc: NextLocTestCase,
    granularity: str,
    tier: str,
    cat_map: dict | None = None,
    cell_pop: dict | None = None,
) -> list[float]:
    user = tc.user

    # ── tier-specific base signals ──────────────────────────────────
    if tier == "latlng":
        unique_count = len(
            {_geocell(c.latitude, c.longitude, RESOLUTION)
             for c in user.checkins})
        cand_cell = _geocell(tc.candidate_lat, tc.candidate_lon, RESOLUTION)
        popularity = (cell_pop or {}).get(
            f"{cand_cell[0]},{cand_cell[1]}", 0)
        cell_counts = Counter(
            _geocell(c.latitude, c.longitude, RESOLUTION)
            for c in user.checkins)
        visited = cand_cell in cell_counts
        visits_to = cell_counts.get(cand_cell, 0)
    else:
        unique_count = user.unique_locations
        popularity = tc.candidate_popularity
        visited = tc.user_visited_candidate
        visits_to = tc.user_visits_to_candidate

    # pre-compute category info once (enriched G1+)
    cand_cat = ""
    cat_prof = None
    user_cats: Counter = Counter()
    if tier == "enriched" and cat_map and granularity != "G0":
        cand_cat = get_category(tc.candidate_location_id, cat_map)
        cat_prof = user_category_profile(user, cat_map)
        user_cats = cat_prof["category_counts"]

    feats: list[float] = []

    # ── G0  (3) ─────────────────────────────────────────────────────
    feats.extend([
        float(user.total_checkins),
        float(unique_count),
        float(popularity),
    ])

    # ── G1  (+2 base, +3 enriched) ──────────────────────────────────
    if granularity in ("G1", "G2", "G3", "G4"):
        cand_region = _region_key(tc.candidate_lat, tc.candidate_lon)
        feats.extend([
            float(cand_region == user.primary_region),
            tc.distance_to_user_centroid_km,
        ])
        if tier == "enriched" and cat_map:
            top_raw = (max(user_cats, key=user_cats.get)
                       if user_cats else "")
            feats.extend([
                float(cand_cat != "unknown"),
                float(cand_cat in user_cats),
                float(top_raw == cand_cat and cand_cat != "unknown"),
            ])

    # ── G2  (+5 base, +3 enriched) ──────────────────────────────────
    if granularity in ("G2", "G3", "G4"):
        last_ci = user.checkins[-1] if user.checkins else None
        dist_last = (haversine_km(last_ci.latitude, last_ci.longitude,
                                  tc.candidate_lat, tc.candidate_lon)
                     if last_ci else 0.0)
        feats.extend([
            user.geo_spread_km,
            float(user.active_days),
            float(visited),
            float(visits_to),
            dist_last,
        ])
        if tier == "enriched" and cat_prof:
            total_known = sum(user_cats.values()) or 1
            feats.extend([
                cat_prof["entropy"],
                float(user_cats.get(cand_cat, 0)),
                user_cats.get(cand_cat, 0) / total_known,
            ])

    # ── G3  (+3 base, +1 enriched) ──────────────────────────────────
    if granularity in ("G3", "G4"):
        recent = user.checkins[-5:]
        dists = [haversine_km(c.latitude, c.longitude,
                              tc.candidate_lat, tc.candidate_lon)
                 for c in recent]
        feats.extend([
            min(dists) if dists else 0.0,
            sum(dists) / len(dists) if dists else 0.0,
            dists[-1] if dists else 0.0,
        ])
        if tier == "enriched" and cat_map:
            recent15 = user.checkins[-15:]
            same_cat = sum(
                1 for c in recent15
                if (get_category(c.location_id, cat_map) == cand_cat
                    and cand_cat != "unknown"))
            feats.append(float(same_cat))

    # ── G4  (+3) ────────────────────────────────────────────────────
    if granularity == "G4":
        hours: list[int] = []
        for c in user.checkins[-15:]:
            try:
                hours.append(int(c.timestamp[11:13]))
            except (ValueError, IndexError):
                pass
        mean_h = sum(hours) / len(hours) if hours else 12.0
        std_h = ((sum((h - mean_h) ** 2 for h in hours)
                  / len(hours)) ** 0.5
                 if len(hours) > 1 else 0.0)

        if tier == "latlng":
            c_cell = _geocell(tc.candidate_lat, tc.candidate_lon,
                              RESOLUTION)
            visit_hrs = [
                int(c.timestamp[11:13]) for c in user.checkins
                if _geocell(c.latitude, c.longitude, RESOLUTION) == c_cell
            ]
        else:
            visit_hrs = [
                int(c.timestamp[11:13]) for c in user.checkins
                if c.location_id == tc.candidate_location_id
            ]
        avg_vh = (sum(visit_hrs) / len(visit_hrs)
                  if visit_hrs else -1.0)
        feats.extend([mean_h, std_h, avg_vh])

    return feats
