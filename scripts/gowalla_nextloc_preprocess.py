#!/usr/bin/env python3
"""Preprocess Gowalla data for next-location prediction (pointwise).

Uses the same user pool as friendship prediction (>=20 check-ins, >=80%
Overture coverage).  For each user the last check-in is the ground truth;
19 negatives are sampled from the same geographic region.

Outputs (to nextloc.data_dir):
  train_cases.json        1000 users x 20 = 20 000 pointwise cases
  test_cases.json          200 users x 20 =  4 000 pointwise cases
  venue_popularity.json    location_id -> visit count (train users only)
  cell_popularity.json     geo-cell     -> visit count (train users only)
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gowalla.data_loader import (
    load_checkins,
    build_user_profile,
    haversine_km,
    _region_key,
    NextLocTestCase,
    save_nextloc_cases,
)
from src.gowalla.enrichment import get_category


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    ds = cfg["dataset"]
    enr = cfg["enrichment"]
    nl = cfg["nextloc"]
    min_known = enr.get("min_known_ratio", 0.80)

    cat_map = json.loads((ROOT / enr["categories_path"]).read_text())
    print(f"Category map: {len(cat_map):,} locations")

    print("Loading check-ins …")
    user_checkins = load_checkins(ROOT / ds["checkins_path"])
    print(f"  {len(user_checkins):,} users total")

    # ── same filters as friendship task ──────────────────────────────
    min_ci = ds["min_user_checkins"]
    eligible = set()
    for uid, cis in user_checkins.items():
        if len(cis) < min_ci:
            continue
        known = sum(1 for c in cis
                    if get_category(c.location_id, cat_map) != "unknown")
        if known / len(cis) >= min_known:
            eligible.add(uid)
    print(f"  {len(eligible):,} eligible (>={min_ci} CIs, "
          f">={min_known*100:.0f}% Overture)")

    # ── venue coordinate map + region index ─────────────────────────
    venue_coords: dict[int, tuple[float, float]] = {}
    for uid in eligible:
        for c in user_checkins[uid]:
            if c.location_id not in venue_coords:
                venue_coords[c.location_id] = (c.latitude, c.longitude)

    region_venues: dict[str, set[int]] = {}
    for loc_id, (lat, lon) in venue_coords.items():
        region_venues.setdefault(_region_key(lat, lon), set()).add(loc_id)

    # ── filter: enough regional negatives ───────────────────────────
    def _neg_pool(uid: int) -> list[int]:
        last = user_checkins[uid][-1]
        rlat = round(last.latitude, 1)
        rlon = round(last.longitude, 1)
        pool: set[int] = set()
        for dl in (-0.1, 0.0, 0.1):
            for dn in (-0.1, 0.0, 0.1):
                rk = f"{round(rlat + dl, 1)},{round(rlon + dn, 1)}"
                pool.update(region_venues.get(rk, set()))
        pool.discard(last.location_id)
        return list(pool)

    nextloc_eligible: list[int] = []
    for uid in eligible:
        if len(_neg_pool(uid)) >= 19:
            nextloc_eligible.append(uid)
    print(f"  {len(nextloc_eligible):,} users with >=19 regional candidates")

    # ── split users ─────────────────────────────────────────────────
    random.seed(42)
    random.shuffle(nextloc_eligible)

    n_train = nl["n_train_users"]
    n_test = nl["n_test_users"]
    train_uids = nextloc_eligible[:n_train]
    test_uids = nextloc_eligible[n_train : n_train + n_test]
    print(f"  Split: {len(train_uids)} train, {len(test_uids)} test users")

    # ── popularity (from train users only) ──────────────────────────
    venue_pop: Counter[int] = Counter()
    cell_pop: Counter[str] = Counter()
    for uid in train_uids:
        for c in user_checkins[uid]:
            venue_pop[c.location_id] += 1
            cell_pop[f"{round(c.latitude, 3)},{round(c.longitude, 3)}"] += 1

    # ── generate pointwise test cases ───────────────────────────────
    def _generate(uids: list[int], seed_offset: int) -> list[NextLocTestCase]:
        rng = random.Random(42 + seed_offset)
        cases: list[NextLocTestCase] = []
        for uid in uids:
            cis = user_checkins[uid]
            last_ci = cis[-1]
            history = cis[:-1]
            profile = build_user_profile(uid, history)
            loc_counts = Counter(c.location_id for c in history)

            def _make(loc_id: int, lat: float, lon: float,
                      label: int) -> NextLocTestCase:
                return NextLocTestCase(
                    user=profile,
                    candidate_location_id=loc_id,
                    candidate_lat=lat,
                    candidate_lon=lon,
                    label=label,
                    user_visited_candidate=loc_id in loc_counts,
                    user_visits_to_candidate=loc_counts.get(loc_id, 0),
                    distance_to_user_centroid_km=round(
                        haversine_km(profile.centroid_lat,
                                     profile.centroid_lon, lat, lon), 2),
                    candidate_popularity=venue_pop.get(loc_id, 0),
                )

            cases.append(_make(last_ci.location_id, last_ci.latitude,
                               last_ci.longitude, label=1))

            pool = _neg_pool(uid)
            neg_ids = rng.sample(pool, 19)
            for nid in neg_ids:
                nlat, nlon = venue_coords[nid]
                cases.append(_make(nid, nlat, nlon, label=0))
        return cases

    out_dir = ROOT / nl["data_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating train cases …")
    train_cases = _generate(train_uids, seed_offset=1)
    save_nextloc_cases(train_cases, out_dir / "train_cases.json")
    print(f"  {len(train_cases)} cases ({len(train_uids)} users x "
          f"{nl['n_candidates']})")

    print("Generating test cases …")
    test_cases = _generate(test_uids, seed_offset=2)
    save_nextloc_cases(test_cases, out_dir / "test_cases.json")
    print(f"  {len(test_cases)} cases ({len(test_uids)} users x "
          f"{nl['n_candidates']})")

    (out_dir / "venue_popularity.json").write_text(
        json.dumps({str(k): v for k, v in venue_pop.items()}))
    (out_dir / "cell_popularity.json").write_text(
        json.dumps(dict(cell_pop)))
    print(f"\nvenue_popularity: {len(venue_pop)} venues")
    print(f"cell_popularity:  {len(cell_pop)} cells")
    print(f"\nOutput → {out_dir}/")


if __name__ == "__main__":
    main()
