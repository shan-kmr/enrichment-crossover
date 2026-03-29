"""Gowalla data loader: parse SNAP check-in and edge files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CheckIn:
    user_id: int
    timestamp: str
    latitude: float
    longitude: float
    location_id: int


@dataclass
class UserProfile:
    user_id: int
    checkins: list[CheckIn] = field(default_factory=list)
    unique_locations: int = 0
    total_checkins: int = 0
    centroid_lat: float = 0.0
    centroid_lon: float = 0.0
    geo_spread_km: float = 0.0
    primary_region: str = ""
    active_days: int = 0


@dataclass
class FriendshipTestCase:
    user_a: UserProfile
    user_b: UserProfile
    label: int  # 1 = friends, 0 = not friends

    # precomputed pair features
    shared_locations: int = 0
    jaccard_locations: float = 0.0
    centroid_distance_km: float = 0.0
    same_region: int = 0
    temporal_co_occurrences: int = 0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _region_key(lat: float, lon: float) -> str:
    return f"{round(lat, 1)},{round(lon, 1)}"


def load_edges(path: Path) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) == 2:
                a, b = int(parts[0]), int(parts[1])
                edges.add((min(a, b), max(a, b)))
    return edges


def load_checkins(path: Path) -> dict[int, list[CheckIn]]:
    user_checkins: dict[int, list[CheckIn]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 5:
                continue
            uid = int(parts[0])
            ci = CheckIn(
                user_id=uid,
                timestamp=parts[1],
                latitude=float(parts[2]),
                longitude=float(parts[3]),
                location_id=int(parts[4]),
            )
            user_checkins.setdefault(uid, []).append(ci)
    for uid in user_checkins:
        user_checkins[uid].sort(key=lambda c: c.timestamp)
    return user_checkins


def build_user_profile(user_id: int, checkins: list[CheckIn]) -> UserProfile:
    locs = {c.location_id for c in checkins}
    lats = [c.latitude for c in checkins]
    lons = [c.longitude for c in checkins]
    clat = sum(lats) / len(lats)
    clon = sum(lons) / len(lons)

    dists = [haversine_km(clat, clon, la, lo) for la, lo in zip(lats, lons)]
    spread = (sum(d ** 2 for d in dists) / len(dists)) ** 0.5

    region_counts: dict[str, int] = {}
    for c in checkins:
        rk = _region_key(c.latitude, c.longitude)
        region_counts[rk] = region_counts.get(rk, 0) + 1
    primary = max(region_counts, key=region_counts.get) if region_counts else ""

    days = {c.timestamp[:10] for c in checkins}

    return UserProfile(
        user_id=user_id,
        checkins=checkins,
        unique_locations=len(locs),
        total_checkins=len(checkins),
        centroid_lat=clat,
        centroid_lon=clon,
        geo_spread_km=round(spread, 2),
        primary_region=primary,
        active_days=len(days),
    )


def compute_pair_features(a: UserProfile, b: UserProfile) -> dict:
    locs_a = {c.location_id for c in a.checkins}
    locs_b = {c.location_id for c in b.checkins}
    shared = len(locs_a & locs_b)
    union = len(locs_a | locs_b)
    jaccard = shared / union if union else 0.0
    dist = haversine_km(a.centroid_lat, a.centroid_lon, b.centroid_lat, b.centroid_lon)
    same_region = 1 if a.primary_region == b.primary_region else 0

    # temporal co-occurrences: same location within 1 hour
    a_by_loc: dict[int, list[str]] = {}
    for c in a.checkins:
        a_by_loc.setdefault(c.location_id, []).append(c.timestamp)
    co_occ = 0
    for c in b.checkins:
        if c.location_id in a_by_loc:
            for ts_a in a_by_loc[c.location_id]:
                if abs(len(ts_a) - len(c.timestamp)) < 2:
                    if ts_a[:13] == c.timestamp[:13]:
                        co_occ += 1
                        break

    return {
        "shared_locations": shared,
        "jaccard_locations": round(jaccard, 4),
        "centroid_distance_km": round(dist, 2),
        "same_region": same_region,
        "temporal_co_occurrences": co_occ,
    }


def save_test_cases(cases: list[FriendshipTestCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for tc in cases:
        d = {
            "user_a": asdict(tc.user_a),
            "user_b": asdict(tc.user_b),
            "label": tc.label,
            "shared_locations": tc.shared_locations,
            "jaccard_locations": tc.jaccard_locations,
            "centroid_distance_km": tc.centroid_distance_km,
            "same_region": tc.same_region,
            "temporal_co_occurrences": tc.temporal_co_occurrences,
        }
        data.append(d)
    path.write_text(json.dumps(data))


def load_test_cases(path: Path) -> list[FriendshipTestCase]:
    raw = json.loads(path.read_text())
    cases = []
    for d in raw:
        ua = UserProfile(**{k: v for k, v in d["user_a"].items() if k != "checkins"})
        ua.checkins = [CheckIn(**c) for c in d["user_a"]["checkins"]]
        ub = UserProfile(**{k: v for k, v in d["user_b"].items() if k != "checkins"})
        ub.checkins = [CheckIn(**c) for c in d["user_b"]["checkins"]]
        tc = FriendshipTestCase(
            user_a=ua, user_b=ub, label=d["label"],
            shared_locations=d.get("shared_locations", 0),
            jaccard_locations=d.get("jaccard_locations", 0.0),
            centroid_distance_km=d.get("centroid_distance_km", 0.0),
            same_region=d.get("same_region", 0),
            temporal_co_occurrences=d.get("temporal_co_occurrences", 0),
        )
        cases.append(tc)
    return cases


# ---------------------------------------------------------------------------
# Next-Location Prediction
# ---------------------------------------------------------------------------

@dataclass
class NextLocTestCase:
    user: UserProfile
    candidate_location_id: int
    candidate_lat: float
    candidate_lon: float
    label: int  # 1 = ground-truth next venue, 0 = negative
    user_visited_candidate: bool = False
    user_visits_to_candidate: int = 0
    distance_to_user_centroid_km: float = 0.0
    candidate_popularity: int = 0


def save_nextloc_cases(cases: list[NextLocTestCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for tc in cases:
        data.append({
            "user": asdict(tc.user),
            "candidate_location_id": tc.candidate_location_id,
            "candidate_lat": tc.candidate_lat,
            "candidate_lon": tc.candidate_lon,
            "label": tc.label,
            "user_visited_candidate": tc.user_visited_candidate,
            "user_visits_to_candidate": tc.user_visits_to_candidate,
            "distance_to_user_centroid_km": tc.distance_to_user_centroid_km,
            "candidate_popularity": tc.candidate_popularity,
        })
    path.write_text(json.dumps(data))


def load_nextloc_cases(path: Path) -> list[NextLocTestCase]:
    raw = json.loads(path.read_text())
    cases = []
    for d in raw:
        u = UserProfile(**{k: v for k, v in d["user"].items() if k != "checkins"})
        u.checkins = [CheckIn(**c) for c in d["user"]["checkins"]]
        cases.append(NextLocTestCase(
            user=u,
            candidate_location_id=d["candidate_location_id"],
            candidate_lat=d["candidate_lat"],
            candidate_lon=d["candidate_lon"],
            label=d["label"],
            user_visited_candidate=d.get("user_visited_candidate", False),
            user_visits_to_candidate=d.get("user_visits_to_candidate", 0),
            distance_to_user_centroid_km=d.get("distance_to_user_centroid_km", 0.0),
            candidate_popularity=d.get("candidate_popularity", 0),
        ))
    return cases
