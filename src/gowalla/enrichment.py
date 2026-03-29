"""Shared enrichment utilities for Gowalla venue category enrichment via Overture Maps."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from src.gowalla.data_loader import UserProfile, FriendshipTestCase


def load_category_map(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text())


def get_category(location_id: int, cat_map: dict) -> str:
    entry = cat_map.get(str(location_id), {})
    return entry.get("category", "unknown")


def format_category(cat: str) -> str:
    if cat == "unknown":
        return "Unknown Venue"
    return cat.replace("_", " ").title()


def _entropy(counts: Counter) -> float:
    """Shannon entropy of a category frequency distribution (bits)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)


def _herfindahl(counts: Counter) -> float:
    """Herfindahl-Hirschman index: sum of squared frequency shares."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return sum((c / total) ** 2 for c in counts.values())


def _js_divergence(p_counts: Counter, q_counts: Counter) -> float:
    """Jensen-Shannon divergence between two category distributions."""
    all_cats = set(p_counts) | set(q_counts)
    if not all_cats:
        return 0.0
    total_p = sum(p_counts.values()) or 1
    total_q = sum(q_counts.values()) or 1
    p = {c: p_counts.get(c, 0) / total_p for c in all_cats}
    q = {c: q_counts.get(c, 0) / total_q for c in all_cats}
    m = {c: (p[c] + q[c]) / 2 for c in all_cats}

    def _kl(dist, ref):
        return sum(
            dist[c] * math.log2(dist[c] / ref[c])
            for c in all_cats
            if dist[c] > 0 and ref[c] > 0
        )

    return (_kl(p, m) + _kl(q, m)) / 2


def _cosine(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    mag_a = sum(x ** 2 for x in vec_a) ** 0.5
    mag_b = sum(x ** 2 for x in vec_b) ** 0.5
    return dot / (mag_a * mag_b) if mag_a * mag_b else 0.0


def user_category_profile(profile: UserProfile, cat_map: dict) -> dict:
    cats = [get_category(c.location_id, cat_map) for c in profile.checkins]
    known = [c for c in cats if c != "unknown"]
    counter = Counter(known)
    total = len(known) or 1
    top = [
        (format_category(cat), count, count / total * 100)
        for cat, count in counter.most_common(5)
    ]
    return {
        "category_counts": counter,
        "top_categories": top,
        "unique_categories": len(counter),
        "known_ratio": len(known) / max(len(cats), 1),
        "entropy": _entropy(counter),
        "concentration": _herfindahl(counter),
    }


def pair_category_features(a: UserProfile, b: UserProfile, cat_map: dict) -> dict:
    cats_a = Counter(get_category(c.location_id, cat_map) for c in a.checkins)
    cats_b = Counter(get_category(c.location_id, cat_map) for c in b.checkins)
    cats_a.pop("unknown", None)
    cats_b.pop("unknown", None)

    set_a, set_b = set(cats_a), set(cats_b)
    shared = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = shared / union if union else 0.0

    all_cats = set_a | set_b
    if all_cats:
        sorted_cats = sorted(all_cats)
        vec_a = [cats_a.get(c, 0) for c in sorted_cats]
        vec_b = [cats_b.get(c, 0) for c in sorted_cats]
        cosine = _cosine(vec_a, vec_b)
    else:
        cosine = 0.0

    top_a = max(cats_a, key=cats_a.get) if cats_a else ""
    top_b = max(cats_b, key=cats_b.get) if cats_b else ""

    # Category features on NON-shared locations only:
    # captures behavioral affinity beyond geographic co-location
    shared_locs = {c.location_id for c in a.checkins} & {c.location_id for c in b.checkins}
    ns_cats_a = Counter(
        get_category(c.location_id, cat_map)
        for c in a.checkins if c.location_id not in shared_locs
    )
    ns_cats_b = Counter(
        get_category(c.location_id, cat_map)
        for c in b.checkins if c.location_id not in shared_locs
    )
    ns_cats_a.pop("unknown", None)
    ns_cats_b.pop("unknown", None)

    ns_set_a, ns_set_b = set(ns_cats_a), set(ns_cats_b)
    ns_shared = len(ns_set_a & ns_set_b)
    ns_union = len(ns_set_a | ns_set_b)
    ns_jaccard = ns_shared / ns_union if ns_union else 0.0

    ns_all = ns_set_a | ns_set_b
    if ns_all:
        ns_sorted = sorted(ns_all)
        ns_cosine = _cosine(
            [ns_cats_a.get(c, 0) for c in ns_sorted],
            [ns_cats_b.get(c, 0) for c in ns_sorted],
        )
    else:
        ns_cosine = 0.0

    js_div = _js_divergence(cats_a, cats_b)

    top3_a = {c for c, _ in cats_a.most_common(3)}
    top3_b = {c for c, _ in cats_b.most_common(3)}
    top3_overlap = len(top3_a & top3_b) if (top3_a and top3_b) else 0

    return {
        "shared_categories": shared,
        "category_jaccard": round(jaccard, 4),
        "category_cosine": round(cosine, 4),
        "unique_categories_a": len(set_a),
        "unique_categories_b": len(set_b),
        "same_top_category": 1 if (top_a and top_a == top_b) else 0,
        "non_shared_category_jaccard": round(ns_jaccard, 4),
        "non_shared_category_cosine": round(ns_cosine, 4),
        "category_js_divergence": round(js_div, 4),
        "top3_category_overlap": top3_overlap,
    }


def compute_enriched_handcrafted(
    tc: FriendshipTestCase, granularity: str, cat_map: dict
) -> list[float]:
    """Handcrafted features with enhanced category enrichment."""
    a, b = tc.user_a, tc.user_b
    cat_feats = pair_category_features(a, b, cat_map)
    cat_prof_a = user_category_profile(a, cat_map)
    cat_prof_b = user_category_profile(b, cat_map)

    # G0: 3 features
    feats = [
        float(a.total_checkins),
        float(b.total_checkins),
        float(a.total_checkins + b.total_checkins),
    ]

    if granularity in ("G1", "G2", "G3", "G4"):
        # +7 = 10: region info + user-level category profile
        feats.extend([
            float(tc.same_region),
            tc.centroid_distance_km,
            float(a.primary_region == b.primary_region),
            float(cat_feats["same_top_category"]),
            cat_prof_a["entropy"],
            cat_prof_b["entropy"],
            float(cat_feats["top3_category_overlap"]),
        ])

    if granularity in ("G2", "G3", "G4"):
        # +12 = 22: overlap stats + distributional category features
        feats.extend([
            float(a.unique_locations),
            float(b.unique_locations),
            float(tc.shared_locations),
            tc.jaccard_locations,
            float(cat_feats["shared_categories"]),
            cat_feats["category_jaccard"],
            float(cat_prof_a["unique_categories"]),
            float(cat_prof_b["unique_categories"]),
            cat_feats["non_shared_category_jaccard"],
            cat_feats["category_js_divergence"],
            cat_prof_a["concentration"],
            cat_prof_b["concentration"],
        ])

    if granularity in ("G3", "G4"):
        # +4 = 26
        feats.extend([
            a.geo_spread_km,
            b.geo_spread_km,
            float(a.active_days),
            float(b.active_days),
        ])

    if granularity == "G4":
        # +5 = 31: temporal + detailed distributional similarity
        feats.extend([
            float(tc.temporal_co_occurrences),
            tc.centroid_distance_km / max(a.geo_spread_km + b.geo_spread_km, 0.01),
            tc.jaccard_locations * float(tc.temporal_co_occurrences + 1),
            cat_feats["category_cosine"],
            cat_feats["non_shared_category_cosine"],
        ])

    return feats
