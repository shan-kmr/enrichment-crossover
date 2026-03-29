"""Enriched Gowalla friendship prediction prompt builder (G0-G4).

Adds venue category information from Overture Maps enrichment.
G0 provides basic user activity stats (aligned with ML G0 baseline).
G1-G4 add semantic venue labels.
"""

from __future__ import annotations

from src.gowalla.data_loader import FriendshipTestCase, UserProfile
from src.gowalla.enrichment import (
    get_category, format_category,
    user_category_profile, pair_category_features,
)

MAX_HISTORY = 15

SYSTEM = (
    "You are an expert at predicting social relationships from location data. "
    "Given information about two users' check-in patterns and venue preferences, "
    "predict whether they are friends. Answer ONLY 'Yes' or 'No'."
)


def _user_desc(profile: UserProfile, tag: str, granularity: str, cat_map: dict) -> str:
    parts: list[str] = []

    if granularity in ("G1", "G2", "G3", "G4"):
        parts.append(f"{tag} is primarily active in region {profile.primary_region}.")
        cat_prof = user_category_profile(profile, cat_map)
        top_str = ", ".join(
            f"{name} ({pct:.0f}%)"
            for name, _, pct in cat_prof["top_categories"][:5]
        )
        if top_str:
            parts.append(f"{tag} frequently visits: {top_str}.")

    if granularity in ("G2", "G3", "G4"):
        cat_prof = user_category_profile(profile, cat_map)
        parts.append(
            f"{tag} statistics: {profile.total_checkins} total check-ins, "
            f"{profile.unique_locations} unique locations, "
            f"{cat_prof['unique_categories']} venue categories, "
            f"geographic spread {profile.geo_spread_km:.1f} km, "
            f"active for {profile.active_days} days."
        )

    if granularity == "G3":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [
            f"  {i}. {format_category(get_category(c.location_id, cat_map))} "
            f"at ({c.latitude:.4f}, {c.longitude:.4f})"
            for i, c in enumerate(recent, 1)
        ]
        parts.append(f"{tag} recent {len(recent)} check-ins:\n" + "\n".join(lines))

    if granularity == "G4":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [
            f"  {i}. [{c.timestamp}] "
            f"{format_category(get_category(c.location_id, cat_map))} "
            f"at ({c.latitude:.4f}, {c.longitude:.4f})"
            for i, c in enumerate(recent, 1)
        ]
        parts.append(f"{tag} recent {len(recent)} check-ins:\n" + "\n".join(lines))

    if granularity == "G0":
        parts.append(
            f"{tag}: {profile.total_checkins} total check-ins "
            f"across {profile.unique_locations} unique locations."
        )

    return "\n".join(parts)


def _pair_context(tc: FriendshipTestCase, granularity: str, cat_map: dict) -> str:
    if granularity not in ("G2", "G3", "G4"):
        return ""

    cat_feats = pair_category_features(tc.user_a, tc.user_b, cat_map)

    lines = [
        f"Shared locations: {tc.shared_locations}",
        f"Location Jaccard similarity: {tc.jaccard_locations:.4f}",
        f"Shared venue categories: {cat_feats['shared_categories']}",
        f"Category similarity (Jaccard): {cat_feats['category_jaccard']:.4f}",
        f"Distance between centroids: {tc.centroid_distance_km:.1f} km",
        f"Same primary region: {'Yes' if tc.same_region else 'No'}",
        f"Same top venue category: {'Yes' if cat_feats['same_top_category'] else 'No'}",
    ]
    if granularity == "G4":
        lines.append(
            f"Temporal co-occurrences (same location within 1h): "
            f"{tc.temporal_co_occurrences}"
        )
    return "Pair statistics:\n" + "\n".join(f"  - {l}" for l in lines)


def build_prompt(
    tc: FriendshipTestCase, granularity: str, cat_map: dict
) -> list[dict[str, str]]:
    user_a_desc = _user_desc(tc.user_a, "User A", granularity, cat_map)
    user_b_desc = _user_desc(tc.user_b, "User B", granularity, cat_map)
    pair_ctx = _pair_context(tc, granularity, cat_map)

    body = f"{user_a_desc}\n\n{user_b_desc}"
    if pair_ctx:
        body += f"\n\n{pair_ctx}"
    body += "\n\nAre these two users friends? Answer ONLY 'Yes' or 'No'."

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body},
    ]
