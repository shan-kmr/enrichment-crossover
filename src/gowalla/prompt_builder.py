"""Gowalla friendship prediction prompt builder (G0-G4)."""

from __future__ import annotations

from src.gowalla.data_loader import FriendshipTestCase, UserProfile

MAX_HISTORY = 15

SYSTEM = (
    "You are an expert at predicting social relationships from location data. "
    "Given information about two users' check-in patterns, predict whether "
    "they are friends. Answer ONLY 'Yes' or 'No'."
)


def _user_desc(profile: UserProfile, tag: str, granularity: str) -> str:
    parts: list[str] = []

    if granularity in ("G1", "G2", "G3", "G4"):
        parts.append(f"{tag} is primarily active in region {profile.primary_region}.")

    if granularity in ("G2", "G3", "G4"):
        parts.append(
            f"{tag} statistics: {profile.total_checkins} total check-ins, "
            f"{profile.unique_locations} unique locations, "
            f"geographic spread {profile.geo_spread_km:.1f} km, "
            f"active for {profile.active_days} days."
        )

    if granularity == "G3":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [
            f"  {i}. Location {c.location_id} at ({c.latitude:.4f}, {c.longitude:.4f})"
            for i, c in enumerate(recent, 1)
        ]
        parts.append(f"{tag} recent {len(recent)} check-ins:\n" + "\n".join(lines))

    if granularity == "G4":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [
            f"  {i}. [{c.timestamp}] Location {c.location_id} "
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


def _pair_context(tc: FriendshipTestCase, granularity: str) -> str:
    if granularity not in ("G2", "G3", "G4"):
        return ""
    lines = [
        f"Shared locations: {tc.shared_locations}",
        f"Location Jaccard similarity: {tc.jaccard_locations:.4f}",
        f"Distance between centroids: {tc.centroid_distance_km:.1f} km",
        f"Same primary region: {'Yes' if tc.same_region else 'No'}",
    ]
    if granularity == "G4":
        lines.append(f"Temporal co-occurrences (same location within 1h): {tc.temporal_co_occurrences}")
    return "Pair statistics:\n" + "\n".join(f"  - {l}" for l in lines)


def build_prompt(tc: FriendshipTestCase, granularity: str) -> list[dict[str, str]]:
    user_a_desc = _user_desc(tc.user_a, "User A", granularity)
    user_b_desc = _user_desc(tc.user_b, "User B", granularity)
    pair_ctx = _pair_context(tc, granularity)

    body = f"{user_a_desc}\n\n{user_b_desc}"
    if pair_ctx:
        body += f"\n\n{pair_ctx}"
    body += "\n\nAre these two users friends? Answer ONLY 'Yes' or 'No'."

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body},
    ]
