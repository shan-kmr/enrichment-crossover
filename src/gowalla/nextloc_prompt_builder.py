"""Next-location prediction prompts for all 3 tiers.

Tier parameter controls how candidates and history are described:
  latlng   – rounded coordinates only, no venue IDs
  venue_id – Location {id} at (lat, lon)
  enriched – Category (Location {id}) at (lat, lon)
"""

from __future__ import annotations

from collections import Counter

from src.gowalla.data_loader import (
    NextLocTestCase, UserProfile, _region_key,
)
from src.gowalla.latlng_features import _geocell
from src.gowalla.enrichment import (
    get_category, format_category, user_category_profile,
)

MAX_HISTORY = 15
RESOLUTION = 3

SYSTEM = (
    "You are an expert at predicting location visits from check-in data. "
    "Given a user's check-in history and a candidate venue, predict whether "
    "the user will visit this venue next. Answer ONLY 'Yes' or 'No'."
)


def _candidate_desc(tc: NextLocTestCase, tier: str,
                    cat_map: dict | None) -> str:
    lat, lon = tc.candidate_lat, tc.candidate_lon
    if tier == "latlng":
        return f"Candidate venue at ({lat:.3f}, {lon:.3f})."
    if tier == "venue_id":
        return (f"Candidate: Location {tc.candidate_location_id} "
                f"at ({lat:.4f}, {lon:.4f}).")
    cat = (format_category(get_category(tc.candidate_location_id, cat_map))
           if cat_map else "Unknown Venue")
    return (f"Candidate: {cat} (Location {tc.candidate_location_id}) "
            f"at ({lat:.4f}, {lon:.4f}).")


def _user_desc(user: UserProfile, granularity: str, tier: str,
               cat_map: dict | None) -> str:
    parts: list[str] = []

    if granularity == "G0":
        if tier == "latlng":
            cells = len({_geocell(c.latitude, c.longitude, RESOLUTION)
                         for c in user.checkins})
            parts.append(f"User: {user.total_checkins} total check-ins "
                         f"across {cells} unique areas.")
        else:
            parts.append(f"User: {user.total_checkins} total check-ins "
                         f"across {user.unique_locations} unique locations.")
        return "\n".join(parts)

    # G1+
    if granularity in ("G1", "G2", "G3", "G4"):
        parts.append(f"User is primarily active in region "
                     f"{user.primary_region}.")
        if tier == "enriched" and cat_map:
            cp = user_category_profile(user, cat_map)
            top = ", ".join(f"{n} ({p:.0f}%)"
                            for n, _, p in cp["top_categories"][:5])
            if top:
                parts.append(f"User frequently visits: {top}.")

    # G2+
    if granularity in ("G2", "G3", "G4"):
        if tier == "latlng":
            cells = len({_geocell(c.latitude, c.longitude, RESOLUTION)
                         for c in user.checkins})
            parts.append(
                f"User statistics: {user.total_checkins} total check-ins, "
                f"{cells} unique areas, "
                f"geographic spread {user.geo_spread_km:.1f} km, "
                f"active for {user.active_days} days.")
        else:
            parts.append(
                f"User statistics: {user.total_checkins} total check-ins, "
                f"{user.unique_locations} unique locations, "
                f"geographic spread {user.geo_spread_km:.1f} km, "
                f"active for {user.active_days} days.")

    # G3 trajectory
    if granularity == "G3":
        recent = user.checkins[-MAX_HISTORY:]
        if tier == "latlng":
            lines = [f"  {i}. at ({c.latitude:.3f}, {c.longitude:.3f})"
                     for i, c in enumerate(recent, 1)]
        elif tier == "venue_id":
            lines = [f"  {i}. Location {c.location_id} "
                     f"at ({c.latitude:.4f}, {c.longitude:.4f})"
                     for i, c in enumerate(recent, 1)]
        else:
            lines = [
                f"  {i}. {format_category(get_category(c.location_id, cat_map))} "
                f"at ({c.latitude:.4f}, {c.longitude:.4f})"
                for i, c in enumerate(recent, 1)]
        parts.append(f"User's recent {len(recent)} check-ins:\n"
                     + "\n".join(lines))

    # G4 trajectory with timestamps
    if granularity == "G4":
        recent = user.checkins[-MAX_HISTORY:]
        if tier == "latlng":
            lines = [f"  {i}. [{c.timestamp}] "
                     f"at ({c.latitude:.3f}, {c.longitude:.3f})"
                     for i, c in enumerate(recent, 1)]
        elif tier == "venue_id":
            lines = [f"  {i}. [{c.timestamp}] Location {c.location_id} "
                     f"at ({c.latitude:.4f}, {c.longitude:.4f})"
                     for i, c in enumerate(recent, 1)]
        else:
            lines = [
                f"  {i}. [{c.timestamp}] "
                f"{format_category(get_category(c.location_id, cat_map))} "
                f"at ({c.latitude:.4f}, {c.longitude:.4f})"
                for i, c in enumerate(recent, 1)]
        parts.append(f"User's recent {len(recent)} check-ins:\n"
                     + "\n".join(lines))

    return "\n".join(parts)


def _visit_context(tc: NextLocTestCase, granularity: str, tier: str,
                   cat_map: dict | None) -> str:
    if granularity == "G0":
        return ""

    lines: list[str] = []

    # G1+: regional context
    if granularity in ("G1", "G2", "G3", "G4"):
        cand_region = _region_key(tc.candidate_lat, tc.candidate_lon)
        where = "in" if cand_region == tc.user.primary_region else "outside"
        lines.append(f"Candidate is {where} user's primary region.")
        lines.append(f"Distance from user's center of activity: "
                     f"{tc.distance_to_user_centroid_km:.1f} km.")

    # G2+: visit history
    if granularity in ("G2", "G3", "G4"):
        if tier == "latlng":
            cand_cell = _geocell(tc.candidate_lat, tc.candidate_lon,
                                 RESOLUTION)
            cell_counts = Counter(
                _geocell(c.latitude, c.longitude, RESOLUTION)
                for c in tc.user.checkins)
            v = cell_counts.get(cand_cell, 0)
            if v:
                lines.append(
                    f"User has visited this area {v} time(s) before.")
            else:
                lines.append("User has never visited this area.")
        else:
            if tc.user_visited_candidate:
                lines.append(
                    f"User has visited this venue "
                    f"{tc.user_visits_to_candidate} time(s) before.")
            else:
                lines.append("User has never visited this venue.")

        if tier == "enriched" and cat_map:
            cand_cat = get_category(tc.candidate_location_id, cat_map)
            if cand_cat != "unknown":
                cc = Counter(
                    get_category(c.location_id, cat_map)
                    for c in tc.user.checkins)
                cc.pop("unknown", None)
                total = sum(cc.values()) or 1
                cv = cc.get(cand_cat, 0)
                lines.append(
                    f"User has visited {format_category(cand_cat)} venues "
                    f"{cv} time(s) "
                    f"({cv / total * 100:.0f}% of check-ins).")

    return "\n".join(lines)


def build_prompt(
    tc: NextLocTestCase,
    granularity: str,
    tier: str,
    cat_map: dict | None = None,
) -> list[dict[str, str]]:
    user_desc = _user_desc(tc.user, granularity, tier, cat_map)
    cand_desc = _candidate_desc(tc, tier, cat_map)
    ctx = _visit_context(tc, granularity, tier, cat_map)

    body = f"{user_desc}\n\n{cand_desc}"
    if ctx:
        body += f"\n\n{ctx}"
    body += ("\n\nWill this user visit this venue next? "
             "Answer ONLY 'Yes' or 'No'.")

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": body},
    ]
