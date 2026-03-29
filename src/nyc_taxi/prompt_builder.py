"""Build text prompts for NYC taxi tasks across tiers and granularities.

Mirrors src/gowalla/nextloc_prompt_builder.py structure.
"""

from __future__ import annotations

from src.nyc_taxi.features import _haversine_km


def _zone_desc(zone_id: int, agg: dict, zone_cats: dict | None, tier: str) -> str:
    """Human-readable zone description."""
    borough = agg.get("zone_borough", {}).get(str(zone_id), "Unknown")
    parts = [f"Zone {zone_id} ({borough})"]
    if tier == "enriched" and zone_cats:
        info = zone_cats.get(str(zone_id), {})
        primary = info.get("primary_category", "unknown")
        if primary != "unknown":
            cats = info.get("all_categories", [])[:5]
            parts.append(f"Primary category: {primary.replace('_', ' ').title()}")
            if len(cats) > 1:
                cat_str = ", ".join(c.replace("_", " ").title() for c in cats)
                parts.append(f"Nearby POIs: {cat_str}")
    return ". ".join(parts)


def build_ranking_prompt(
    case: dict,
    granularity: str,
    tier: str,
    agg: dict,
    centroids: dict,
    zone_cats: dict | None = None,
) -> str:
    """Build a prompt for the ranking task (is this candidate the next dropoff?)."""
    origin = case["origin_zone"]
    cand = case["candidate_zone"]
    origin_s, cand_s = str(origin), str(cand)

    lines: list[str] = []

    # Origin description
    lines.append(f"Origin: {_zone_desc(origin, agg, zone_cats, tier)}")

    # G0: aggregate stats
    total_trips = agg.get("zone_pu_count", {}).get(origin_s, 0)
    unique_do = agg.get("origin_unique_do", {}).get(origin_s, 0)
    cand_pop = agg.get("zone_do_count", {}).get(cand_s, 0)
    lines.append(f"Origin has {total_trips:,} total trips to {unique_do} distinct dropoff zones.")
    lines.append(f"Candidate: {_zone_desc(cand, agg, zone_cats, tier)}")
    lines.append(f"Candidate dropoff popularity: {cand_pop:,} trips.")

    # G1: region + distance
    if granularity in ("G1", "G2", "G3", "G4"):
        o_borough = agg.get("zone_borough", {}).get(origin_s, "")
        c_borough = agg.get("zone_borough", {}).get(cand_s, "")
        same = "Yes" if (o_borough == c_borough and o_borough) else "No"
        o_lat, o_lon = centroids.get(origin, (0, 0))
        c_lat, c_lon = centroids.get(cand, (0, 0))
        dist = _haversine_km(o_lat, o_lon, c_lat, c_lon) if o_lat else 0
        lines.append(f"Same borough: {same}. Distance: {dist:.1f} km.")

    # G2: history
    if granularity in ("G2", "G3", "G4"):
        geo_spread = agg.get("origin_geo_spread", {}).get(origin_s, 0)
        active_days = agg.get("origin_active_days", {}).get(origin_s, 0)
        od_key = f"{origin}_{cand}"
        trips_od = agg.get("od_count", {}).get(od_key, 0)
        lines.append(
            f"Origin geo-spread: {geo_spread:.1f} km. Active days: {active_days}. "
            f"Trips from origin to candidate: {trips_od}."
        )

    # G3: recent trajectory
    if granularity in ("G3", "G4"):
        recent5 = agg.get("recent5", {}).get(origin_s, [])
        if recent5:
            dests = [str(r["DOLocationID"]) for r in recent5]
            lines.append(f"Recent 5 dropoff zones from origin: {', '.join(dests)}.")

    # G4: temporal
    if granularity == "G4":
        mean_h = agg.get("mean_do_hour", {}).get(origin_s, 12)
        std_h = agg.get("std_do_hour", {}).get(origin_s, 0)
        lines.append(f"Mean dropoff hour: {mean_h:.1f} (std: {std_h:.1f}).")

    lines.append("Question: Is this candidate the next dropoff zone? Answer Yes or No.")
    return "\n".join(lines)


def build_duration_prompt(
    case: dict,
    granularity: str,
    tier: str,
    agg: dict,
    centroids: dict,
    zone_cats: dict | None = None,
) -> str:
    """Build a prompt for the duration prediction task."""
    origin = case["PULocationID"]
    dest = case["DOLocationID"]
    origin_s, dest_s = str(origin), str(dest)

    lines: list[str] = []

    # Origin and destination
    lines.append(f"Origin: {_zone_desc(origin, agg, zone_cats, tier)}")
    lines.append(f"Destination: {_zone_desc(dest, agg, zone_cats, tier)}")

    # G0: aggregate durations
    mean_dur_o = agg.get("zone_mean_dur_pu", {}).get(origin_s, 0)
    mean_dur_d = agg.get("zone_mean_dur_do", {}).get(dest_s, 0)
    total_o = agg.get("zone_pu_count", {}).get(origin_s, 0)
    total_d = agg.get("zone_do_count", {}).get(dest_s, 0)
    lines.append(
        f"Origin has {total_o:,} trips (mean duration: {mean_dur_o:.1f} min). "
        f"Destination has {total_d:,} arrivals (mean duration: {mean_dur_d:.1f} min)."
    )

    # G1: region + distance
    if granularity in ("G1", "G2", "G3", "G4"):
        o_borough = agg.get("zone_borough", {}).get(origin_s, "")
        d_borough = agg.get("zone_borough", {}).get(dest_s, "")
        same = "Yes" if (o_borough == d_borough and o_borough) else "No"
        o_lat, o_lon = centroids.get(origin, (0, 0))
        d_lat, d_lon = centroids.get(dest, (0, 0))
        dist = _haversine_km(o_lat, o_lon, d_lat, d_lon) if o_lat else 0
        lines.append(f"Same borough: {same}. Distance: {dist:.1f} km.")

    # G2: OD history
    if granularity in ("G2", "G3", "G4"):
        od_key = f"{origin}_{dest}"
        trips_od = agg.get("od_count", {}).get(od_key, 0)
        mean_dur_od = agg.get("od_mean_dur", {}).get(od_key, 0)
        trip_dist_mi = case.get("trip_distance", 0)
        trip_dist_km = trip_dist_mi * 1.60934
        lines.append(
            f"Trips on this OD pair: {trips_od} (mean duration: {mean_dur_od:.1f} min). "
            f"Trip distance: {trip_dist_km:.1f} km."
        )

    # G3: recent durations
    if granularity in ("G3", "G4"):
        recent5 = agg.get("recent5", {}).get(origin_s, [])
        if recent5:
            durs = [f"{r['duration_min']:.0f}" for r in recent5]
            lines.append(f"Recent 5 trip durations from origin: {', '.join(durs)} min.")

    # G4: temporal
    if granularity == "G4":
        pickup_hour = case.get("pickup_hour", 12)
        pickup_dow = case.get("pickup_dow", 0)
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                     "Friday", "Saturday", "Sunday"]
        dow_str = dow_names[pickup_dow] if pickup_dow < 7 else "Unknown"
        hourly_key = f"{origin}_{pickup_hour}"
        h_info = agg.get("hourly_dur", {}).get(hourly_key, {})
        h_mean = h_info.get("mean", 0)
        lines.append(
            f"Pickup: {dow_str} at {pickup_hour:02d}:00. "
            f"Mean duration from origin at this hour: {h_mean:.1f} min."
        )

    lines.append("Question: Predict the trip duration in minutes. Answer with a number only.")
    return "\n".join(lines)
