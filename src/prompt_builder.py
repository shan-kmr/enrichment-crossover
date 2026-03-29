"""Build prompts at 5 granularity levels for next-POI candidate ranking.

Granularity levels (each strictly adds information):
  G0: No context         -- candidate list only
  G1: City-level         -- adds city name
  G2: Category profile   -- adds aggregated visit counts by category
  G3: Recent trajectory  -- adds ordered recent check-ins (names + categories)
  G4: Full spatiotemporal -- adds lat/lon + timestamps to trajectory
"""

from __future__ import annotations

from src.trajectory_builder import CheckIn, TestCase

SYSTEM_MESSAGE = (
    "You are an expert location recommendation system. "
    "Given information about a user and a list of candidate places, "
    "rank the candidates from most to least likely for the user to visit next. "
    "Return ONLY a JSON list of candidate numbers in ranked order, e.g. [3, 1, 7, ...]. "
    "Do not include any explanation."
)

VALID_LEVELS = ("G0", "G1", "G2", "G3", "G4")
MAX_RECENT_CHECKINS = 10


def _format_candidate_list(candidates: list[CheckIn]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"  {i}. {c.category} (Venue {c.venue_id})")
    return "\n".join(lines)


def _format_candidate_list_with_coords(candidates: list[CheckIn]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"  {i}. {c.category} (Venue {c.venue_id}) "
            f"at ({c.latitude:.4f}, {c.longitude:.4f})"
        )
    return "\n".join(lines)


def _format_category_profile(profile: dict[str, int]) -> str:
    sorted_cats = sorted(profile.items(), key=lambda x: -x[1])
    lines = [f"  - {cat}: {count} visits" for cat, count in sorted_cats[:15]]
    if len(sorted_cats) > 15:
        lines.append(f"  - ... and {len(sorted_cats) - 15} other categories")
    return "\n".join(lines)


def _format_trajectory_semantic(trajectory: list[CheckIn]) -> str:
    recent = trajectory[-MAX_RECENT_CHECKINS:]
    lines = []
    for i, c in enumerate(recent, 1):
        lines.append(f"  {i}. {c.category} (Venue {c.venue_id})")
    return "\n".join(lines)


def _format_trajectory_full(trajectory: list[CheckIn]) -> str:
    recent = trajectory[-MAX_RECENT_CHECKINS:]
    lines = []
    for i, c in enumerate(recent, 1):
        ts = c.timestamp.replace("T", " ") if "T" in c.timestamp else c.timestamp
        lines.append(
            f"  {i}. [{ts}] {c.category} (Venue {c.venue_id}) "
            f"at ({c.latitude:.4f}, {c.longitude:.4f})"
        )
    return "\n".join(lines)


def build_prompt(
    test_case: TestCase,
    granularity: str,
    user_profile: dict[str, int] | None = None,
    city: str = "New York City",
) -> list[dict[str, str]]:
    """Build a chat-format prompt (system + user messages) for the given granularity.

    Returns list of {"role": ..., "content": ...} dicts.
    """
    if granularity not in VALID_LEVELS:
        raise ValueError(f"Unknown granularity {granularity!r}, expected one of {VALID_LEVELS}")

    parts: list[str] = []
    use_coords_in_candidates = granularity == "G4"

    # --- G1+: City context ---
    if granularity in ("G1", "G2", "G3", "G4"):
        parts.append(f"The user is located in {city}.")

    # --- G2+: Category profile ---
    if granularity in ("G2", "G3", "G4") and user_profile:
        parts.append("The user's historical visit profile (category: visit count):")
        parts.append(_format_category_profile(user_profile))

    # --- G3+: Recent trajectory (semantic) ---
    if granularity == "G3":
        parts.append(
            f"The user's {min(len(test_case.trajectory), MAX_RECENT_CHECKINS)} "
            "most recent check-ins (oldest to newest):"
        )
        parts.append(_format_trajectory_semantic(test_case.trajectory))

    # --- G4: Full spatiotemporal trajectory ---
    if granularity == "G4":
        parts.append(
            f"The user's {min(len(test_case.trajectory), MAX_RECENT_CHECKINS)} "
            "most recent check-ins with locations and timestamps (oldest to newest):"
        )
        parts.append(_format_trajectory_full(test_case.trajectory))

    # --- Candidate list (always present) ---
    parts.append("Candidate places to rank:")
    if use_coords_in_candidates:
        parts.append(_format_candidate_list_with_coords(test_case.candidates))
    else:
        parts.append(_format_candidate_list(test_case.candidates))

    num_candidates = len(test_case.candidates)
    parts.append(
        f"Rank all {num_candidates} candidates from most to least likely "
        "for this user to visit next. Return ONLY a JSON list of candidate "
        "numbers (1-indexed)."
    )

    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
