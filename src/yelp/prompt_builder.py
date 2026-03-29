"""Build prompts for Yelp star rating prediction at 5 granularity levels.

G0: Zero-shot   -- business info only, no user context
G1: City-level   -- adds user city
G2: Category profile -- adds category visit counts + avg ratings per category
G3: Recent trajectory -- adds ordered recent visits with names/categories/ratings
G4: Full spatiotemporal -- adds lat/lon + dates to trajectory
"""

from __future__ import annotations

from src.yelp.data_loader import ReviewRecord, YelpTestCase

SYSTEM_MESSAGE = (
    "You are an expert restaurant and business rating predictor. "
    "Given information about a user and a target business, predict the star rating "
    "the user would give (1 to 5 stars, integers only). "
    "Return ONLY a single integer from 1 to 5. No explanation."
)

VALID_LEVELS = ("G0", "G1", "G2", "G3", "G4")
MAX_HISTORY = 15


def _format_business(target: ReviewRecord, include_coords: bool = False) -> str:
    line = f'"{target.business_name}" (Categories: {target.categories})'
    if include_coords:
        line += f" at ({target.latitude:.4f}, {target.longitude:.4f})"
    return line


def _format_category_profile(profile: dict[str, dict]) -> str:
    sorted_cats = sorted(profile.items(), key=lambda x: -x[1]["count"])
    lines = []
    for cat, info in sorted_cats[:15]:
        lines.append(f"  - {cat}: {info['count']} visits, avg {info['avg_stars']} stars")
    if len(sorted_cats) > 15:
        lines.append(f"  - ... and {len(sorted_cats) - 15} other categories")
    return "\n".join(lines)


def _format_history_semantic(history: list[ReviewRecord]) -> str:
    recent = history[-MAX_HISTORY:]
    lines = []
    for i, r in enumerate(recent, 1):
        lines.append(f"  {i}. {r.business_name} ({r.categories}) -> {r.stars} stars")
    return "\n".join(lines)


def _format_history_full(history: list[ReviewRecord]) -> str:
    recent = history[-MAX_HISTORY:]
    lines = []
    for i, r in enumerate(recent, 1):
        lines.append(
            f"  {i}. [{r.date}] {r.business_name} ({r.categories}) "
            f"at ({r.latitude:.4f}, {r.longitude:.4f}) -> {r.stars} stars"
        )
    return "\n".join(lines)


def build_prompt(
    test_case: YelpTestCase,
    granularity: str,
) -> list[dict[str, str]]:
    if granularity not in VALID_LEVELS:
        raise ValueError(f"Unknown granularity {granularity!r}")

    parts: list[str] = []
    include_coords = granularity == "G4"

    # G1+: City context
    if granularity in ("G1", "G2", "G3", "G4"):
        city = test_case.target.city
        if not city:
            city = test_case.history[-1].city if test_case.history else "Unknown"
        parts.append(f"The user is located in {city}.")

    # G2+: Category profile
    if granularity in ("G2", "G3", "G4") and test_case.user_profile:
        parts.append("The user's review history by category (visits and average rating):")
        parts.append(_format_category_profile(test_case.user_profile))

    # G3: Semantic trajectory
    if granularity == "G3" and test_case.history:
        n = min(len(test_case.history), MAX_HISTORY)
        parts.append(f"The user's {n} most recent reviews (oldest to newest):")
        parts.append(_format_history_semantic(test_case.history))

    # G4: Full spatiotemporal
    if granularity == "G4" and test_case.history:
        n = min(len(test_case.history), MAX_HISTORY)
        parts.append(f"The user's {n} most recent reviews with locations and dates (oldest to newest):")
        parts.append(_format_history_full(test_case.history))

    # Target business (always present)
    parts.append(f"Target business: {_format_business(test_case.target, include_coords)}")
    parts.append("Predict the star rating (1-5) this user would give. Return ONLY a single integer.")

    user_content = "\n\n".join(parts)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": user_content},
    ]
