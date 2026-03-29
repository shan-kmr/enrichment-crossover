"""Build prompts for Yelp next-business candidate ranking at 5 granularity levels.

Same granularity hierarchy as the rating experiment, but the task is:
rank 20 candidate businesses by likelihood of the user visiting next.
"""

from __future__ import annotations

from src.yelp.data_loader import ReviewRecord
from src.yelp.ranking_builder import RankingTestCase

SYSTEM_MESSAGE = (
    "You are an expert business recommendation system. "
    "Given information about a user and a list of candidate businesses, "
    "rank the candidates from most to least likely for the user to visit next. "
    "Return ONLY a JSON list of candidate numbers in ranked order, e.g. [3, 1, 7, ...]. "
    "Do not include any explanation."
)

VALID_LEVELS = ("G0", "G1", "G2", "G3", "G4")
MAX_HISTORY = 15


def _format_candidate_list(candidates: list[ReviewRecord]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(f"  {i}. {c.business_name} ({c.categories})")
    return "\n".join(lines)


def _format_candidate_list_with_coords(candidates: list[ReviewRecord]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"  {i}. {c.business_name} ({c.categories}) "
            f"at ({c.latitude:.4f}, {c.longitude:.4f})"
        )
    return "\n".join(lines)


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


def build_ranking_prompt(
    test_case: RankingTestCase,
    granularity: str,
) -> list[dict[str, str]]:
    if granularity not in VALID_LEVELS:
        raise ValueError(f"Unknown granularity {granularity!r}")

    parts: list[str] = []
    use_coords = granularity == "G4"

    if granularity in ("G1", "G2", "G3", "G4"):
        city = test_case.ground_truth.city
        if not city and test_case.history:
            city = test_case.history[-1].city
        parts.append(f"The user is located in {city or 'Unknown'}.")

    if granularity in ("G2", "G3", "G4") and test_case.user_profile:
        parts.append("The user's review history by category (visits and average rating):")
        parts.append(_format_category_profile(test_case.user_profile))

    if granularity == "G3" and test_case.history:
        n = min(len(test_case.history), MAX_HISTORY)
        parts.append(f"The user's {n} most recent visits (oldest to newest):")
        parts.append(_format_history_semantic(test_case.history))

    if granularity == "G4" and test_case.history:
        n = min(len(test_case.history), MAX_HISTORY)
        parts.append(f"The user's {n} most recent visits with locations and dates (oldest to newest):")
        parts.append(_format_history_full(test_case.history))

    parts.append("Candidate businesses to rank:")
    if use_coords:
        parts.append(_format_candidate_list_with_coords(test_case.candidates))
    else:
        parts.append(_format_candidate_list(test_case.candidates))

    num = len(test_case.candidates)
    parts.append(
        f"Rank all {num} candidates from most to least likely "
        "for this user to visit next. Return ONLY a JSON list of candidate numbers (1-indexed)."
    )

    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
