"""Parse LLM ranking output into structured candidate orderings."""

from __future__ import annotations

import json
import re


def parse_ranking(response_text: str, num_candidates: int) -> list[int] | None:
    """Parse a ranked list of 1-indexed candidate numbers from LLM output.

    Returns a list of 0-indexed candidate positions, or None if parsing fails.
    """
    text = response_text.strip()

    # Try parsing as JSON array first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(x, int) for x in parsed):
            return [x - 1 for x in parsed if 1 <= x <= num_candidates]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting a JSON array from within the text
    match = re.search(r"\[[\d,\s]+\]", text)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [x - 1 for x in parsed if isinstance(x, int) and 1 <= x <= num_candidates]
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: extract all integers from the text
    numbers = [int(x) for x in re.findall(r"\b(\d+)\b", text)]
    if numbers:
        seen = set()
        result = []
        for n in numbers:
            if 1 <= n <= num_candidates and n not in seen:
                result.append(n - 1)
                seen.add(n)
        if result:
            return result

    return None
