"""Build user trajectories, chronological splits, and candidate sets."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CheckIn:
    venue_id: str
    category: str
    category_id: str
    latitude: float
    longitude: float
    timestamp: str  # ISO format string


@dataclass
class TestCase:
    user_id: int
    trajectory: list[CheckIn]  # context check-ins (all but last)
    ground_truth: CheckIn       # held-out last check-in
    candidates: list[CheckIn]   # 20 POIs including ground truth
    ground_truth_idx: int       # position of ground truth in candidates


def _row_to_checkin(row: pd.Series) -> CheckIn:
    return CheckIn(
        venue_id=str(row["venueId"]),
        category=str(row["venueCategory"]),
        category_id=str(row["venueCategoryId"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        timestamp=row["utcTimestamp"].isoformat(),
    )


def build_trajectories(
    df: pd.DataFrame,
    session_gap_hours: float = 24.0,
    min_length: int = 3,
) -> list[tuple[int, list[CheckIn]]]:
    """Split each user's check-ins into session-based trajectories."""
    gap = pd.Timedelta(hours=session_gap_hours)
    trajectories = []

    for user_id, group in df.groupby("userId"):
        group = group.sort_values("utcTimestamp")
        session: list[CheckIn] = []
        prev_time = None

        for _, row in group.iterrows():
            if prev_time is not None and (row["utcTimestamp"] - prev_time) > gap:
                if len(session) >= min_length:
                    trajectories.append((int(user_id), session))
                session = []
            session.append(_row_to_checkin(row))
            prev_time = row["utcTimestamp"]

        if len(session) >= min_length:
            trajectories.append((int(user_id), session))

    return trajectories


def chronological_split(
    trajectories: list[tuple[int, list[CheckIn]]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[list, list, list]:
    """Split trajectories chronologically by their first timestamp."""
    sorted_trajs = sorted(trajectories, key=lambda t: t[1][0].timestamp)
    n = len(sorted_trajs)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return sorted_trajs[:train_end], sorted_trajs[train_end:val_end], sorted_trajs[val_end:]


def build_test_cases(
    test_trajectories: list[tuple[int, list[CheckIn]]],
    venue_meta: pd.DataFrame,
    num_negatives: int = 19,
    sample_size: int | None = 300,
    seed: int = 42,
) -> list[TestCase]:
    """For each test trajectory, hold out the last check-in and sample negative candidates."""
    rng = random.Random(seed)
    all_venue_ids = set(venue_meta["venueId"].astype(str).tolist())
    venue_lookup = {
        str(row["venueId"]): CheckIn(
            venue_id=str(row["venueId"]),
            category=str(row["venueCategory"]),
            category_id=str(row["venueCategoryId"]),
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            timestamp="",
        )
        for _, row in venue_meta.iterrows()
    }

    trajs = list(test_trajectories)
    if sample_size and len(trajs) > sample_size:
        rng.shuffle(trajs)
        trajs = trajs[:sample_size]

    test_cases = []
    for user_id, traj in trajs:
        if len(traj) < 2:
            continue
        context = traj[:-1]
        gt = traj[-1]

        excluded = {c.venue_id for c in traj}
        available = list(all_venue_ids - excluded)
        if len(available) < num_negatives:
            available = list(all_venue_ids - {gt.venue_id})
        negatives_ids = rng.sample(available, min(num_negatives, len(available)))
        negatives = [venue_lookup[vid] for vid in negatives_ids]

        candidates = negatives + [
            CheckIn(
                venue_id=gt.venue_id,
                category=gt.category,
                category_id=gt.category_id,
                latitude=gt.latitude,
                longitude=gt.longitude,
                timestamp="",
            )
        ]
        rng.shuffle(candidates)
        gt_idx = next(i for i, c in enumerate(candidates) if c.venue_id == gt.venue_id)

        test_cases.append(TestCase(
            user_id=user_id,
            trajectory=context,
            ground_truth=gt,
            candidates=candidates,
            ground_truth_idx=gt_idx,
        ))

    return test_cases


def build_user_profiles(
    train_trajectories: list[tuple[int, list[CheckIn]]],
) -> dict[int, dict[str, int]]:
    """Aggregate category visit counts per user from training data."""
    profiles: dict[int, dict[str, int]] = {}
    for user_id, traj in train_trajectories:
        if user_id not in profiles:
            profiles[user_id] = {}
        for ci in traj:
            profiles[user_id][ci.category] = profiles[user_id].get(ci.category, 0) + 1
    return profiles


def save_test_cases(test_cases: list[TestCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(tc) for tc in test_cases]
    path.write_text(json.dumps(data, indent=2))


def load_test_cases(path: str | Path) -> list[TestCase]:
    data = json.loads(Path(path).read_text())
    cases = []
    for d in data:
        cases.append(TestCase(
            user_id=d["user_id"],
            trajectory=[CheckIn(**c) for c in d["trajectory"]],
            ground_truth=CheckIn(**d["ground_truth"]),
            candidates=[CheckIn(**c) for c in d["candidates"]],
            ground_truth_idx=d["ground_truth_idx"],
        ))
    return cases


def save_user_profiles(profiles: dict[int, dict[str, int]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(k): v for k, v in profiles.items()}
    path.write_text(json.dumps(serializable, indent=2))


def load_user_profiles(path: str | Path) -> dict[int, dict[str, int]]:
    data = json.loads(Path(path).read_text())
    return {int(k): v for k, v in data.items()}
