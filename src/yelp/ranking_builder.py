"""Build test cases for next-business candidate ranking on Yelp."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from src.yelp.data_loader import ReviewRecord


@dataclass
class RankingTestCase:
    user_id: str
    history: list[ReviewRecord]
    ground_truth: ReviewRecord
    candidates: list[ReviewRecord]  # 20 businesses (1 GT + 19 negatives)
    ground_truth_idx: int           # position of GT in candidates
    user_profile: dict


def build_ranking_test_cases(
    df: pd.DataFrame,
    min_user_reviews: int = 20,
    num_negatives: int = 19,
    sample_size: int = 25,
    seed: int = 42,
) -> list[RankingTestCase]:
    rng = random.Random(seed)

    all_businesses = {}
    for _, row in df.drop_duplicates("business_id").iterrows():
        all_businesses[row["business_id"]] = ReviewRecord(
            user_id="",
            business_id=row["business_id"],
            business_name=row["business_name"],
            categories=row["categories"],
            city=row["city"],
            latitude=float(row["latitude"]),
            longitude=float(row["longitude"]),
            stars=0,
            date="",
        )

    all_bids = list(all_businesses.keys())
    user_counts = df.groupby("user_id").size()
    eligible = user_counts[user_counts >= min_user_reviews].index.tolist()
    rng.shuffle(eligible)

    test_cases = []
    for uid in eligible:
        if len(test_cases) >= sample_size:
            break

        user_df = df[df["user_id"] == uid].sort_values("date")
        reviews = []
        for _, row in user_df.iterrows():
            fields = {k: row[k] for k in ReviewRecord.__dataclass_fields__}
            fields["date"] = str(fields["date"])[:10]
            reviews.append(ReviewRecord(**fields))

        history = reviews[:-1]
        gt = reviews[-1]

        # Build category profile from history
        profile: dict[str, dict] = {}
        for r in history:
            for cat in r.categories.split(", "):
                cat = cat.strip()
                if not cat:
                    continue
                if cat not in profile:
                    profile[cat] = {"count": 0, "total_stars": 0}
                profile[cat]["count"] += 1
                profile[cat]["total_stars"] += r.stars
        for cat in profile:
            profile[cat]["avg_stars"] = round(
                profile[cat]["total_stars"] / profile[cat]["count"], 1
            )
            del profile[cat]["total_stars"]

        # Sample negatives (businesses the user hasn't reviewed)
        user_bids = {r.business_id for r in reviews}
        available = [bid for bid in all_bids if bid not in user_bids]
        if len(available) < num_negatives:
            continue
        neg_ids = rng.sample(available, num_negatives)
        negatives = [all_businesses[bid] for bid in neg_ids]

        candidates = negatives + [ReviewRecord(
            user_id="", business_id=gt.business_id,
            business_name=gt.business_name, categories=gt.categories,
            city=gt.city, latitude=gt.latitude, longitude=gt.longitude,
            stars=0, date="",
        )]
        rng.shuffle(candidates)
        gt_idx = next(i for i, c in enumerate(candidates) if c.business_id == gt.business_id)

        test_cases.append(RankingTestCase(
            user_id=uid, history=history, ground_truth=gt,
            candidates=candidates, ground_truth_idx=gt_idx,
            user_profile=profile,
        ))

    return test_cases


def save_ranking_test_cases(cases: list[RankingTestCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(c) for c in cases], indent=2))


def load_ranking_test_cases(path: str | Path) -> list[RankingTestCase]:
    data = json.loads(Path(path).read_text())
    return [
        RankingTestCase(
            user_id=d["user_id"],
            history=[ReviewRecord(**r) for r in d["history"]],
            ground_truth=ReviewRecord(**d["ground_truth"]),
            candidates=[ReviewRecord(**r) for r in d["candidates"]],
            ground_truth_idx=d["ground_truth_idx"],
            user_profile=d["user_profile"],
        )
        for d in data
    ]
