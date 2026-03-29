#!/usr/bin/env python3
"""Build training ranking cases for embedding/handcrafted ML experiments.

Selects 500 users NOT in the test set, builds one ranking case each
(1 ground truth + 19 negatives = 20 candidates), yielding 10,000
pointwise training samples.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yelp.data_loader import ReviewRecord, load_yelp_data
from src.yelp.ranking_builder import (
    RankingTestCase,
    load_ranking_test_cases,
    save_ranking_test_cases,
)


def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]

    test_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    test_user_ids = {tc.user_id for tc in test_cases}
    print(f"Loaded {len(test_user_ids)} test users to exclude")

    print("Loading Yelp data...")
    df = load_yelp_data(ROOT / ds["business_path"], ROOT / ds["review_path"])
    print(f"  {len(df):,} reviews, {df['user_id'].nunique():,} users")

    all_businesses: dict[str, ReviewRecord] = {}
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
    eligible = user_counts[user_counts >= ds["min_user_reviews"]].index.tolist()
    eligible = [u for u in eligible if u not in test_user_ids]
    print(f"  {len(eligible)} eligible training users")

    rng = random.Random(123)
    rng.shuffle(eligible)
    train_users = eligible[:500]
    num_negatives = ds["num_negative_candidates"]

    train_cases: list[RankingTestCase] = []
    for uid in train_users:
        user_df = df[df["user_id"] == uid].sort_values("date")
        reviews: list[ReviewRecord] = []
        for _, row in user_df.iterrows():
            fields = {k: row[k] for k in ReviewRecord.__dataclass_fields__}
            fields["date"] = str(fields["date"])[:10]
            reviews.append(ReviewRecord(**fields))

        history = reviews[:-1]
        gt = reviews[-1]

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

        user_bids = {r.business_id for r in reviews}
        available = [bid for bid in all_bids if bid not in user_bids]
        if len(available) < num_negatives:
            continue

        neg_ids = rng.sample(available, num_negatives)
        negatives = [all_businesses[bid] for bid in neg_ids]

        candidates = negatives + [
            ReviewRecord(
                user_id="",
                business_id=gt.business_id,
                business_name=gt.business_name,
                categories=gt.categories,
                city=gt.city,
                latitude=gt.latitude,
                longitude=gt.longitude,
                stars=0,
                date="",
            )
        ]
        rng.shuffle(candidates)
        gt_idx = next(
            i for i, c in enumerate(candidates) if c.business_id == gt.business_id
        )

        train_cases.append(
            RankingTestCase(
                user_id=uid,
                history=history,
                ground_truth=gt,
                candidates=candidates,
                ground_truth_idx=gt_idx,
                user_profile=profile,
            )
        )

    print(f"Built {len(train_cases)} training ranking cases ({len(train_cases) * (num_negatives + 1)} pointwise samples)")

    out_path = ROOT / ds["processed_dir"] / "train_cases.json"
    save_ranking_test_cases(train_cases, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
