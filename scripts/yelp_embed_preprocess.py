#!/usr/bin/env python3
"""Build training cases for the embedding experiment from users not in the test set."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yelp.data_loader import ReviewRecord, YelpTestCase, load_test_cases, load_yelp_data, save_test_cases


def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]

    test_cases_path = ROOT / ds["processed_dir"] / "test_cases.json"
    test_cases = load_test_cases(test_cases_path)
    test_user_ids = {tc.user_id for tc in test_cases}
    print(f"Loaded {len(test_user_ids)} test users to exclude")

    print("Loading Yelp data (this may take a minute)...")
    df = load_yelp_data(ROOT / ds["business_path"], ROOT / ds["review_path"])
    print(f"  {len(df):,} reviews, {df['user_id'].nunique():,} users")

    user_counts = df.groupby("user_id").size()
    eligible = user_counts[user_counts >= ds["min_user_reviews"]].index.tolist()
    eligible = [u for u in eligible if u not in test_user_ids]
    print(f"  {len(eligible)} eligible training users")

    rng = random.Random(123)
    rng.shuffle(eligible)
    train_users = eligible[:1500]

    train_cases: list[YelpTestCase] = []
    test_per_user = ds["test_reviews_per_user"]

    for uid in train_users:
        user_df = df[df["user_id"] == uid].sort_values("date")
        reviews = []
        for _, row in user_df.iterrows():
            fields = {k: row[k] for k in ReviewRecord.__dataclass_fields__}
            fields["date"] = str(fields["date"])[:10]
            reviews.append(ReviewRecord(**fields))

        split_idx = len(reviews) - test_per_user
        history = reviews[:split_idx]
        targets = reviews[split_idx:]

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

        for target in targets:
            train_cases.append(YelpTestCase(
                user_id=uid,
                history=history,
                target=target,
                user_profile=profile,
            ))

    print(f"Built {len(train_cases)} training cases from {len(train_users)} users")

    out_path = ROOT / ds["processed_dir"] / "train_cases.json"
    save_test_cases(train_cases, out_path)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
