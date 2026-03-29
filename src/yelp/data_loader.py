"""Load Yelp Open Dataset JSON files and build user review histories."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ReviewRecord:
    user_id: str
    business_id: str
    business_name: str
    categories: str
    city: str
    latitude: float
    longitude: float
    stars: int
    date: str  # YYYY-MM-DD


@dataclass
class YelpTestCase:
    user_id: str
    history: list[ReviewRecord]  # past reviews (context)
    target: ReviewRecord         # review to predict the star rating for
    user_profile: dict           # {category: {"count": N, "avg_stars": X}}


def load_yelp_data(business_path: str | Path, review_path: str | Path) -> pd.DataFrame:
    """Load and join business + review JSON files into a single DataFrame."""
    businesses = {}
    with open(business_path, "r") as f:
        for line in f:
            b = json.loads(line)
            businesses[b["business_id"]] = {
                "business_name": b.get("name", ""),
                "categories": b.get("categories") or "",
                "city": b.get("city", ""),
                "latitude": b.get("latitude", 0.0),
                "longitude": b.get("longitude", 0.0),
            }

    records = []
    with open(review_path, "r") as f:
        for line in f:
            r = json.loads(line)
            bid = r["business_id"]
            if bid not in businesses:
                continue
            b = businesses[bid]
            records.append({
                "user_id": r["user_id"],
                "business_id": bid,
                "business_name": b["business_name"],
                "categories": b["categories"],
                "city": b["city"],
                "latitude": b["latitude"],
                "longitude": b["longitude"],
                "stars": int(r["stars"]),
                "date": r["date"],
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["user_id", "date"]).reset_index(drop=True)
    return df


def build_test_cases(
    df: pd.DataFrame,
    min_user_reviews: int = 20,
    test_reviews_per_user: int = 3,
    sample_size: int = 25,
    seed: int = 42,
) -> list[YelpTestCase]:
    rng = random.Random(seed)

    user_counts = df.groupby("user_id").size()
    eligible_users = user_counts[user_counts >= min_user_reviews].index.tolist()
    rng.shuffle(eligible_users)

    test_cases = []
    for uid in eligible_users:
        if len(test_cases) >= sample_size:
            break

        user_df = df[df["user_id"] == uid].sort_values("date")
        reviews = []
        for _, row in user_df.iterrows():
            fields = {k: row[k] for k in ReviewRecord.__dataclass_fields__}
            fields["date"] = str(fields["date"])[:10]  # YYYY-MM-DD
            reviews.append(ReviewRecord(**fields))


        split_idx = len(reviews) - test_reviews_per_user
        history = reviews[:split_idx]
        test_reviews = reviews[split_idx:]

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

        # Pick one test review per user (the last one)
        target = test_reviews[-1]
        test_cases.append(YelpTestCase(
            user_id=uid,
            history=history,
            target=target,
            user_profile=profile,
        ))

    return test_cases


def save_test_cases(test_cases: list[YelpTestCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(tc) for tc in test_cases]
    path.write_text(json.dumps(data, indent=2))


def load_test_cases(path: str | Path) -> list[YelpTestCase]:
    data = json.loads(Path(path).read_text())
    return [
        YelpTestCase(
            user_id=d["user_id"],
            history=[ReviewRecord(**r) for r in d["history"]],
            target=ReviewRecord(**d["target"]),
            user_profile=d["user_profile"],
        )
        for d in data
    ]
