"""Load and preprocess the Foursquare NYC check-in CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_foursquare_nyc(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [
        "userId",
        "venueId",
        "venueCategoryId",
        "venueCategory",
        "latitude",
        "longitude",
        "timezoneOffset",
        "utcTimestamp",
    ]
    df["utcTimestamp"] = pd.to_datetime(df["utcTimestamp"])
    df = df.sort_values(["userId", "utcTimestamp"]).reset_index(drop=True)

    # Build a venue metadata lookup (deduplicated, take first occurrence)
    venue_meta = (
        df.groupby("venueId")
        .agg(
            venueCategory=("venueCategory", "first"),
            venueCategoryId=("venueCategoryId", "first"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )
    return df, venue_meta
