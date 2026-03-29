#!/usr/bin/env python3
"""Preprocess Foursquare NYC data into trajectories and test cases."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_foursquare_nyc
from src.trajectory_builder import (
    build_test_cases,
    build_trajectories,
    build_user_profiles,
    chronological_split,
    save_test_cases,
    save_user_profiles,
)


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    ds = cfg["dataset"]

    raw_path = ROOT / ds["raw_path"]
    if not raw_path.exists():
        print(f"ERROR: Dataset not found at {raw_path}")
        print("Download from: https://www.kaggle.com/datasets/chetanism/foursquare-nyc-and-tokyo-checkin-dataset")
        print(f"Place dataset_TSMC2014_NYC.csv in {raw_path.parent}/")
        sys.exit(1)

    print(f"Loading dataset from {raw_path}...")
    df, venue_meta = load_foursquare_nyc(raw_path)
    print(f"  {len(df):,} check-ins, {df['userId'].nunique()} users, {df['venueId'].nunique()} venues")

    print("Building trajectories...")
    trajectories = build_trajectories(
        df,
        session_gap_hours=ds["session_gap_hours"],
        min_length=ds["min_trajectory_length"],
    )
    print(f"  {len(trajectories)} trajectories (min length {ds['min_trajectory_length']})")

    print("Splitting chronologically...")
    train, val, test = chronological_split(
        trajectories,
        train_ratio=ds["train_ratio"],
        val_ratio=ds["val_ratio"],
    )
    print(f"  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print("Building user profiles from training data...")
    profiles = build_user_profiles(train)
    print(f"  Profiles for {len(profiles)} users")

    print(f"Building test cases (sampling {ds['test_sample_size']} trajectories)...")
    test_cases = build_test_cases(
        test,
        venue_meta,
        num_negatives=ds["num_negative_candidates"],
        sample_size=ds["test_sample_size"],
    )
    print(f"  {len(test_cases)} test cases with {ds['num_negative_candidates'] + 1} candidates each")

    out_dir = ROOT / ds["processed_dir"]
    save_test_cases(test_cases, out_dir / "test_cases.json")
    save_user_profiles(profiles, out_dir / "user_profiles.json")
    venue_meta.to_csv(out_dir / "venue_meta.csv", index=False)
    print(f"Saved processed data to {out_dir}/")


if __name__ == "__main__":
    main()
