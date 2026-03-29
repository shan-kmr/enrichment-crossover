#!/usr/bin/env python3
"""Preprocess Yelp data into candidate ranking test cases."""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yelp.data_loader import load_yelp_data
from src.yelp.ranking_builder import build_ranking_test_cases, save_ranking_test_cases


def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]

    biz_path = ROOT / ds["business_path"]
    rev_path = ROOT / ds["review_path"]

    for p, label in [(biz_path, "business"), (rev_path, "review")]:
        if not p.exists():
            print(f"ERROR: {label} file not found at {p}")
            sys.exit(1)

    print("Loading Yelp data (this may take a minute)...")
    df = load_yelp_data(biz_path, rev_path)
    print(f"  {len(df):,} reviews, {df['user_id'].nunique():,} users, {df['business_id'].nunique():,} businesses")

    print("Building ranking test cases...")
    test_cases = build_ranking_test_cases(
        df,
        min_user_reviews=ds["min_user_reviews"],
        num_negatives=ds["num_negative_candidates"],
        sample_size=ds["test_sample_size"],
    )
    print(f"  {len(test_cases)} test cases with {ds['num_negative_candidates'] + 1} candidates each")
    print(f"  Avg history length: {sum(len(tc.history) for tc in test_cases) / len(test_cases):.0f} reviews")

    out_dir = ROOT / ds["processed_dir"]
    save_ranking_test_cases(test_cases, out_dir / "test_cases.json")
    print(f"Saved to {out_dir}/test_cases.json")


if __name__ == "__main__":
    main()
