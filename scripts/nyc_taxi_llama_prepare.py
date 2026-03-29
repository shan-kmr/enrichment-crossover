#!/usr/bin/env python3
"""Prepare JSONL files for Llama 3.1 8B on NYC taxi tasks (all 3 tiers).

For each (task, tier) generates per-granularity files:
  test_G0..G4.jsonl    prompts with metadata for evaluation
  train_G0..G4.jsonl   prompts with assistant answers for SFT

Task A (ranking): answer = "Yes" / "No", metadata = origin_zone + candidate_zone
Task B (duration): answer = duration in minutes, metadata = ground truth
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.nyc_taxi.prompt_builder import build_ranking_prompt, build_duration_prompt

TIERS = ["latlng", "zone_id", "enriched"]

RANKING_SYSTEM = (
    "You are a transportation analyst. Given information about a taxi trip "
    "origin zone and a candidate dropoff zone, determine if this candidate "
    "is the actual next dropoff. Answer only Yes or No."
)
DURATION_SYSTEM = (
    "You are a transportation analyst. Given information about a taxi trip, "
    "predict the trip duration in minutes. Answer with a number only."
)


def main():
    cfg = yaml.safe_load((ROOT / "nyc_taxi_config.yaml").read_text())
    levels = cfg["granularity_levels"]
    proc_dir = ROOT / cfg["dataset"]["processed_dir"]

    agg = json.loads((proc_dir / "train_aggregates.json").read_text())
    centroids_raw = json.loads((proc_dir / "zone_centroids.json").read_text())
    centroids = {int(k): tuple(v) for k, v in centroids_raw.items()}

    zone_cats_path = ROOT / cfg["enrichment"]["zone_categories_path"]
    zone_cats = json.loads(zone_cats_path.read_text()) if zone_cats_path.exists() else None

    ranking_dir = proc_dir / "ranking"
    rank_train = json.loads((ranking_dir / "train_cases.json").read_text())
    rank_test = json.loads((ranking_dir / "test_cases.json").read_text())

    dur_dir = proc_dir / "duration"
    dur_train = json.loads((dur_dir / "train_cases.json").read_text())
    dur_test = json.loads((dur_dir / "test_cases.json").read_text())

    # Subsample duration for Llama: test 200, train 20K (match Gowalla FT size)
    llama_dur_test = dur_test[:cfg["duration"]["llama_test_samples"]]
    llama_dur_train_size = cfg["duration"].get("llama_train_samples", 20000)
    llama_dur_train = dur_train[:llama_dur_train_size]

    print(f"Ranking: {len(rank_train)} train, {len(rank_test)} test")
    print(f"Duration: {len(dur_train)} total → {len(llama_dur_train)} Llama train, {len(llama_dur_test)} Llama test")

    for tier in TIERS:
        zc = zone_cats if tier == "enriched" else None

        # ─── Ranking ─────────────────────────────────────────────
        out_dir = proc_dir / f"llama_ranking_{tier}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'=' * 60}")
        print(f"RANKING — Tier: {tier} → {out_dir}")

        for level in levels:
            test_path = out_dir / f"test_{level}.jsonl"
            with open(test_path, "w") as f:
                for c in rank_test:
                    prompt = build_ranking_prompt(c, level, tier, agg, centroids, zc)
                    msgs = [
                        {"role": "system", "content": RANKING_SYSTEM},
                        {"role": "user", "content": prompt},
                    ]
                    f.write(json.dumps({
                        "messages": msgs,
                        "label": c["label"],
                        "origin_zone": c["origin_zone"],
                        "candidate_zone": c["candidate_zone"],
                    }) + "\n")
            print(f"  {test_path.name}: {len(rank_test)} examples")

            train_path = out_dir / f"train_{level}.jsonl"
            with open(train_path, "w") as f:
                for c in rank_train:
                    prompt = build_ranking_prompt(c, level, tier, agg, centroids, zc)
                    msgs = [
                        {"role": "system", "content": RANKING_SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": "Yes" if c["label"] == 1 else "No"},
                    ]
                    f.write(json.dumps({"messages": msgs}) + "\n")
            size_mb = train_path.stat().st_size / (1024 * 1024)
            print(f"  {train_path.name}: {len(rank_train)} examples ({size_mb:.1f} MB)")

        # ─── Duration ────────────────────────────────────────────
        out_dir = proc_dir / f"llama_duration_{tier}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDURATION — Tier: {tier} → {out_dir}")

        for level in levels:
            test_path = out_dir / f"test_{level}.jsonl"
            with open(test_path, "w") as f:
                for c in llama_dur_test:
                    prompt = build_duration_prompt(c, level, tier, agg, centroids, zc)
                    msgs = [
                        {"role": "system", "content": DURATION_SYSTEM},
                        {"role": "user", "content": prompt},
                    ]
                    f.write(json.dumps({
                        "messages": msgs,
                        "duration_min": c["duration_min"],
                        "PULocationID": c["PULocationID"],
                        "DOLocationID": c["DOLocationID"],
                    }) + "\n")
            print(f"  {test_path.name}: {len(llama_dur_test)} examples")

            train_path = out_dir / f"train_{level}.jsonl"
            with open(train_path, "w") as f:
                for c in llama_dur_train:
                    prompt = build_duration_prompt(c, level, tier, agg, centroids, zc)
                    answer = f"{c['duration_min']:.1f}"
                    msgs = [
                        {"role": "system", "content": DURATION_SYSTEM},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer},
                    ]
                    f.write(json.dumps({"messages": msgs}) + "\n")
            size_mb = train_path.stat().st_size / (1024 * 1024)
            print(f"  {train_path.name}: {len(llama_dur_train)} examples ({size_mb:.1f} MB)")

    print("\nAll JSONL files generated.")


if __name__ == "__main__":
    main()
