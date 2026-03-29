#!/usr/bin/env python3
"""Prepare JSONL files for Llama 3.1 8B next-location prediction (all 3 tiers).

For each tier generates per-granularity files:
  test_G0..G4.jsonl    4 000 prompts  (no assistant answer, with metadata)
  train_G0..G4.jsonl  20 000 prompts  (with assistant answer for SFT)

Metadata (user_id, candidate_location_id) is included in test lines
so the HPC evaluation script can reconstruct per-user rankings.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gowalla.data_loader import load_nextloc_cases
from src.gowalla.nextloc_prompt_builder import build_prompt, SYSTEM
from src.gowalla.enrichment import load_category_map

TIERS = ["latlng", "venue_id", "enriched"]


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    nl = cfg["nextloc"]
    levels = cfg["granularity_levels"]

    data_dir = ROOT / nl["data_dir"]
    train_cases = load_nextloc_cases(data_dir / "train_cases.json")
    test_cases = load_nextloc_cases(data_dir / "test_cases.json")
    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test, "
          f"{len(cat_map)} categories")

    for tier in TIERS:
        out_dir = (ROOT / "data" / "processed" / "gowalla"
                   / f"nextloc_llama_{tier}")
        out_dir.mkdir(parents=True, exist_ok=True)

        cm = cat_map if tier == "enriched" else None

        print(f"\n{'=' * 60}")
        print(f"Tier: {tier} → {out_dir}")
        print(f"{'=' * 60}")

        for level in levels:
            # ── test ────────────────────────────────────────────────
            test_path = out_dir / f"test_{level}.jsonl"
            with open(test_path, "w") as f:
                for tc in tqdm(test_cases, desc=f"test_{level}"):
                    msgs = build_prompt(tc, level, tier, cm)
                    f.write(json.dumps({
                        "messages": msgs,
                        "label": tc.label,
                        "user_id": tc.user.user_id,
                        "candidate_location_id": tc.candidate_location_id,
                    }) + "\n")
            print(f"  {test_path.name}: {len(test_cases)} examples")

            # ── train ───────────────────────────────────────────────
            train_path = out_dir / f"train_{level}.jsonl"
            with open(train_path, "w") as f:
                for tc in tqdm(train_cases, desc=f"train_{level}"):
                    msgs = build_prompt(tc, level, tier, cm)
                    msgs.append({
                        "role": "assistant",
                        "content": "Yes" if tc.label == 1 else "No",
                    })
                    f.write(json.dumps({"messages": msgs}) + "\n")
            size_mb = train_path.stat().st_size / (1024 * 1024)
            print(f"  {train_path.name}: {len(train_cases)} examples "
                  f"({size_mb:.1f} MB)")

    print("\nAll JSONL files generated.")


if __name__ == "__main__":
    main()
