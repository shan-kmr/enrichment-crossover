#!/usr/bin/env python3
"""Prepare PURE LAT/LNG JSONL files for Llama 3.1 8B ablation experiment.

Same filtered user pool (80% Overture coverage) as enriched/non-enriched,
but prompts use only rounded coordinates and geo-cell features.
No venue IDs, no category names.

Generates 5 train + 5 test JSONL files:
  - test_G0..G4.jsonl   (200 test prompts, no assistant answer)
  - train_G0..G4.jsonl  (20K train prompts with assistant answer for SFT)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gowalla.data_loader import load_test_cases
from src.gowalla.latlng_prompt_builder import build_prompt, SYSTEM


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    levels = cfg["granularity_levels"]

    filt_dir = ROOT / enr["filtered_dir"]
    test_cases = load_test_cases(filt_dir / "test_cases.json")
    ml_train_cases = load_test_cases(filt_dir / "ml_train_cases.json")
    print(f"Loaded {len(test_cases)} test, {len(ml_train_cases)} ML train cases "
          f"(same filtered pool as enriched)")

    out_dir = ROOT / "data" / "processed" / "gowalla" / "llama_jsonl_latlng"
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        test_path = out_dir / f"test_{level}.jsonl"
        with open(test_path, "w") as f:
            for tc in tqdm(test_cases, desc=f"test_{level}"):
                msgs = build_prompt(tc, level)
                f.write(json.dumps({"messages": msgs, "label": tc.label}) + "\n")
        print(f"  {test_path.name}: {len(test_cases)} examples")

        train_path = out_dir / f"train_{level}.jsonl"
        with open(train_path, "w") as f:
            for tc in tqdm(ml_train_cases, desc=f"train_{level}"):
                msgs = build_prompt(tc, level)
                msgs.append({"role": "assistant", "content": "Yes" if tc.label == 1 else "No"})
                f.write(json.dumps({"messages": msgs}) + "\n")
        size_mb = train_path.stat().st_size / (1024 * 1024)
        print(f"  {train_path.name}: {len(ml_train_cases)} examples ({size_mb:.1f} MB)")

    print(f"\nAll lat/lng JSONL files saved to {out_dir}/")


if __name__ == "__main__":
    main()
