#!/usr/bin/env python3
"""Prepare fine-tuning JSONL, upload to OpenAI, and launch fine-tuning jobs.

Creates one fine-tuned model per (base_model, granularity) pair.
Training data: 200 balanced examples (100 pos + 100 neg) per granularity,
drawn from the enriched train set.

Models: gpt-4o-mini, gpt-4.1-mini
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from openai import OpenAI

from src.gowalla.data_loader import load_test_cases
from src.gowalla.enrichment import load_category_map
from src.gowalla.enriched_prompt_builder import build_prompt

N_TRAIN = 200
BASE_MODELS = {
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
}
SEED = 42


def select_balanced_subset(cases, n, seed=42):
    rng = random.Random(seed)
    pos = [tc for tc in cases if tc.label == 1]
    neg = [tc for tc in cases if tc.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    selected = pos[:half] + neg[:half]
    rng.shuffle(selected)
    return selected


def build_jsonl(cases, granularity, cat_map) -> str:
    lines = []
    for tc in cases:
        messages = build_prompt(tc, granularity, cat_map)
        answer = "Yes" if tc.label == 1 else "No"
        messages.append({"role": "assistant", "content": answer})
        lines.append(json.dumps({"messages": messages}))
    return "\n".join(lines)


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    levels = cfg["granularity_levels"]

    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded category map: {len(cat_map)} locations")

    filt_dir = ROOT / enr["filtered_dir"]
    train_cases = load_test_cases(filt_dir / "train_cases.json")
    print(f"Loaded {len(train_cases)} train cases")

    subset = select_balanced_subset(train_cases, N_TRAIN, SEED)
    pos = sum(1 for tc in subset if tc.label == 1)
    print(f"Selected {len(subset)} training examples ({pos} pos, {len(subset)-pos} neg)")

    jsonl_dir = ROOT / "data" / "processed" / "gowalla" / "finetune_jsonl"
    jsonl_dir.mkdir(parents=True, exist_ok=True)

    jsonl_paths = {}
    for level in levels:
        content = build_jsonl(subset, level, cat_map)
        path = jsonl_dir / f"train_{level}.jsonl"
        path.write_text(content)
        jsonl_paths[level] = path
        n_lines = len(content.strip().split("\n"))
        print(f"  {level}: {n_lines} examples -> {path.name}")

    client = OpenAI()

    uploaded_files = {}
    for level in levels:
        print(f"\nUploading {level}...")
        with open(jsonl_paths[level], "rb") as f:
            file_obj = client.files.create(file=f, purpose="fine-tune")
        uploaded_files[level] = file_obj.id
        print(f"  File ID: {file_obj.id}")

    jobs_meta = {}
    for level in levels:
        for short_name, api_name in BASE_MODELS.items():
            job_key = f"{short_name}__{level}"
            print(f"  Launching fine-tune: {job_key}...")
            try:
                job = client.fine_tuning.jobs.create(
                    training_file=uploaded_files[level],
                    model=api_name,
                    hyperparameters={"n_epochs": 3},
                    suffix=f"gowalla-{level.lower()}",
                )
                jobs_meta[job_key] = {
                    "job_id": job.id,
                    "base_model": short_name,
                    "api_model": api_name,
                    "granularity": level,
                    "file_id": uploaded_files[level],
                    "status": job.status,
                }
                print(f"    Job ID: {job.id}, status: {job.status}")
            except Exception as e:
                print(f"    FAILED: {e}")
                jobs_meta[job_key] = {
                    "base_model": short_name,
                    "api_model": api_name,
                    "granularity": level,
                    "status": "failed_to_launch",
                    "error": str(e),
                }

    meta_path = ROOT / enr["results_dir"] / "finetune_jobs.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(jobs_meta, indent=2))
    print(f"\nJob metadata saved to {meta_path}")
    print("Next: python scripts/gowalla_finetune_evaluate.py --poll")


if __name__ == "__main__":
    main()
