#!/usr/bin/env python3
"""Submit KAR reasoning requests for Yelp rating prediction via OpenAI Batch API (50% cheaper).

Workflow: submit -> wait (~24h) -> collect (yelp_kar_batch_collect.py) -> train (yelp_kar_train.py)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from openai import OpenAI

from src.yelp.data_loader import YelpTestCase, load_test_cases
from src.yelp.prompt_builder import build_prompt

REASONING_MODEL = "gpt-5-nano"
REASONING_MAX_TOKENS = 4096
MAX_PER_BATCH = 50_000

KAR_SYSTEM = (
    "You are an expert at analyzing user preferences. "
    "Given information about a user and a target business, "
    "reason briefly (2-3 sentences) about what star rating this user "
    "would likely give and why. Consider their history, category "
    "preferences, and alignment with the business."
)


def build_kar_prompt(tc: YelpTestCase, granularity: str) -> list[dict[str, str]]:
    messages = build_prompt(tc, granularity)
    messages[0]["content"] = KAR_SYSTEM
    messages[1]["content"] = messages[1]["content"].replace(
        "Predict the star rating (1-5) this user would give. Return ONLY a single integer.",
        "Reason briefly about what star rating this user would give to this business and why.",
    )
    return messages


def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    train_cases = load_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    test_cases = load_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test cases")

    requests = []
    for split, cases in [("train", train_cases), ("test", test_cases)]:
        for level in levels:
            for i, tc in enumerate(cases):
                msgs = build_kar_prompt(tc, level)
                requests.append({
                    "custom_id": f"{split}|{level}|{i:05d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": REASONING_MODEL,
                        "messages": msgs,
                        "max_completion_tokens": REASONING_MAX_TOKENS,
                    },
                })

    print(f"Total requests: {len(requests)}")

    batch_dir = ROOT / "results" / "kar_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)

    chunks = [
        requests[i : i + MAX_PER_BATCH]
        for i in range(0, len(requests), MAX_PER_BATCH)
    ]

    client = OpenAI()
    batch_ids = []

    for ci, chunk in enumerate(chunks):
        jsonl_path = batch_dir / f"rating_input_{ci}.jsonl"
        with open(jsonl_path, "w") as f:
            for req in chunk:
                f.write(json.dumps(req) + "\n")
        print(f"Wrote {len(chunk)} requests to {jsonl_path.name}")

        with open(jsonl_path, "rb") as f:
            uploaded = client.files.create(file=f, purpose="batch")
        print(f"Uploaded file: {uploaded.id}")

        batch = client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_ids.append(batch.id)
        print(f"Created batch: {batch.id} (status: {batch.status})")

    info = {
        "task": "rating",
        "batch_ids": batch_ids,
        "total_requests": len(requests),
        "levels": levels,
        "train_count": len(train_cases),
        "test_count": len(test_cases),
    }
    info_path = batch_dir / "rating_batch_info.json"
    info_path.write_text(json.dumps(info, indent=2))

    print(f"\nBatch info saved to {info_path}")
    print(f"Batch IDs: {batch_ids}")
    print(f"\nCheck status:  python3 scripts/yelp_kar_batch_collect.py --status")
    print(f"Collect results: python3 scripts/yelp_kar_batch_collect.py --wait")


if __name__ == "__main__":
    main()
