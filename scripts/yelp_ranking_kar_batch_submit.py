#!/usr/bin/env python3
"""Submit KAR reasoning requests for Yelp next-business ranking via OpenAI Batch API (50% cheaper).

Generates one reasoning request per (user, candidate, granularity) tuple.
Ranking has ~60K requests so this will split into multiple batches if needed.

Workflow: submit -> wait (~24h) -> collect (yelp_ranking_kar_batch_collect.py) -> train (yelp_ranking_kar_train.py)
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

from src.yelp.data_loader import ReviewRecord
from src.yelp.ranking_builder import RankingTestCase, load_ranking_test_cases

REASONING_MODEL = "gpt-5-nano"
REASONING_MAX_TOKENS = 4096
MAX_PER_BATCH = 50_000
MAX_HISTORY = 15

KAR_SYSTEM = (
    "You are an expert at predicting user behavior. "
    "Given information about a user and a candidate business, "
    "reason briefly (1-2 sentences) about how likely this user is "
    "to visit this business next and why."
)


def _user_context(tc: RankingTestCase, granularity: str) -> str:
    parts: list[str] = []

    if granularity in ("G1", "G2", "G3", "G4"):
        city = tc.ground_truth.city
        if not city and tc.history:
            city = tc.history[-1].city
        parts.append(f"The user is located in {city or 'Unknown'}.")

    if granularity in ("G2", "G3", "G4") and tc.user_profile:
        sorted_cats = sorted(tc.user_profile.items(), key=lambda x: -x[1]["count"])
        lines = [
            f"  - {cat}: {info['count']} visits, avg {info['avg_stars']} stars"
            for cat, info in sorted_cats[:15]
        ]
        parts.append("The user's review history by category:\n" + "\n".join(lines))

    if granularity == "G3" and tc.history:
        recent = tc.history[-MAX_HISTORY:]
        lines = [
            f"  {i}. {r.business_name} ({r.categories}) -> {r.stars} stars"
            for i, r in enumerate(recent, 1)
        ]
        parts.append(f"Recent {len(recent)} visits:\n" + "\n".join(lines))

    if granularity == "G4" and tc.history:
        recent = tc.history[-MAX_HISTORY:]
        lines = [
            f"  {i}. [{r.date}] {r.business_name} ({r.categories}) "
            f"at ({r.latitude:.4f}, {r.longitude:.4f}) -> {r.stars} stars"
            for i, r in enumerate(recent, 1)
        ]
        parts.append(f"Recent {len(recent)} visits:\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "No user context available."


def _candidate_desc(cand: ReviewRecord, granularity: str) -> str:
    coords = (
        f" at ({cand.latitude:.4f}, {cand.longitude:.4f})"
        if granularity == "G4"
        else ""
    )
    return f'"{cand.business_name}" (Categories: {cand.categories}){coords}'


def build_kar_prompt(
    tc: RankingTestCase, cand: ReviewRecord, granularity: str
) -> list[dict[str, str]]:
    user_ctx = _user_context(tc, granularity)
    cand_desc = _candidate_desc(cand, granularity)
    user_msg = (
        f"{user_ctx}\n\n"
        f"Candidate business: {cand_desc}\n\n"
        f"Reason briefly about how likely this user is to visit this business next."
    )
    return [
        {"role": "system", "content": KAR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    train_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    test_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test ranking cases")

    n_cands = len(train_cases[0].candidates)
    print(f"Candidates per case: {n_cands}")

    requests = []
    for split, cases in [("train", train_cases), ("test", test_cases)]:
        for level in levels:
            for ci, tc in enumerate(cases):
                for cj, cand in enumerate(tc.candidates):
                    msgs = build_kar_prompt(tc, cand, level)
                    requests.append({
                        "custom_id": f"{split}|{level}|{ci:04d}|{cj:02d}",
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
    print(f"Splitting into {len(chunks)} batch file(s)")

    client = OpenAI()
    batch_ids = []

    for ci, chunk in enumerate(chunks):
        jsonl_path = batch_dir / f"ranking_input_{ci}.jsonl"
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
        "task": "ranking",
        "batch_ids": batch_ids,
        "total_requests": len(requests),
        "levels": levels,
        "train_count": len(train_cases),
        "test_count": len(test_cases),
        "n_candidates": n_cands,
    }
    info_path = batch_dir / "ranking_batch_info.json"
    info_path.write_text(json.dumps(info, indent=2))

    print(f"\nBatch info saved to {info_path}")
    print(f"Batch IDs: {batch_ids}")
    print(f"\nCheck status:    python3 scripts/yelp_ranking_kar_batch_collect.py --status")
    print(f"Collect results: python3 scripts/yelp_ranking_kar_batch_collect.py --wait")


if __name__ == "__main__":
    main()
