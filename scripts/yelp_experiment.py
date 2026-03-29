#!/usr/bin/env python3
"""Run the Yelp star rating prediction experiment."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from src.llm_client import CachedLLMClient
from src.yelp.data_loader import load_test_cases
from src.yelp.prompt_builder import build_prompt


def parse_star_rating(text: str) -> int | None:
    text = text.strip()
    # Direct single digit
    if text in ("1", "2", "3", "4", "5"):
        return int(text)
    # First integer 1-5 found in text
    match = re.search(r"\b([1-5])\b", text)
    return int(match.group(1)) if match else None


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_checkpoint(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def run_condition(client, model, granularity, test_cases, cfg, results_dir):
    condition_key = f"{model}__{granularity}"
    result_path = results_dir / f"{condition_key}.json"
    checkpoint_path = results_dir / f"{condition_key}_checkpoint.json"

    existing = load_checkpoint(checkpoint_path)
    completed_indices = set(existing.get("completed", []))
    results = existing.get("results", [])

    llm_cfg = cfg["llm"]

    for i, tc in enumerate(tqdm(test_cases, desc=f"  {model} / {granularity}", leave=False)):
        if i in completed_indices:
            continue

        messages = build_prompt(tc, granularity)

        is_reasoning = "gpt-5" in model or "o3" in model or "o4" in model
        tokens = max(llm_cfg["max_tokens"], 10000) if is_reasoning else llm_cfg["max_tokens"]

        try:
            response = client.chat(
                model=model,
                messages=messages,
                temperature=llm_cfg["temperature"],
                max_tokens=tokens,
            )
        except RuntimeError as e:
            print(f"    FAILED test case {i}: {e}")
            results.append({
                "index": i, "user_id": tc.user_id,
                "error": str(e), "predicted": None,
                "actual": tc.target.stars,
            })
            completed_indices.add(i)
            continue

        predicted = parse_star_rating(response["content"])
        results.append({
            "index": i,
            "user_id": tc.user_id,
            "predicted": predicted,
            "actual": tc.target.stars,
            "raw_response": response["content"],
            "usage": response.get("usage"),
            "cached": response.get("cached", False),
        })
        completed_indices.add(i)

        if len(completed_indices) % 25 == 0:
            save_checkpoint({"completed": sorted(completed_indices), "results": results}, checkpoint_path)

    save_checkpoint({"completed": sorted(completed_indices), "results": results}, checkpoint_path)
    result_path.write_text(json.dumps(results, indent=2))
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return {"condition": condition_key, "num_results": len(results)}


def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds_cfg = cfg["dataset"]

    test_cases_path = ROOT / ds_cfg["processed_dir"] / "test_cases.json"
    if not test_cases_path.exists():
        print("ERROR: Run scripts/yelp_preprocess.py first.")
        sys.exit(1)

    print("Loading test cases...")
    test_cases = load_test_cases(test_cases_path)
    print(f"  {len(test_cases)} test cases")

    client = CachedLLMClient(
        cache_dir=ROOT / cfg["llm"]["cache_dir"],
        max_retries=cfg["llm"]["max_retries"],
        retry_base_delay=cfg["llm"]["retry_base_delay"],
        request_timeout=cfg["llm"]["request_timeout"],
    )

    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    models = cfg["models"]
    levels = cfg["granularity_levels"]
    total = len(models) * len(levels)

    print(f"\nRunning {total} conditions ({len(models)} models x {len(levels)} levels)")
    print(f"  Models: {models}")
    print(f"  Levels: {levels}")
    print()

    summary = []
    for model in models:
        for level in levels:
            result = run_condition(client, model, level, test_cases, cfg, results_dir)
            summary.append(result)
            print(f"  Completed {result['condition']}: {result['num_results']} results")

    summary_path = results_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary at {summary_path}")


if __name__ == "__main__":
    main()
