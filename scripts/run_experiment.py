#!/usr/bin/env python3
"""Run the full experiment: models x granularity levels x test cases."""

import json
import sys
from pathlib import Path

import os

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from src.llm_client import CachedLLMClient
from src.parser import parse_ranking
from src.prompt_builder import build_prompt
from src.trajectory_builder import load_test_cases, load_user_profiles


def load_checkpoint(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_checkpoint(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def run_condition(
    client: CachedLLMClient,
    model: str,
    granularity: str,
    test_cases: list,
    user_profiles: dict,
    cfg: dict,
    results_dir: Path,
) -> dict:
    """Run one (model, granularity) condition over all test cases."""
    condition_key = f"{model}__{granularity}"
    result_path = results_dir / f"{condition_key}.json"
    checkpoint_path = results_dir / f"{condition_key}_checkpoint.json"

    existing = load_checkpoint(checkpoint_path)
    completed_indices = set(existing.get("completed", []))
    results = existing.get("results", [])

    llm_cfg = cfg["llm"]

    for i, tc in enumerate(tqdm(
        test_cases,
        desc=f"  {model} / {granularity}",
        leave=False,
    )):
        if i in completed_indices:
            continue

        profile = user_profiles.get(tc.user_id)
        messages = build_prompt(tc, granularity, user_profile=profile, city=cfg["dataset"]["city"])

        try:
            response = client.chat(
                model=model,
                messages=messages,
                temperature=llm_cfg["temperature"],
                max_tokens=llm_cfg["max_tokens"],
            )
        except RuntimeError as e:
            print(f"    FAILED test case {i}: {e}")
            results.append({
                "index": i,
                "user_id": tc.user_id,
                "error": str(e),
                "ranking": None,
                "ground_truth_idx": tc.ground_truth_idx,
            })
            completed_indices.add(i)
            continue

        ranking = parse_ranking(response["content"], len(tc.candidates))
        results.append({
            "index": i,
            "user_id": tc.user_id,
            "ranking": ranking,
            "ground_truth_idx": tc.ground_truth_idx,
            "raw_response": response["content"],
            "usage": response.get("usage"),
            "cached": response.get("cached", False),
        })
        completed_indices.add(i)

        if len(completed_indices) % 25 == 0:
            save_checkpoint(
                {"completed": sorted(completed_indices), "results": results},
                checkpoint_path,
            )

    save_checkpoint(
        {"completed": sorted(completed_indices), "results": results},
        checkpoint_path,
    )
    result_path.write_text(json.dumps(results, indent=2))

    # Clean up checkpoint after full completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return {"condition": condition_key, "num_results": len(results)}


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    ds_cfg = cfg["dataset"]
    llm_cfg = cfg["llm"]

    processed_dir = ROOT / ds_cfg["processed_dir"]
    test_cases_path = processed_dir / "test_cases.json"
    profiles_path = processed_dir / "user_profiles.json"

    if not test_cases_path.exists():
        print("ERROR: Preprocessed data not found. Run scripts/preprocess.py first.")
        sys.exit(1)

    print("Loading test cases and user profiles...")
    test_cases = load_test_cases(test_cases_path)
    user_profiles = load_user_profiles(profiles_path)
    print(f"  {len(test_cases)} test cases, {len(user_profiles)} user profiles")

    client = CachedLLMClient(
        cache_dir=ROOT / llm_cfg["cache_dir"],
        max_retries=llm_cfg["max_retries"],
        retry_base_delay=llm_cfg["retry_base_delay"],
        request_timeout=llm_cfg["request_timeout"],
    )

    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    models = cfg["models"]
    levels = cfg["granularity_levels"]
    total = len(models) * len(levels)

    print(f"\nRunning {total} conditions ({len(models)} models x {len(levels)} granularity levels)")
    print(f"  Models: {models}")
    print(f"  Levels: {levels}")
    print(f"  Test cases: {len(test_cases)}")
    print()

    summary = []
    for model in models:
        for level in levels:
            result = run_condition(
                client, model, level, test_cases, user_profiles, cfg, results_dir,
            )
            summary.append(result)
            print(f"  Completed {result['condition']}: {result['num_results']} results")

    summary_path = results_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nExperiment complete. Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
