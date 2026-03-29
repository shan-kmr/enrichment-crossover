#!/usr/bin/env python3
"""Poll fine-tuning jobs and evaluate fine-tuned models.

Usage:
  python scripts/gowalla_finetune_evaluate.py --status    # Check job statuses
  python scripts/gowalla_finetune_evaluate.py --poll      # Poll until all done, then evaluate
  python scripts/gowalla_finetune_evaluate.py --evaluate   # Run evaluation on completed jobs
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from openai import OpenAI

from src.gowalla.data_loader import load_test_cases
from src.gowalla.enrichment import load_category_map
from src.gowalla.enriched_prompt_builder import build_prompt


def parse_yes_no(text: str) -> int | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def check_status(client: OpenAI, jobs_meta: dict) -> bool:
    """Returns True if all jobs are in a terminal state."""
    all_done = True
    for key, meta in jobs_meta.items():
        if "job_id" not in meta:
            continue
        job = client.fine_tuning.jobs.retrieve(meta["job_id"])
        meta["status"] = job.status
        if job.fine_tuned_model:
            meta["fine_tuned_model"] = job.fine_tuned_model
        suffix = f" -> {job.fine_tuned_model}" if job.fine_tuned_model else ""
        print(f"  {key}: {job.status}{suffix}")
        if job.status not in ("succeeded", "failed", "cancelled"):
            all_done = False
    return all_done


def evaluate(client: OpenAI, jobs_meta: dict, test_cases, cat_map, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)

    for key, meta in jobs_meta.items():
        if meta.get("status") != "succeeded" or "fine_tuned_model" not in meta:
            print(f"  {key}: skipping (status={meta.get('status')})")
            continue

        ft_model = meta["fine_tuned_model"]
        level = meta["granularity"]
        out_path = results_dir / f"{key}.json"

        if out_path.exists():
            print(f"  {key}: results exist, skipping")
            continue

        print(f"  {key}: evaluating {ft_model} on {len(test_cases)} test cases...")
        results = []
        correct = 0
        for tc in tqdm(test_cases, desc=key):
            messages = build_prompt(tc, level, cat_map)
            try:
                response = client.chat.completions.create(
                    model=ft_model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=64,
                    timeout=60.0,
                )
                content = response.choices[0].message.content
                pred = parse_yes_no(content or "")
                if pred == tc.label:
                    correct += 1
                results.append({
                    "label": tc.label,
                    "prediction": pred,
                    "raw": content,
                    "model": response.model,
                })
            except Exception as e:
                print(f"    Error: {e}")
                results.append({
                    "label": tc.label,
                    "prediction": None,
                    "raw": str(e),
                    "model": ft_model,
                })

        out_path.write_text(json.dumps(results, indent=2))
        valid = sum(1 for r in results if r["prediction"] is not None)
        print(f"    {correct}/{valid} correct ({correct/max(valid,1)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Check job statuses")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate completed jobs")
    parser.add_argument("--poll", action="store_true",
                        help="Poll until all done, then evaluate")
    args = parser.parse_args()

    if not any([args.status, args.evaluate, args.poll]):
        args.status = True

    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]

    enr_results_dir = ROOT / enr["results_dir"]
    meta_path = enr_results_dir / "finetune_jobs.json"

    if not meta_path.exists():
        print("No finetune_jobs.json found. Run gowalla_finetune_prepare.py first.")
        return

    jobs_meta = json.loads(meta_path.read_text())
    client = OpenAI()

    if args.poll:
        print("Polling fine-tuning jobs until all complete...")
        while True:
            done = check_status(client, jobs_meta)
            meta_path.write_text(json.dumps(jobs_meta, indent=2))
            if done:
                print("\nAll jobs complete.")
                break
            print("  Waiting 60s...\n")
            time.sleep(60)
        args.evaluate = True

    elif args.status:
        print("Checking fine-tuning job statuses...")
        check_status(client, jobs_meta)
        meta_path.write_text(json.dumps(jobs_meta, indent=2))

    if args.evaluate:
        cat_map = load_category_map(ROOT / enr["categories_path"])
        filt_dir = ROOT / enr["filtered_dir"]
        test_cases = load_test_cases(filt_dir / "test_cases.json")
        results_dir = enr_results_dir / "finetune"
        print(f"\nEvaluating on {len(test_cases)} test cases...")
        evaluate(client, jobs_meta, test_cases, cat_map, results_dir)

    meta_path.write_text(json.dumps(jobs_meta, indent=2))
    print("\nDone.")


if __name__ == "__main__":
    main()
