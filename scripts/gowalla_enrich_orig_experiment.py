#!/usr/bin/env python3
"""Run ORIGINAL (non-enriched) prompts on filtered test cases for baseline comparison.

Uses the same filtered test pairs as the enriched experiment, but with the
original prompt_builder (raw coordinates, no venue categories).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from src.llm_client import CachedLLMClient
from src.gowalla.data_loader import load_test_cases
from src.gowalla.prompt_builder import build_prompt  # ORIGINAL prompt builder


def parse_yes_no(text: str) -> int | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    ds = cfg["dataset"]
    enr = cfg["enrichment"]
    llm_cfg = cfg["llm"]
    models = cfg["models"]
    levels = cfg["granularity_levels"]

    filt_dir = ROOT / enr["filtered_dir"]
    test_cases = load_test_cases(filt_dir / "test_cases.json")
    print(f"Loaded {len(test_cases)} filtered test cases")

    results_dir = ROOT / enr["results_dir"] / "original_baseline"
    results_dir.mkdir(parents=True, exist_ok=True)

    llm = CachedLLMClient(
        cache_dir=ROOT / llm_cfg["cache_dir"],
        max_retries=llm_cfg["max_retries"],
        retry_base_delay=llm_cfg["retry_base_delay"],
        request_timeout=llm_cfg["request_timeout"],
    )

    for model in models:
        for level in levels:
            out_path = results_dir / f"{model}__{level}.json"
            if out_path.exists():
                print(f"{model}/{level}: exists, skipping")
                continue

            max_tokens = llm_cfg["max_tokens"]
            is_reasoning = "gpt-5" in model or "o3" in model or "o4" in model
            if is_reasoning:
                max_tokens = 8192

            results = []
            for tc in tqdm(test_cases, desc=f"{model} / {level} (orig)"):
                messages = build_prompt(tc, level)
                resp = llm.chat(
                    model=model,
                    messages=messages,
                    temperature=llm_cfg["temperature"],
                    max_tokens=max_tokens,
                )
                pred = parse_yes_no(resp["content"] or "")
                results.append({
                    "label": tc.label,
                    "prediction": pred,
                    "raw": resp["content"],
                    "model": resp.get("model", model),
                    "cached": resp.get("cached", False),
                })

            out_path.write_text(json.dumps(results, indent=2))
            valid = sum(1 for r in results if r["prediction"] is not None)
            correct = sum(1 for r in results if r["prediction"] == r["label"])
            print(f"  {model}/{level}: {correct}/{valid} correct ({correct/max(valid,1)*100:.1f}%)")

    print("\nDone. Original baseline results saved.")


if __name__ == "__main__":
    main()
