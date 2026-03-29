#!/usr/bin/env python3
"""Run enriched Gowalla friendship prediction on hard-negative test set.

Hard negatives = same-region non-friends, making the task much harder.
Only runs gpt-5-mini and gpt-5-nano (reasoning models).
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
from src.gowalla.enrichment import load_category_map
from src.gowalla.enriched_prompt_builder import build_prompt


def parse_yes_no(text: str) -> int | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    llm_cfg = cfg["llm"]
    levels = cfg["granularity_levels"]
    models = ["gpt-5-mini", "gpt-5-nano"]

    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded category map: {len(cat_map)} locations")

    test_cases = load_test_cases(ROOT / enr["hard_neg_dir"] / "test_cases.json")
    pos = sum(1 for tc in test_cases if tc.label == 1)
    neg = len(test_cases) - pos
    print(f"Loaded {len(test_cases)} hard-neg test cases ({pos} pos, {neg} neg)")

    results_dir = ROOT / enr["hard_neg_results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    llm = CachedLLMClient(
        cache_dir=ROOT / enr["cache_dir"],
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

            results = []
            for tc in tqdm(test_cases, desc=f"{model} / {level}"):
                messages = build_prompt(tc, level, cat_map)
                resp = llm.chat(
                    model=model,
                    messages=messages,
                    temperature=llm_cfg["temperature"],
                    max_tokens=8192,
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
            print(f"  {model}/{level}: {correct}/{valid} correct "
                  f"({correct/max(valid,1)*100:.1f}%)")

    print("\nDone. Hard-neg results saved to", results_dir)


if __name__ == "__main__":
    main()
