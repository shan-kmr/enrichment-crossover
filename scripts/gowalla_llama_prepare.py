#!/usr/bin/env python3
"""Prepare JSONL files for Llama 3.1 8B experiments on Google Colab.

Generates 15 JSONL files:
  - test_G0..G4.jsonl         (200 test prompts, no assistant answer)
  - test_fewshot_G0..G4.jsonl (200 test prompts with 2 in-context examples)
  - train_G0..G4.jsonl        (20K train prompts with assistant answer for SFT)

All prompts use the enriched prompt builder (venue categories included).
Run locally before uploading to Google Drive for Colab.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gowalla.data_loader import load_test_cases, FriendshipTestCase
from src.gowalla.enrichment import load_category_map
from src.gowalla.enriched_prompt_builder import build_prompt, SYSTEM


def select_fewshot_examples(cases, n, seed=42):
    rng = random.Random(seed)
    pos = [tc for tc in cases if tc.label == 1]
    neg = [tc for tc in cases if tc.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    selected = pos[:half] + neg[:half]
    rng.shuffle(selected)
    return selected


def build_fewshot_prompt(test_tc, examples, granularity, cat_map):
    parts = [f"Here are {len(examples)} labeled examples:\n"]
    for i, ex in enumerate(examples, 1):
        ex_prompt = build_prompt(ex, granularity, cat_map)
        ex_body = ex_prompt[1]["content"]
        ex_body = ex_body.rsplit("\n\nAre these two users friends?", 1)[0]
        answer = "Yes" if ex.label == 1 else "No"
        parts.append(f"--- Example {i} ---\n{ex_body}\nAnswer: {answer}\n")
    test_prompt = build_prompt(test_tc, granularity, cat_map)
    parts.append("--- Now predict for this new pair ---")
    parts.append(test_prompt[1]["content"])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    levels = cfg["granularity_levels"]

    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded category map: {len(cat_map)} locations")

    filt_dir = ROOT / enr["filtered_dir"]
    test_cases = load_test_cases(filt_dir / "test_cases.json")
    train_cases_small = load_test_cases(filt_dir / "train_cases.json")
    ml_train_cases = load_test_cases(filt_dir / "ml_train_cases.json")
    print(f"Loaded {len(test_cases)} test, {len(train_cases_small)} LLM train, "
          f"{len(ml_train_cases)} ML train cases")

    fewshot_examples = select_fewshot_examples(train_cases_small, 2)
    pos_ex = sum(1 for e in fewshot_examples if e.label == 1)
    print(f"Selected {len(fewshot_examples)} few-shot examples ({pos_ex} pos, "
          f"{len(fewshot_examples)-pos_ex} neg)")

    out_dir = ROOT / "data" / "processed" / "gowalla" / "llama_jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    for level in levels:
        # --- Test (zero-shot): system + user, no assistant ---
        test_path = out_dir / f"test_{level}.jsonl"
        with open(test_path, "w") as f:
            for tc in tqdm(test_cases, desc=f"test_{level}"):
                msgs = build_prompt(tc, level, cat_map)
                line = json.dumps({
                    "messages": msgs,
                    "label": tc.label,
                })
                f.write(line + "\n")
        print(f"  {test_path.name}: {len(test_cases)} examples")

        # --- Test few-shot: system + user (with examples prepended), no assistant ---
        fs_path = out_dir / f"test_fewshot_{level}.jsonl"
        with open(fs_path, "w") as f:
            for tc in tqdm(test_cases, desc=f"test_fewshot_{level}"):
                msgs = build_fewshot_prompt(tc, fewshot_examples, level, cat_map)
                line = json.dumps({
                    "messages": msgs,
                    "label": tc.label,
                })
                f.write(line + "\n")
        print(f"  {fs_path.name}: {len(test_cases)} examples")

        # --- Train (SFT): system + user + assistant ---
        train_path = out_dir / f"train_{level}.jsonl"
        with open(train_path, "w") as f:
            for tc in tqdm(ml_train_cases, desc=f"train_{level}"):
                msgs = build_prompt(tc, level, cat_map)
                answer = "Yes" if tc.label == 1 else "No"
                msgs.append({"role": "assistant", "content": answer})
                line = json.dumps({"messages": msgs})
                f.write(line + "\n")
        size_mb = train_path.stat().st_size / (1024 * 1024)
        print(f"  {train_path.name}: {len(ml_train_cases)} examples ({size_mb:.1f} MB)")

    print(f"\nAll JSONL files saved to {out_dir}/")
    print("Upload this folder to Google Drive, then run the Colab notebook.")


if __name__ == "__main__":
    main()
