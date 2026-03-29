#!/usr/bin/env python3
"""Few-shot enriched Gowalla friendship prediction (10 in-context examples).

Prepends 10 labeled examples (5 friends, 5 non-friends) from the train set
to each test prompt.  Only runs gpt-5-mini and gpt-5-nano.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from src.llm_client import CachedLLMClient
from src.gowalla.data_loader import load_test_cases, FriendshipTestCase
from src.gowalla.enrichment import load_category_map
from src.gowalla.enriched_prompt_builder import build_prompt, SYSTEM


def parse_yes_no(text: str) -> int | None:
    t = text.strip().lower()
    if t.startswith("yes"):
        return 1
    if t.startswith("no"):
        return 0
    return None


def _select_fewshot_examples(
    train_cases: list[FriendshipTestCase], n: int, seed: int = 42
) -> list[FriendshipTestCase]:
    """Select n/2 positive + n/2 negative examples from train set."""
    rng = random.Random(seed)
    pos = [tc for tc in train_cases if tc.label == 1]
    neg = [tc for tc in train_cases if tc.label == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    selected = pos[:half] + neg[:half]
    rng.shuffle(selected)
    return selected


def build_fewshot_prompt(
    test_tc: FriendshipTestCase,
    examples: list[FriendshipTestCase],
    granularity: str,
    cat_map: dict,
) -> list[dict[str, str]]:
    """Build a prompt with in-context examples followed by the test case."""
    # Each example is formatted at the same granularity as the test case
    parts: list[str] = []
    parts.append(f"Here are {len(examples)} labeled examples:\n")

    for i, ex in enumerate(examples, 1):
        ex_prompt = build_prompt(ex, granularity, cat_map)
        ex_body = ex_prompt[1]["content"]
        # Strip the final question from the example body
        ex_body = ex_body.rsplit("\n\nAre these two users friends?", 1)[0]
        answer = "Yes" if ex.label == 1 else "No"
        parts.append(f"--- Example {i} ---\n{ex_body}\nAnswer: {answer}\n")

    # Now append the actual test case
    test_prompt = build_prompt(test_tc, granularity, cat_map)
    test_body = test_prompt[1]["content"]

    parts.append("--- Now predict for this new pair ---")
    parts.append(test_body)

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    llm_cfg = cfg["llm"]
    levels = cfg["granularity_levels"]
    models = ["gpt-5-mini", "gpt-5-nano"]
    n_examples = enr.get("fewshot_n_examples", 10)

    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded category map: {len(cat_map)} locations")

    filt_dir = ROOT / enr["filtered_dir"]
    test_cases = load_test_cases(filt_dir / "test_cases.json")
    train_cases = load_test_cases(filt_dir / "train_cases.json")
    print(f"Loaded {len(test_cases)} test, {len(train_cases)} train cases")

    examples = _select_fewshot_examples(train_cases, n_examples)
    pos_ex = sum(1 for e in examples if e.label == 1)
    print(f"Selected {len(examples)} few-shot examples ({pos_ex} pos, "
          f"{len(examples)-pos_ex} neg)")

    results_dir = ROOT / enr["fewshot_results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    llm = CachedLLMClient(
        cache_dir=ROOT / enr["cache_dir"],
        max_retries=llm_cfg["max_retries"],
        retry_base_delay=10.0,
        request_timeout=300.0,
    )

    for model in models:
        for level in levels:
            out_path = results_dir / f"{model}__{level}.json"
            if out_path.exists():
                print(f"{model}/{level}: exists, skipping")
                continue

            results = []
            for tc in tqdm(test_cases, desc=f"{model} / {level} (few-shot)"):
                messages = build_fewshot_prompt(tc, examples, level, cat_map)
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

    print("\nDone. Few-shot results saved to", results_dir)


if __name__ == "__main__":
    main()
