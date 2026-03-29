#!/usr/bin/env python3
"""Analyze Gowalla friendship prediction results: Accuracy, F1, AUC."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def compute_metrics(results: list[dict]) -> dict:
    valid = [r for r in results if r["prediction"] is not None]
    if not valid:
        return {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}

    tp = sum(1 for r in valid if r["prediction"] == 1 and r["label"] == 1)
    tn = sum(1 for r in valid if r["prediction"] == 0 and r["label"] == 0)
    fp = sum(1 for r in valid if r["prediction"] == 1 and r["label"] == 0)
    fn = sum(1 for r in valid if r["prediction"] == 0 and r["label"] == 1)

    acc = (tp + tn) / len(valid) if valid else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    return {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "valid": len(valid), "total": len(results),
    }


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    models = cfg["models"]
    levels = cfg["granularity_levels"]
    results_dir = ROOT / cfg["results_dir"]

    all_metrics: dict[str, dict[str, dict]] = {}

    for model in models:
        all_metrics[model] = {}
        for level in levels:
            path = results_dir / f"{model}__{level}.json"
            if not path.exists():
                all_metrics[model][level] = {"accuracy": "NA", "f1": "NA"}
                continue
            results = json.loads(path.read_text())
            all_metrics[model][level] = compute_metrics(results)

    print("\n" + "=" * 90)
    print("GOWALLA FRIENDSHIP PREDICTION: LLM DIRECT PROMPTING")
    print("=" * 90)

    for metric in ["accuracy", "f1"]:
        label = "Accuracy" if metric == "accuracy" else "F1 Score"
        print(f"\n--- {label} (higher is better) ---")
        header = f"{'Method':<30}" + "".join(f"{l:>10}" for l in levels)
        print(header)
        print("-" * len(header))
        print(f"{'Random':<30}" + "".join(f"{'0.5000':>10}" for _ in levels))
        for model in models:
            row = f"{model:<30}"
            for level in levels:
                v = all_metrics[model][level].get(metric, "NA")
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

    print("\n" + "=" * 90)

    summary_path = results_dir / "llm_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
