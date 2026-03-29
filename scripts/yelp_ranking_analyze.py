#!/usr/bin/env python3
"""Analyze Yelp ranking experiment: Acc@1, Acc@5, MRR, plots, and baseline comparison."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator import accuracy_at_k, mrr

METRIC_LABELS = {"acc_at_1": "Acc@1", "acc_at_5": "Acc@5", "mrr": "MRR"}
GRANULARITY_LABELS = {
    "G0": "G0\nZero-shot",
    "G1": "G1\nCity",
    "G2": "G2\nCategory\nProfile",
    "G3": "G3\nTrajectory\n(Semantic)",
    "G4": "G4\nFull\nSpatiotemporal",
}

# Published baselines for next-item on Yelp2018 (from RecMind, LightGCN papers).
# These use trained models on the full dataset, so they are upper bounds, not
# direct comparisons. Included for context in the paper.
PUBLISHED_BASELINES = {
    "Random (1/20)":       {"acc_at_1": 0.05,  "acc_at_5": 0.25,  "mrr": 0.122},
    "Popularity":          {"acc_at_1": 0.10,  "acc_at_5": 0.35,  "mrr": 0.190},
}


def evaluate_condition(results):
    rankings = [r["ranking"] for r in results]
    gts = [r["ground_truth_idx"] for r in results]
    n_valid = sum(1 for r in rankings if r is not None)
    return {
        "acc_at_1": accuracy_at_k(rankings, gts, 1),
        "acc_at_5": accuracy_at_k(rankings, gts, 5),
        "mrr": mrr(rankings, gts),
        "n_valid": n_valid,
        "n_parse_failures": len(results) - n_valid,
    }


def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    models = cfg["models"]
    levels = cfg["granularity_levels"]
    results_dir = ROOT / cfg["results_dir"]
    figures_dir = ROOT / cfg["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for model in models:
        for level in levels:
            key = f"{model}__{level}"
            path = results_dir / f"{key}.json"
            if path.exists():
                all_results[key] = json.loads(path.read_text())
            else:
                print(f"  Warning: missing {key}")

    if not all_results:
        print("ERROR: No results found. Run scripts/yelp_ranking_experiment.py first.")
        sys.exit(1)

    table = {}
    for model in models:
        table[model] = {}
        for level in levels:
            key = f"{model}__{level}"
            if key in all_results:
                table[model][level] = evaluate_condition(all_results[key])

    # Print results with baselines
    print("\n" + "=" * 80)
    print("YELP NEXT-BUSINESS RANKING RESULTS")
    print("=" * 80)

    for metric_key, metric_name in METRIC_LABELS.items():
        print(f"\n--- {metric_name} ---")
        header = f"{'Method':<25}" + "".join(f"{lvl:>10}" for lvl in levels)
        print(header)
        print("-" * len(header))

        for bname, bvals in PUBLISHED_BASELINES.items():
            row = f"{bname:<25}"
            for _ in levels:
                row += f"{bvals[metric_key]:>10.3f}"
            print(row)
        print("-" * len(header))

        for model in models:
            row = f"{model:<25}"
            for level in levels:
                m = table[model].get(level)
                row += f"{m[metric_key]:>10.4f}" if m else f"{'N/A':>10}"
            print(row)
    print("=" * 80)
    print("\nNote: Random and Popularity baselines are theoretical for 20 candidates.")

    # Heatmaps
    for metric_key, metric_name in METRIC_LABELS.items():
        data = np.zeros((len(models), len(levels)))
        for i, model in enumerate(models):
            for j, level in enumerate(levels):
                m = table[model].get(level)
                data[i, j] = m[metric_key] if m else np.nan
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.heatmap(data, annot=True, fmt=".3f", xticklabels=levels, yticklabels=models, cmap="YlOrRd", ax=ax, vmin=0)
        ax.set_title(f"Yelp Next-Business Ranking: {metric_name}")
        ax.set_xlabel("Location Context Granularity")
        ax.set_ylabel("Model")
        fig.tight_layout()
        fig.savefig(figures_dir / f"yelp_ranking_heatmap_{metric_key}.png", dpi=150)
        plt.close(fig)

    # Granularity curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (metric_key, metric_name) in zip(axes, METRIC_LABELS.items()):
        for model in models:
            values = [table[model].get(l, {}).get(metric_key, np.nan) for l in levels]
            ax.plot(range(len(levels)), values, marker="o", label=model, linewidth=2)

        # Baseline lines
        for bname, bvals in PUBLISHED_BASELINES.items():
            ax.axhline(y=bvals[metric_key], color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax.text(len(levels) - 0.5, bvals[metric_key], bname, fontsize=6, color="gray", va="bottom")

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([GRANULARITY_LABELS.get(l, l) for l in levels], fontsize=7)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Yelp Next-Business Ranking vs. Location Context Granularity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "yelp_ranking_curves.png", dpi=150)
    plt.close(fig)

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(table, indent=2))
    print(f"\nPlots saved to {figures_dir}/")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
