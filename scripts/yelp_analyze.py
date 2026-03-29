#!/usr/bin/env python3
"""Analyze Yelp experiment results: MAE, RMSE, accuracy, plots."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRIC_LABELS = {
    "mae": "MAE (lower is better)",
    "rmse": "RMSE (lower is better)",
    "exact_acc": "Exact Accuracy",
    "within1_acc": "Within-1 Accuracy",
}
GRANULARITY_LABELS = {
    "G0": "G0\nZero-shot",
    "G1": "G1\nCity",
    "G2": "G2\nCategory\nProfile",
    "G3": "G3\nTrajectory\n(Semantic)",
    "G4": "G4\nFull\nSpatiotemporal",
}


def evaluate_condition(results: list[dict]) -> dict:
    preds, actuals = [], []
    for r in results:
        if r.get("predicted") is not None:
            preds.append(r["predicted"])
            actuals.append(r["actual"])

    if not preds:
        return {"mae": None, "rmse": None, "exact_acc": None, "within1_acc": None, "n_valid": 0}

    preds, actuals = np.array(preds), np.array(actuals)
    errors = np.abs(preds - actuals)

    return {
        "mae": float(np.mean(errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "exact_acc": float(np.mean(errors == 0)),
        "within1_acc": float(np.mean(errors <= 1)),
        "n_valid": len(preds),
        "n_parse_failures": len(results) - len(preds),
        "total_tokens": sum(r.get("usage", {}).get("total_tokens", 0) for r in results if r.get("usage")),
    }


def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    models = cfg["models"]
    levels = cfg["granularity_levels"]
    results_dir = ROOT / cfg["results_dir"]
    figures_dir = ROOT / cfg["figures_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load results
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
        print("ERROR: No results found. Run scripts/yelp_experiment.py first.")
        sys.exit(1)

    # Compute metrics
    table = {}
    for model in models:
        table[model] = {}
        for level in levels:
            key = f"{model}__{level}"
            if key in all_results:
                table[model][level] = evaluate_condition(all_results[key])

    # Print table
    for metric_key, metric_name in METRIC_LABELS.items():
        print(f"\n--- {metric_name} ---")
        header = f"{'Model':<20}" + "".join(f"{lvl:>12}" for lvl in levels)
        print(header)
        print("-" * len(header))
        for model in models:
            row = f"{model:<20}"
            for level in levels:
                m = table[model].get(level)
                if m and m[metric_key] is not None:
                    row += f"{m[metric_key]:>12.4f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)

    # Heatmap for MAE (lower is better)
    for metric_key, metric_name in METRIC_LABELS.items():
        data = np.zeros((len(models), len(levels)))
        for i, model in enumerate(models):
            for j, level in enumerate(levels):
                m = table[model].get(level)
                data[i, j] = m[metric_key] if m and m[metric_key] is not None else np.nan

        fig, ax = plt.subplots(figsize=(9, 4))
        cmap = "YlOrRd_r" if "acc" in metric_key else "YlOrRd"
        sns.heatmap(data, annot=True, fmt=".3f", xticklabels=levels, yticklabels=models, cmap=cmap, ax=ax)
        ax.set_title(f"Yelp Rating Prediction: {metric_name}")
        ax.set_xlabel("Location Context Granularity")
        ax.set_ylabel("Model")
        fig.tight_layout()
        fig.savefig(figures_dir / f"yelp_heatmap_{metric_key}.png", dpi=150)
        plt.close(fig)

    # Granularity curves
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (metric_key, metric_name) in zip(axes, METRIC_LABELS.items()):
        for model in models:
            values = []
            for level in levels:
                m = table[model].get(level)
                values.append(m[metric_key] if m and m[metric_key] is not None else np.nan)
            ax.plot(range(len(levels)), values, marker="o", label=model, linewidth=2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([GRANULARITY_LABELS.get(l, l) for l in levels], fontsize=7)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name, fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Yelp Rating Prediction vs. Location Context Granularity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(figures_dir / "yelp_granularity_curves.png", dpi=150)
    plt.close(fig)

    # Save metrics JSON
    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(table, indent=2))

    print(f"\nPlots saved to {figures_dir}/")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
