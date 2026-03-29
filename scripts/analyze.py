#!/usr/bin/env python3
"""Analyze experiment results: compute metrics, generate tables and plots."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator import accuracy_at_k, bootstrap_ci, evaluate_condition, mrr

METRIC_LABELS = {
    "acc_at_1": "Acc@1",
    "acc_at_5": "Acc@5",
    "mrr": "MRR",
}
GRANULARITY_LABELS = {
    "G0": "G0\nNo Context",
    "G1": "G1\nCity",
    "G2": "G2\nCategory\nProfile",
    "G3": "G3\nTrajectory\n(Semantic)",
    "G4": "G4\nFull\nSpatiotemporal",
}


def load_all_results(results_dir: Path, models: list[str], levels: list[str]) -> dict:
    all_results = {}
    for model in models:
        for level in levels:
            key = f"{model}__{level}"
            path = results_dir / f"{key}.json"
            if path.exists():
                all_results[key] = json.loads(path.read_text())
            else:
                print(f"  Warning: missing results for {key}")
    return all_results


def build_metrics_table(all_results: dict, models: list[str], levels: list[str]) -> dict:
    table = {}
    for model in models:
        table[model] = {}
        for level in levels:
            key = f"{model}__{level}"
            if key in all_results:
                table[model][level] = evaluate_condition(all_results[key])
            else:
                table[model][level] = None
    return table


def print_metrics_table(table: dict, models: list[str], levels: list[str]) -> None:
    header = f"{'Model':<20}" + "".join(f"{'  ' + lvl + '':>12}" for lvl in levels)
    print("\n" + "=" * len(header))
    for metric_key, metric_name in METRIC_LABELS.items():
        print(f"\n--- {metric_name} ---")
        print(header)
        print("-" * len(header))
        for model in models:
            row = f"{model:<20}"
            for level in levels:
                m = table[model].get(level)
                if m:
                    row += f"{m[metric_key]:>12.4f}"
                else:
                    row += f"{'N/A':>12}"
            print(row)
    print("=" * len(header))


def plot_heatmaps(table: dict, models: list[str], levels: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric_key, metric_name in METRIC_LABELS.items():
        data = np.zeros((len(models), len(levels)))
        for i, model in enumerate(models):
            for j, level in enumerate(levels):
                m = table[model].get(level)
                data[i, j] = m[metric_key] if m else np.nan

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(
            data,
            annot=True,
            fmt=".3f",
            xticklabels=levels,
            yticklabels=models,
            cmap="YlOrRd",
            ax=ax,
            vmin=0,
        )
        ax.set_title(f"{metric_name} by Model and Granularity Level")
        ax.set_xlabel("Location Context Granularity")
        ax.set_ylabel("Model")
        fig.tight_layout()
        fig.savefig(out_dir / f"heatmap_{metric_key}.png", dpi=150)
        plt.close(fig)
        print(f"  Saved {out_dir / f'heatmap_{metric_key}.png'}")


def plot_granularity_curves(table: dict, models: list[str], levels: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (metric_key, metric_name) in zip(axes, METRIC_LABELS.items()):
        for model in models:
            values = []
            for level in levels:
                m = table[model].get(level)
                values.append(m[metric_key] if m else np.nan)
            ax.plot(range(len(levels)), values, marker="o", label=model, linewidth=2)

        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([GRANULARITY_LABELS.get(l, l) for l in levels], fontsize=8)
        ax.set_ylabel(metric_name)
        ax.set_title(metric_name)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Performance vs. Location Context Granularity", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "granularity_curves.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'granularity_curves.png'}")


def plot_marginal_gains(table: dict, models: list[str], levels: list[str], out_dir: Path) -> None:
    """Bar chart of marginal Acc@1 improvement per granularity step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(levels) - 1)
    width = 0.8 / len(models)
    step_labels = [f"{levels[i]} -> {levels[i+1]}" for i in range(len(levels) - 1)]

    for idx, model in enumerate(models):
        deltas = []
        for i in range(len(levels) - 1):
            m_prev = table[model].get(levels[i])
            m_next = table[model].get(levels[i + 1])
            if m_prev and m_next:
                deltas.append(m_next["acc_at_1"] - m_prev["acc_at_1"])
            else:
                deltas.append(0.0)
        ax.bar(x + idx * width, deltas, width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(step_labels, fontsize=9)
    ax.set_ylabel("Marginal Acc@1 Improvement")
    ax.set_title("Marginal Improvement per Granularity Step")
    ax.legend(fontsize=8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "marginal_gains.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {out_dir / 'marginal_gains.png'}")


def compute_confidence_intervals(all_results: dict, models: list[str], levels: list[str]) -> dict:
    """Compute 95% bootstrap CIs for Acc@1."""
    cis = {}
    for model in models:
        cis[model] = {}
        for level in levels:
            key = f"{model}__{level}"
            if key not in all_results:
                continue
            results = all_results[key]
            rankings = [r["ranking"] for r in results]
            gts = [r["ground_truth_idx"] for r in results]
            lo, hi = bootstrap_ci(
                rankings, gts,
                lambda r, g: accuracy_at_k(r, g, 1),
            )
            cis[model][level] = {"ci_lower": lo, "ci_upper": hi}
    return cis


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    models = cfg["models"]
    levels = cfg["granularity_levels"]
    results_dir = ROOT / cfg["results_dir"]
    figures_dir = ROOT / cfg["figures_dir"]

    print("Loading results...")
    all_results = load_all_results(results_dir, models, levels)
    if not all_results:
        print("ERROR: No results found. Run scripts/run_experiment.py first.")
        sys.exit(1)

    print(f"  Loaded {len(all_results)} conditions")

    print("\nComputing metrics...")
    table = build_metrics_table(all_results, models, levels)
    print_metrics_table(table, models, levels)

    print("\nComputing 95% bootstrap confidence intervals for Acc@1...")
    cis = compute_confidence_intervals(all_results, models, levels)
    for model in models:
        for level in levels:
            if level in cis.get(model, {}):
                ci = cis[model][level]
                acc = table[model][level]["acc_at_1"] if table[model].get(level) else 0
                print(f"  {model} / {level}: {acc:.4f} [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    print("\nGenerating plots...")
    plot_heatmaps(table, models, levels, figures_dir)
    plot_granularity_curves(table, models, levels, figures_dir)
    plot_marginal_gains(table, models, levels, figures_dir)

    # Save full metrics as JSON
    metrics_out = {}
    for model in models:
        metrics_out[model] = {}
        for level in levels:
            entry = table[model].get(level)
            if entry:
                entry_with_ci = {**entry}
                if level in cis.get(model, {}):
                    entry_with_ci["acc_at_1_ci"] = cis[model][level]
                metrics_out[model][level] = entry_with_ci

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))
    print(f"\nFull metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
