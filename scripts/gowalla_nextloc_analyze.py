#!/usr/bin/env python3
"""Analyze next-location prediction results across tiers and models.

Loads ML results and Llama results, prints comparison tables for
Acc@1, Acc@5, MRR, and identifies crossover patterns between tiers.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TIERS = ["latlng", "venue_id", "enriched"]
METRICS = ["acc_at_1", "acc_at_5", "mrr"]
METRIC_LABELS = {"acc_at_1": "Acc@1", "acc_at_5": "Acc@5", "mrr": "MRR"}


def ranking_from_predictions(predictions: list[dict]) -> dict:
    """Compute ranking metrics from a list of pointwise predictions."""
    user_preds: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p in predictions:
        score = 1.0 if p["prediction"].strip().lower().startswith("yes") \
            else 0.0
        user_preds[p["user_id"]].append((score, p["label"]))

    acc1, acc5, mrr_vals = [], [], []
    for uid, preds in user_preds.items():
        rng = random.Random(uid)
        ranked = sorted(preds, key=lambda x: (-x[0], rng.random()))
        for rank, (_, label) in enumerate(ranked, 1):
            if label == 1:
                acc1.append(float(rank == 1))
                acc5.append(float(rank <= 5))
                mrr_vals.append(1.0 / rank)
                break
    n = len(acc1)
    return {
        "acc_at_1": round(sum(acc1) / n, 4) if n else 0.0,
        "acc_at_5": round(sum(acc5) / n, 4) if n else 0.0,
        "mrr": round(sum(mrr_vals) / n, 4) if n else 0.0,
        "n_users": n,
    }


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:>10.4f}"
    return f"{v!s:>10}"


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    nl = cfg["nextloc"]
    levels = cfg["granularity_levels"]

    results_dir = ROOT / nl["results_dir"]
    llama_dir = ROOT / nl["llama_results_dir"]

    # ── load ML results ─────────────────────────────────────────────
    ml: dict[str, dict] = {}
    for tier in TIERS:
        p = results_dir / f"ml_{tier}.json"
        if p.exists():
            ml[tier] = json.loads(p.read_text())

    # ── load Llama results ──────────────────────────────────────────
    llama: dict[str, dict[str, dict]] = {}
    for tier in TIERS:
        for mode in ("zeroshot", "finetuned"):
            key = f"llama_{mode}_{tier}"
            for level in levels:
                path = llama_dir / f"llama_{mode}_{tier}__{level}.json"
                if path.exists():
                    preds = json.loads(path.read_text())
                    m = ranking_from_predictions(preds)
                    llama.setdefault(key, {})[level] = m

    # ── comparison tables ───────────────────────────────────────────
    for metric in METRICS:
        label = METRIC_LABELS[metric]
        print(f"\n{'=' * 90}")
        print(f"  {label}  (higher is better)")
        print(f"{'=' * 90}")
        hdr = f"{'Method':<35}" + "".join(f"{l:>10}" for l in levels)
        print(hdr)
        print("-" * len(hdr))

        baseline = 1.0 / 20
        print(f"{'Random (1/20)':<35}"
              + "".join(f"{baseline:>10.4f}" for _ in levels))

        for tier in TIERS:
            for model in ("lgb", "mlp"):
                tag = f"{'LGB' if model == 'lgb' else 'MLP'} ({tier})"
                row = f"{tag:<35}"
                for lv in levels:
                    v = ml.get(tier, {}).get(lv, {}).get(model, {}) \
                        .get(metric, "NA")
                    row += _fmt(v)
                print(row)

        print("-" * len(hdr))

        for tier in TIERS:
            for mode in ("zeroshot", "finetuned"):
                key = f"llama_{mode}_{tier}"
                tag = f"Llama {'ZS' if mode == 'zeroshot' else 'FT'} ({tier})"
                row = f"{tag:<35}"
                for lv in levels:
                    v = llama.get(key, {}).get(lv, {}).get(metric, "NA")
                    row += _fmt(v)
                print(row)

    # ── crossover analysis ──────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  CROSSOVER: enriched vs venue_id")
    print(f"{'=' * 90}")

    for metric in METRICS:
        label = METRIC_LABELS[metric]
        print(f"\n  {label}:")

        for model in ("lgb", "mlp"):
            tag = "LGB" if model == "lgb" else "MLP"
            for lv in levels:
                e = (ml.get("enriched", {}).get(lv, {})
                     .get(model, {}).get(metric))
                v = (ml.get("venue_id", {}).get(lv, {})
                     .get(model, {}).get(metric))
                if e is not None and v is not None:
                    w = ("enriched" if e > v
                         else "venue_id" if v > e else "tie")
                    print(f"    {tag} {lv}: enriched={e:.4f}  "
                          f"venue_id={v:.4f}  → {w} ({e - v:+.4f})")

        for mode in ("finetuned",):
            ek = f"llama_{mode}_enriched"
            vk = f"llama_{mode}_venue_id"
            for lv in levels:
                e = llama.get(ek, {}).get(lv, {}).get(metric)
                v = llama.get(vk, {}).get(lv, {}).get(metric)
                if e is not None and v is not None:
                    w = ("enriched" if e > v
                         else "venue_id" if v > e else "tie")
                    print(f"    Llama FT {lv}: enriched={e:.4f}  "
                          f"venue_id={v:.4f}  → {w} ({e - v:+.4f})")

    # ── save summary ────────────────────────────────────────────────
    summary = {"ml": ml, "llama": llama}
    out = results_dir / "nextloc_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary → {out}")


if __name__ == "__main__":
    main()
