#!/usr/bin/env python3
"""Train LightGBM + MLP on next-location prediction for 3 tiers x 5 levels.

Evaluation uses ranking metrics: Acc@1, Acc@5, MRR.
Each user has 20 candidates (1 positive, 19 negative); predictions are
grouped by user_id and ranked by predicted probability.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.gowalla.data_loader import load_nextloc_cases
from src.gowalla.nextloc_features import compute_nextloc_handcrafted
from src.gowalla.enrichment import load_category_map

TIERS = ["latlng", "venue_id", "enriched"]


def compute_ranking_metrics(
    user_preds: dict[int, list[tuple[float, int]]],
) -> dict:
    acc1, acc5, mrr_vals = [], [], []
    for uid, preds in user_preds.items():
        ranked = sorted(preds, key=lambda x: -x[0])
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


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    nl = cfg["nextloc"]
    levels = cfg["granularity_levels"]

    data_dir = ROOT / nl["data_dir"]
    train_cases = load_nextloc_cases(data_dir / "train_cases.json")
    test_cases = load_nextloc_cases(data_dir / "test_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test cases")

    cell_pop = json.loads((data_dir / "cell_popularity.json").read_text())
    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"cell_pop: {len(cell_pop)} cells | "
          f"cat_map: {len(cat_map)} locations")

    results_dir = ROOT / nl["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    train_labels = np.array([tc.label for tc in train_cases])
    test_labels = np.array([tc.label for tc in test_cases])

    for tier in TIERS:
        print(f"\n{'#' * 70}")
        print(f"# TIER: {tier}")
        print(f"{'#' * 70}")

        kw: dict = {}
        if tier == "enriched":
            kw["cat_map"] = cat_map
        if tier == "latlng":
            kw["cell_pop"] = cell_pop

        tier_results: dict[str, dict] = {}

        for level in levels:
            print(f"\n  Level: {level}")

            X_train = np.array(
                [compute_nextloc_handcrafted(tc, level, tier, **kw)
                 for tc in train_cases],
                dtype=np.float32)
            X_test = np.array(
                [compute_nextloc_handcrafted(tc, level, tier, **kw)
                 for tc in test_cases],
                dtype=np.float32)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0,
                                    neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0,
                                   neginf=0.0)
            print(f"    Features: {X_train.shape[1]} dims")

            level_results: dict[str, dict] = {}

            # ── LightGBM ───────────────────────────────────────────
            dtrain = lgb.Dataset(X_train, label=train_labels)
            dval = lgb.Dataset(X_test, label=test_labels,
                               reference=dtrain)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "verbose": -1,
            }
            bst = lgb.train(
                params, dtrain, num_boost_round=300,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

            probs_lgb = bst.predict(X_test)
            up_lgb: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for tc, p in zip(test_cases, probs_lgb):
                up_lgb[tc.user.user_id].append((float(p), tc.label))
            m_lgb = compute_ranking_metrics(up_lgb)
            print(f"    LGB:  Acc@1={m_lgb['acc_at_1']:.4f}  "
                  f"Acc@5={m_lgb['acc_at_5']:.4f}  MRR={m_lgb['mrr']:.4f}")
            level_results["lgb"] = m_lgb

            # ── MLP ────────────────────────────────────────────────
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)
            mlp = MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=500,
                early_stopping=True, n_iter_no_change=20, random_state=42)
            mlp.fit(X_tr_s, train_labels)
            probs_mlp = mlp.predict_proba(X_te_s)[:, 1]

            up_mlp: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for tc, p in zip(test_cases, probs_mlp):
                up_mlp[tc.user.user_id].append((float(p), tc.label))
            m_mlp = compute_ranking_metrics(up_mlp)
            print(f"    MLP:  Acc@1={m_mlp['acc_at_1']:.4f}  "
                  f"Acc@5={m_mlp['acc_at_5']:.4f}  MRR={m_mlp['mrr']:.4f}")
            level_results["mlp"] = m_mlp

            tier_results[level] = level_results

        out_path = results_dir / f"ml_{tier}.json"
        out_path.write_text(json.dumps(tier_results, indent=2))
        print(f"\n  Saved → {out_path}")


if __name__ == "__main__":
    main()
