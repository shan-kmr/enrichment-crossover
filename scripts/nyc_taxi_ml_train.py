#!/usr/bin/env python3
"""Train LightGBM + MLP on NYC taxi tasks for 3 tiers x 5 levels.

Task A (ranking): Acc@1, Acc@5, MRR — grouped by origin zone.
Task B (duration): MAE, RMSE — regression.
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
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.nyc_taxi.features import compute_ranking_features, compute_duration_features

TIERS = ["latlng", "zone_id", "enriched"]


def compute_ranking_metrics(
    origin_preds: dict[int, list[tuple[float, int]]],
) -> dict:
    acc1, acc5, mrr_vals = [], [], []
    for oid, preds in origin_preds.items():
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
        "n_origins": n,
    }


def compute_regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "n_samples": len(y_true),
    }


def load_data(cfg):
    proc_dir = ROOT / cfg["dataset"]["processed_dir"]
    agg = json.loads((proc_dir / "train_aggregates.json").read_text())
    centroids_raw = json.loads((proc_dir / "zone_centroids.json").read_text())
    centroids = {int(k): tuple(v) for k, v in centroids_raw.items()}
    zone_to_cell = json.loads((proc_dir / "zone_to_cell.json").read_text())
    cell_pop = json.loads((proc_dir / "cell_popularity.json").read_text())

    zone_cats_path = ROOT / cfg["enrichment"]["zone_categories_path"]
    zone_cats = json.loads(zone_cats_path.read_text()) if zone_cats_path.exists() else None

    ranking_dir = proc_dir / "ranking"
    rank_train = json.loads((ranking_dir / "train_cases.json").read_text())
    rank_test = json.loads((ranking_dir / "test_cases.json").read_text())

    dur_dir = proc_dir / "duration"
    dur_train = json.loads((dur_dir / "train_cases.json").read_text())
    dur_test = json.loads((dur_dir / "test_cases.json").read_text())

    return agg, centroids, zone_to_cell, cell_pop, zone_cats, \
        rank_train, rank_test, dur_train, dur_test


def run_ranking(cfg, levels, agg, centroids, zone_to_cell, cell_pop, zone_cats,
                train_cases, test_cases, results_dir):
    """Run ranking task for all tiers x levels."""
    for tier in TIERS:
        print(f"\n{'#' * 70}")
        print(f"# RANKING — TIER: {tier}")
        print(f"{'#' * 70}")

        kw = dict(agg=agg, centroids=centroids, zone_to_cell=zone_to_cell)
        if tier == "enriched":
            kw["zone_cats"] = zone_cats
        if tier == "latlng":
            kw["cell_pop"] = cell_pop

        tier_results: dict[str, dict] = {}

        for level in levels:
            print(f"\n  Level: {level}")

            X_train = np.array(
                [compute_ranking_features(c, level, tier, **kw) for c in train_cases],
                dtype=np.float32)
            X_test = np.array(
                [compute_ranking_features(c, level, tier, **kw) for c in test_cases],
                dtype=np.float32)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            y_train = np.array([c["label"] for c in train_cases])
            y_test = np.array([c["label"] for c in test_cases])
            print(f"    Features: {X_train.shape[1]} dims")

            level_results: dict[str, dict] = {}

            # LightGBM
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
            params = {
                "objective": "binary", "metric": "binary_logloss",
                "num_leaves": 31, "learning_rate": 0.05,
                "feature_fraction": 0.9, "verbose": -1,
            }
            bst = lgb.train(
                params, dtrain, num_boost_round=300,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

            probs_lgb = bst.predict(X_test)
            up_lgb: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for c, p in zip(test_cases, probs_lgb):
                up_lgb[c["origin_zone"]].append((float(p), c["label"]))
            m_lgb = compute_ranking_metrics(up_lgb)
            print(f"    LGB:  Acc@1={m_lgb['acc_at_1']:.4f}  "
                  f"Acc@5={m_lgb['acc_at_5']:.4f}  MRR={m_lgb['mrr']:.4f}")
            level_results["lgb"] = m_lgb

            # MLP
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)
            mlp = MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500,
                early_stopping=True, n_iter_no_change=20, random_state=42)
            mlp.fit(X_tr_s, y_train)
            probs_mlp = mlp.predict_proba(X_te_s)[:, 1]

            up_mlp: dict[int, list[tuple[float, int]]] = defaultdict(list)
            for c, p in zip(test_cases, probs_mlp):
                up_mlp[c["origin_zone"]].append((float(p), c["label"]))
            m_mlp = compute_ranking_metrics(up_mlp)
            print(f"    MLP:  Acc@1={m_mlp['acc_at_1']:.4f}  "
                  f"Acc@5={m_mlp['acc_at_5']:.4f}  MRR={m_mlp['mrr']:.4f}")
            level_results["mlp"] = m_mlp

            tier_results[level] = level_results

        out_path = results_dir / "ranking" / f"ml_{tier}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(tier_results, indent=2))
        print(f"\n  Saved → {out_path}")


def run_duration(cfg, levels, agg, centroids, zone_to_cell, cell_pop, zone_cats,
                 train_cases, test_cases, results_dir):
    """Run duration task for all tiers x levels."""
    for tier in TIERS:
        print(f"\n{'#' * 70}")
        print(f"# DURATION — TIER: {tier}")
        print(f"{'#' * 70}")

        kw = dict(agg=agg, centroids=centroids, zone_to_cell=zone_to_cell)
        if tier == "enriched":
            kw["zone_cats"] = zone_cats
        if tier == "latlng":
            kw["cell_pop"] = cell_pop

        tier_results: dict[str, dict] = {}

        for level in levels:
            print(f"\n  Level: {level}")

            X_train = np.array(
                [compute_duration_features(c, level, tier, **kw) for c in train_cases],
                dtype=np.float32)
            X_test = np.array(
                [compute_duration_features(c, level, tier, **kw) for c in test_cases],
                dtype=np.float32)
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

            y_train = np.array([c["duration_min"] for c in train_cases], dtype=np.float32)
            y_test = np.array([c["duration_min"] for c in test_cases], dtype=np.float32)
            print(f"    Features: {X_train.shape[1]} dims")

            level_results: dict[str, dict] = {}

            # LightGBM (regression)
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_test, label=y_test, reference=dtrain)
            params = {
                "objective": "regression", "metric": "mae",
                "num_leaves": 31, "learning_rate": 0.05,
                "feature_fraction": 0.9, "verbose": -1,
            }
            bst = lgb.train(
                params, dtrain, num_boost_round=300,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

            preds_lgb = bst.predict(X_test)
            m_lgb = compute_regression_metrics(y_test, preds_lgb)
            print(f"    LGB:  MAE={m_lgb['mae']:.4f}  RMSE={m_lgb['rmse']:.4f}")
            level_results["lgb"] = m_lgb

            # MLP (regression)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            X_te_s = scaler.transform(X_test)
            mlp = MLPRegressor(
                hidden_layer_sizes=(64, 32), max_iter=500,
                early_stopping=True, n_iter_no_change=20, random_state=42)
            mlp.fit(X_tr_s, y_train)
            preds_mlp = mlp.predict(X_te_s)

            m_mlp = compute_regression_metrics(y_test, preds_mlp)
            print(f"    MLP:  MAE={m_mlp['mae']:.4f}  RMSE={m_mlp['rmse']:.4f}")
            level_results["mlp"] = m_mlp

            tier_results[level] = level_results

        out_path = results_dir / "duration" / f"ml_{tier}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(tier_results, indent=2))
        print(f"\n  Saved → {out_path}")


def main():
    cfg = yaml.safe_load((ROOT / "nyc_taxi_config.yaml").read_text())
    levels = cfg["granularity_levels"]
    results_dir = ROOT / cfg["results_dir"]

    print("Loading data ...")
    agg, centroids, zone_to_cell, cell_pop, zone_cats, \
        rank_train, rank_test, dur_train, dur_test = load_data(cfg)

    print(f"Ranking: {len(rank_train)} train, {len(rank_test)} test")
    print(f"Duration: {len(dur_train)} train, {len(dur_test)} test")

    run_ranking(cfg, levels, agg, centroids, zone_to_cell, cell_pop, zone_cats,
                rank_train, rank_test, results_dir)

    run_duration(cfg, levels, agg, centroids, zone_to_cell, cell_pop, zone_cats,
                 dur_train, dur_test, results_dir)

    print("\n=== All ML experiments done ===")


if __name__ == "__main__":
    main()
