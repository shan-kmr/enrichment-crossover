#!/usr/bin/env python3
"""Compare LogisticRegression vs LightGBM on enriched vs venue_id features
for Gowalla friendship prediction, testing whether enrichment's benefit
requires non-linear modeling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.gowalla.data_loader import FriendshipTestCase, load_test_cases
from src.gowalla.enrichment import load_category_map, compute_enriched_handcrafted


def compute_handcrafted(tc: FriendshipTestCase, granularity: str) -> list[float]:
    a, b = tc.user_a, tc.user_b
    feats = [
        float(a.total_checkins),
        float(b.total_checkins),
        float(a.total_checkins + b.total_checkins),
    ]
    if granularity in ("G1", "G2", "G3", "G4"):
        feats.extend([
            float(tc.same_region),
            tc.centroid_distance_km,
            float(a.primary_region == b.primary_region),
        ])
    if granularity in ("G2", "G3", "G4"):
        feats.extend([
            float(a.unique_locations),
            float(b.unique_locations),
            float(tc.shared_locations),
            tc.jaccard_locations,
        ])
    if granularity in ("G3", "G4"):
        feats.extend([
            a.geo_spread_km,
            b.geo_spread_km,
            float(a.active_days),
            float(b.active_days),
        ])
    if granularity == "G4":
        feats.extend([
            float(tc.temporal_co_occurrences),
            tc.centroid_distance_km / max(a.geo_spread_km + b.geo_spread_km, 0.01),
            tc.jaccard_locations * float(tc.temporal_co_occurrences + 1),
        ])
    return feats


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_pred == y_true).mean())


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    levels = cfg["granularity_levels"]

    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded category map: {len(cat_map)} locations")

    filt_dir = ROOT / enr["filtered_dir"]
    ml_train = filt_dir / "ml_train_cases.json"
    ml_test = filt_dir / "ml_test_cases.json"
    if ml_train.exists() and ml_test.exists():
        train_cases = load_test_cases(ml_train)
        test_cases = load_test_cases(ml_test)
    else:
        train_cases = load_test_cases(filt_dir / "train_cases.json")
        test_cases = load_test_cases(filt_dir / "test_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test")

    train_labels = np.array([tc.label for tc in train_cases])
    test_labels = np.array([tc.label for tc in test_cases])

    results: dict[str, dict] = {}

    for level in levels:
        print(f"\n{'='*60}")
        print(f"Level: {level}")
        print(f"{'='*60}")

        X_tr_vid = np.array([compute_handcrafted(tc, level) for tc in train_cases], dtype=np.float32)
        X_te_vid = np.array([compute_handcrafted(tc, level) for tc in test_cases], dtype=np.float32)
        X_tr_enr = np.array([compute_enriched_handcrafted(tc, level, cat_map) for tc in train_cases], dtype=np.float32)
        X_te_enr = np.array([compute_enriched_handcrafted(tc, level, cat_map) for tc in test_cases], dtype=np.float32)
        print(f"  venue_id dims: {X_tr_vid.shape[1]}, enriched dims: {X_tr_enr.shape[1]}")

        # LightGBM on venue_id
        params = {"objective": "binary", "metric": "binary_logloss",
                  "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.9, "verbose": -1}
        bst_vid = lgb.train(params, lgb.Dataset(X_tr_vid, label=train_labels), num_boost_round=300,
                            valid_sets=[lgb.Dataset(X_te_vid, label=test_labels)],
                            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        acc_lgb_vid = accuracy(test_labels, (bst_vid.predict(X_te_vid) > 0.5).astype(int))

        # LightGBM on enriched
        bst_enr = lgb.train(params, lgb.Dataset(X_tr_enr, label=train_labels), num_boost_round=300,
                            valid_sets=[lgb.Dataset(X_te_enr, label=test_labels)],
                            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
        acc_lgb_enr = accuracy(test_labels, (bst_enr.predict(X_te_enr) > 0.5).astype(int))

        # LogReg on venue_id
        scaler_vid = StandardScaler()
        lr_vid = LogisticRegression(max_iter=1000, random_state=42)
        lr_vid.fit(scaler_vid.fit_transform(X_tr_vid), train_labels)
        acc_lr_vid = accuracy(test_labels, lr_vid.predict(scaler_vid.transform(X_te_vid)))

        # LogReg on enriched
        scaler_enr = StandardScaler()
        lr_enr = LogisticRegression(max_iter=1000, random_state=42)
        lr_enr.fit(scaler_enr.fit_transform(X_tr_enr), train_labels)
        acc_lr_enr = accuracy(test_labels, lr_enr.predict(scaler_enr.transform(X_te_enr)))

        lgb_delta = round((acc_lgb_enr - acc_lgb_vid) * 100, 1)
        lr_delta = round((acc_lr_enr - acc_lr_vid) * 100, 1)

        print(f"  LightGBM  venue_id={acc_lgb_vid:.4f}  enriched={acc_lgb_enr:.4f}  delta={lgb_delta:+.1f}pp")
        print(f"  LogReg    venue_id={acc_lr_vid:.4f}  enriched={acc_lr_enr:.4f}  delta={lr_delta:+.1f}pp")

        results[level] = {
            "lgb_venue_id": round(acc_lgb_vid, 4),
            "lgb_enriched": round(acc_lgb_enr, 4),
            "lgb_delta_pp": lgb_delta,
            "logreg_venue_id": round(acc_lr_vid, 4),
            "logreg_enriched": round(acc_lr_enr, 4),
            "logreg_delta_pp": lr_delta,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Enrichment Delta (enriched - venue_id) in pp")
    print(f"{'='*60}")
    print(f"{'Level':<6} {'LightGBM':>10} {'LogReg':>10} {'Difference':>12}")
    print("-" * 40)
    for level in levels:
        r = results[level]
        diff = round(r["lgb_delta_pp"] - r["logreg_delta_pp"], 1)
        print(f"{level:<6} {r['lgb_delta_pp']:>+10.1f} {r['logreg_delta_pp']:>+10.1f} {diff:>+12.1f}")

    out_dir = ROOT / enr["results_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "logreg_comparison.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
