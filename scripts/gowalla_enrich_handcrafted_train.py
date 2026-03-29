#!/usr/bin/env python3
"""Train LightGBM and MLP on enriched handcrafted features for Gowalla friendship prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.gowalla.data_loader import load_test_cases
from src.gowalla.enrichment import load_category_map, compute_enriched_handcrafted


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": round(acc, 4), "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)}


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    ds = cfg["dataset"]
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
        print(f"Loaded filtered ML datasets: {len(train_cases)} train, {len(test_cases)} test")
    else:
        train_cases = load_test_cases(filt_dir / "train_cases.json")
        test_cases = load_test_cases(filt_dir / "test_cases.json")
        print(f"Loaded filtered {len(train_cases)} train, {len(test_cases)} test")

    train_labels = np.array([tc.label for tc in train_cases])
    test_labels = np.array([tc.label for tc in test_cases])

    results_dir = ROOT / enr["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for level in levels:
        print(f"\n{'='*60}")
        print(f"Level: {level} (enriched handcrafted)")
        print(f"{'='*60}")

        X_train = np.array(
            [compute_enriched_handcrafted(tc, level, cat_map) for tc in train_cases],
            dtype=np.float32,
        )
        X_test = np.array(
            [compute_enriched_handcrafted(tc, level, cat_map) for tc in test_cases],
            dtype=np.float32,
        )
        print(f"  Features: {X_train.shape[1]} dims")

        # LightGBM
        dtrain = lgb.Dataset(X_train, label=train_labels)
        dval = lgb.Dataset(X_test, label=test_labels, reference=dtrain)
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
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        preds_lgb = (bst.predict(X_test) > 0.5).astype(int)
        m_lgb = compute_metrics(test_labels, preds_lgb)
        print(f"  LightGBM: Acc={m_lgb['accuracy']:.4f}, F1={m_lgb['f1']:.4f}")

        # MLP
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=42,
        )
        mlp.fit(X_tr_s, train_labels)
        preds_mlp = mlp.predict(X_te_s)
        m_mlp = compute_metrics(test_labels, preds_mlp)
        print(f"  MLP: Acc={m_mlp['accuracy']:.4f}, F1={m_mlp['f1']:.4f}")

        all_results[level] = {"lgb_handcrafted": m_lgb, "mlp_handcrafted": m_mlp}

    out_path = results_dir / "enriched_handcrafted_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
