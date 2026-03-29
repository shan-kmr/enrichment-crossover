#!/usr/bin/env python3
"""Train LightGBM and MLP on enriched dual embeddings for Gowalla friendship prediction."""

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


def build_features(emb_a: np.ndarray, emb_b: np.ndarray, hc: np.ndarray) -> np.ndarray:
    diff = emb_a - emb_b
    prod = emb_a * emb_b
    return np.hstack([emb_a, emb_b, diff, prod, hc])


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
    enr = cfg["enrichment"]
    levels = cfg["granularity_levels"]

    emb_dir = ROOT / enr["embeddings_dir"]
    results_dir = ROOT / enr["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    train_labels = np.load(emb_dir / "train_labels.npy")
    test_labels = np.load(emb_dir / "test_labels.npy")

    all_results: dict[str, dict] = {}

    for level in levels:
        print(f"\n{'='*60}")
        print(f"Level: {level} (enriched)")
        print(f"{'='*60}")

        tr_a = np.load(emb_dir / f"train_{level}_user_a.npy")
        tr_b = np.load(emb_dir / f"train_{level}_user_b.npy")
        tr_hc = np.load(emb_dir / f"train_{level}_handcrafted.npy")
        te_a = np.load(emb_dir / f"test_{level}_user_a.npy")
        te_b = np.load(emb_dir / f"test_{level}_user_b.npy")
        te_hc = np.load(emb_dir / f"test_{level}_handcrafted.npy")

        X_train = build_features(tr_a, tr_b, tr_hc)
        X_test = build_features(te_a, te_b, te_hc)
        print(f"  Features: {X_train.shape[1]} dims")

        # LightGBM
        dtrain = lgb.Dataset(X_train, label=train_labels)
        dval = lgb.Dataset(X_test, label=test_labels, reference=dtrain)
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        bst = lgb.train(
            params, dtrain, num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )
        preds_lgb = (bst.predict(X_test) > 0.5).astype(int)
        m_lgb = compute_metrics(test_labels, preds_lgb)
        print(f"  LightGBM: Acc={m_lgb['accuracy']:.4f}, F1={m_lgb['f1']:.4f}")

        # MLP
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6))
        X_te_s = scaler.transform(np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6))
        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            batch_size=128,
            random_state=42,
        )
        mlp.fit(X_tr_s, train_labels)
        preds_mlp = mlp.predict(X_te_s)
        m_mlp = compute_metrics(test_labels, preds_mlp)
        print(f"  MLP: Acc={m_mlp['accuracy']:.4f}, F1={m_mlp['f1']:.4f}")

        all_results[level] = {"lgb_embed": m_lgb, "mlp_embed": m_mlp}

    out_path = results_dir / "enriched_embed_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
