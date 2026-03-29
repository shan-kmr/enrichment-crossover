#!/usr/bin/env python3
"""Train LightGBM and MLP on handcrafted features for Gowalla friendship prediction."""

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

from src.gowalla.data_loader import FriendshipTestCase, load_test_cases


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
    levels = cfg["granularity_levels"]

    ml_train = ROOT / ds["processed_dir"] / "ml_train_cases.json"
    ml_test = ROOT / ds["processed_dir"] / "ml_test_cases.json"
    if ml_train.exists() and ml_test.exists():
        train_cases = load_test_cases(ml_train)
        test_cases = load_test_cases(ml_test)
        print(f"Loaded ML datasets: {len(train_cases)} train, {len(test_cases)} test")
    else:
        train_cases = load_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
        test_cases = load_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
        print(f"Loaded {len(train_cases)} train, {len(test_cases)} test")

    train_labels = np.array([tc.label for tc in train_cases])
    test_labels = np.array([tc.label for tc in test_cases])

    results_dir = ROOT / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}

    for level in levels:
        print(f"\n{'='*60}")
        print(f"Level: {level}")
        print(f"{'='*60}")

        X_train = np.array([compute_handcrafted(tc, level) for tc in train_cases], dtype=np.float32)
        X_test = np.array([compute_handcrafted(tc, level) for tc in test_cases], dtype=np.float32)
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

    # Print full comparison table
    print("\n" + "=" * 90)
    print("GOWALLA FRIENDSHIP PREDICTION: FULL COMPARISON")
    print("=" * 90)

    llm_path = results_dir / "llm_summary.json"
    llm_metrics = json.loads(llm_path.read_text()) if llm_path.exists() else {}

    embed_path = results_dir / "embed_results_v2.json"
    embed_metrics = json.loads(embed_path.read_text()) if embed_path.exists() else {}

    for metric in ["accuracy", "f1"]:
        label = "Accuracy" if metric == "accuracy" else "F1 Score"
        print(f"\n--- {label} (higher is better) ---")
        header = f"{'Method':<30}" + "".join(f"{l:>10}" for l in levels)
        print(header)
        print("-" * len(header))
        print(f"{'Random':<30}" + "".join(f"{'0.5000':>10}" for _ in levels))

        # Embed-v2
        for tag, key in [("LightGBM (embed-v2)", "lgb_embed"), ("MLP (embed-v2)", "mlp_embed")]:
            row = f"{tag:<30}"
            for level in levels:
                v = embed_metrics.get(level, {}).get(key, {}).get(metric, "NA")
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

        # Handcrafted
        for tag, key in [("LightGBM (handcrafted)", "lgb_handcrafted"), ("MLP (handcrafted)", "mlp_handcrafted")]:
            row = f"{tag:<30}"
            for level in levels:
                v = all_results.get(level, {}).get(key, {}).get(metric, "NA")
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

        print("-" * len(header))
        for model in llm_metrics:
            row = f"{model:<30}"
            for level in levels:
                v = llm_metrics.get(model, {}).get(level, {}).get(metric, "NA")
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

    print("\n" + "=" * 90)

    out_path = results_dir / "handcrafted_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
