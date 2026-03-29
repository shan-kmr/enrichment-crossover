#!/usr/bin/env python3
"""Train LightGBM + MLP on KAR reasoning embeddings for star rating prediction.

Features: reasoning_emb(1024) + handcrafted(6) = 1030 dims.
Prints a 4-way comparison: KAR vs embed-v2 vs handcrafted vs LLM direct.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_rounded = np.clip(np.round(y_pred), 1, 5).astype(int)
    errors = np.abs(y_true - y_pred)
    errors_r = np.abs(y_true - y_rounded)
    return {
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Exact": float(np.mean(errors_r == 0)),
        "Within-1": float(np.mean(errors_r <= 1)),
    }


def build_features(kar_dir: Path, split: str, level: str) -> np.ndarray:
    reasoning = np.load(kar_dir / f"{split}_{level}_reasoning.npy")
    hc = np.load(kar_dir / f"{split}_handcrafted.npy")
    return np.hstack([reasoning, hc])


def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]
    kar_dir = ROOT / ds["processed_dir"] / "kar_embeddings"

    train_labels = np.load(kar_dir / "train_labels.npy")
    test_labels = np.load(kar_dir / "test_labels.npy")
    print(f"Train: {len(train_labels)}, Test: {len(test_labels)}")

    import lightgbm as lgb
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    model_names = ["LightGBM", "MLP"]
    results: dict[str, dict[str, dict]] = {lv: {} for lv in levels}

    for level in levels:
        X_train = build_features(kar_dir, "train", level)
        X_test = build_features(kar_dir, "test", level)
        print(f"\n{level}: {X_train.shape[1]} dims")

        # LightGBM
        val_size = int(len(train_labels) * 0.15)
        idx = np.random.RandomState(42).permutation(len(train_labels))
        ti, vi = idx[val_size:], idx[:val_size]

        ds_train = lgb.Dataset(X_train[ti], label=train_labels[ti])
        ds_val = lgb.Dataset(X_train[vi], label=train_labels[vi], reference=ds_train)

        params = {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "seed": 42,
        }
        lgb_model = lgb.train(
            params,
            ds_train,
            num_boost_round=1000,
            valid_sets=[ds_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        lgb_pred = lgb_model.predict(X_test)
        results[level]["LightGBM"] = evaluate(test_labels, lgb_pred)
        print(
            f"  LightGBM: {lgb_model.best_iteration} rounds, "
            f"MAE={results[level]['LightGBM']['MAE']:.4f}"
        )

        # MLP
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)

        mlp = MLPRegressor(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=0.001,
            batch_size=128,
            n_iter_no_change=20,
        )
        mlp.fit(Xtr, train_labels)
        mlp_pred = mlp.predict(Xte)
        results[level]["MLP"] = evaluate(test_labels, mlp_pred)
        print(f"  MLP: {mlp.n_iter_} epochs, MAE={results[level]['MLP']['MAE']:.4f}")

    # ------------------------------------------------------------------
    # Load comparison results
    # ------------------------------------------------------------------
    res_dir = ROOT / cfg["results_dir"]
    llm_table = {}
    if (res_dir / "metrics.json").exists():
        llm_table = json.loads((res_dir / "metrics.json").read_text())
    embed_table = {}
    if (res_dir / "embed_results_v2.json").exists():
        embed_table = json.loads((res_dir / "embed_results_v2.json").read_text())
    hc_table = {}
    if (res_dir / "handcrafted_results.json").exists():
        hc_table = json.loads((res_dir / "handcrafted_results.json").read_text())

    metric_map = {
        "MAE": "mae", "RMSE": "rmse", "Exact": "exact_acc", "Within-1": "within1_acc",
    }

    print("\n" + "=" * 110)
    print("KAR vs EMBED-v2 vs HANDCRAFTED vs LLM DIRECT: STAR RATING PREDICTION")
    print("=" * 110)

    for metric in ["MAE", "RMSE", "Exact", "Within-1"]:
        lower = metric in ("MAE", "RMSE")
        print(f"\n--- {metric} ({'lower' if lower else 'higher'} is better) ---")
        header = f"{'Method':<32}" + "".join(f"{lv:>10}" for lv in levels)
        print(header)
        print("-" * len(header))

        for mn in model_names:
            row = f"{mn + ' (KAR)':<32}"
            for lv in levels:
                row += f"{results[lv][mn][metric]:>10.4f}"
            print(row)

        if embed_table:
            for mn in ["LightGBM", "MLP"]:
                if any(mn in embed_table.get(lv, {}) for lv in levels):
                    row = f"{mn + ' (embed-v2)':<32}"
                    for lv in levels:
                        v = embed_table.get(lv, {}).get(mn, {}).get(metric)
                        row += f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
                    print(row)

        if hc_table:
            for mn in ["LightGBM", "MLP"]:
                if any(mn in hc_table.get(lv, {}) for lv in levels):
                    row = f"{mn + ' (handcrafted)':<32}"
                    for lv in levels:
                        v = hc_table.get(lv, {}).get(mn, {}).get(metric)
                        row += f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
                    print(row)

        if llm_table:
            print("-" * len(header))
            for llm_model in cfg["models"]:
                if llm_model not in llm_table:
                    continue
                row = f"{llm_model:<32}"
                for lv in levels:
                    k = metric_map[metric]
                    v = llm_table[llm_model].get(lv, {}).get(k)
                    row += f"{v:>10.4f}" if v is not None else f"{'N/A':>10}"
                print(row)

    print("\n" + "=" * 110)

    serializable = {
        lv: {mn: {k: round(v, 4) for k, v in m.items()} for mn, m in r.items()}
        for lv, r in results.items()
    }
    out = res_dir / "kar_results.json"
    out.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
