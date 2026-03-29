#!/usr/bin/env python3
"""Train LightGBM + MLP on dual context embeddings (v2) with interaction
features and handcrafted numerics.  Compare with LLM direct prompting."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred_rounded = np.clip(np.round(y_pred), 1, 5).astype(int)
    errors = np.abs(y_true - y_pred)
    errors_rounded = np.abs(y_true - y_pred_rounded)
    return {
        "MAE": float(np.mean(errors)),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "Exact": float(np.mean(errors_rounded == 0)),
        "Within-1": float(np.mean(errors_rounded <= 1)),
    }


def build_features(embed_dir: Path, split: str, level: str) -> np.ndarray:
    """Assemble feature matrix: [user, biz, user-biz, user*biz, handcrafted]."""
    user = np.load(embed_dir / f"{split}_{level}_user.npy")
    biz = np.load(embed_dir / f"{split}_{level}_biz.npy")
    hc = np.load(embed_dir / f"{split}_handcrafted.npy")

    diff = user - biz
    product = user * biz

    return np.hstack([user, biz, diff, product, hc])


def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]
    embed_dir = ROOT / ds["processed_dir"] / "embeddings_v2"

    train_labels = np.load(embed_dir / "train_labels.npy")
    test_labels = np.load(embed_dir / "test_labels.npy")
    print(f"Train: {len(train_labels)} cases, Test: {len(test_labels)} cases")

    import lightgbm as lgb
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    model_names = ["LightGBM", "MLP"]
    results: dict[str, dict[str, dict]] = {level: {} for level in levels}

    for level in levels:
        X_train = build_features(embed_dir, "train", level)
        X_test = build_features(embed_dir, "test", level)
        print(f"\n{level}: feature matrix {X_train.shape[1]} dims")

        # --- LightGBM with early stopping ---
        val_size = int(len(train_labels) * 0.15)
        idx = np.random.RandomState(42).permutation(len(train_labels))
        train_idx, val_idx = idx[val_size:], idx[:val_size]

        lgb_train = lgb.Dataset(X_train[train_idx], label=train_labels[train_idx])
        lgb_val = lgb.Dataset(X_train[val_idx], label=train_labels[val_idx], reference=lgb_train)

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
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        lgb_model = lgb.train(
            params, lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=callbacks,
        )
        lgb_pred = lgb_model.predict(X_test)
        results[level]["LightGBM"] = evaluate(test_labels, lgb_pred)
        print(f"  LightGBM: {lgb_model.best_iteration} rounds, MAE={results[level]['LightGBM']['MAE']:.4f}")

        # --- MLP (larger architecture, longer training) ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

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
        mlp.fit(X_train_s, train_labels)
        mlp_pred = mlp.predict(X_test_s)
        results[level]["MLP"] = evaluate(test_labels, mlp_pred)
        print(f"  MLP: {mlp.n_iter_} epochs, MAE={results[level]['MLP']['MAE']:.4f}")

    # ------------------------------------------------------------------
    # Load LLM direct-prompting results for comparison
    # ------------------------------------------------------------------
    llm_metrics_path = ROOT / cfg["results_dir"] / "metrics.json"
    llm_table = {}
    if llm_metrics_path.exists():
        llm_table = json.loads(llm_metrics_path.read_text())

    metric_map = {"MAE": "mae", "RMSE": "rmse", "Exact": "exact_acc", "Within-1": "within1_acc"}

    print("\n" + "=" * 90)
    print("EMBEDDING-BASED vs LLM DIRECT PROMPTING: RATING PREDICTION (v2)")
    print("=" * 90)

    for metric in ["MAE", "RMSE", "Exact", "Within-1"]:
        lower_better = metric in ("MAE", "RMSE")
        label = f"{metric} ({'lower' if lower_better else 'higher'} is better)"
        print(f"\n--- {label} ---")
        header = f"{'Method':<25}" + "".join(f"{lvl:>10}" for lvl in levels)
        print(header)
        print("-" * len(header))

        for model_name in model_names:
            row = f"{model_name + ' (embed-v2)':<25}"
            for level in levels:
                val = results[level][model_name][metric]
                row += f"{val:>10.4f}"
            print(row)

        if llm_table:
            print("-" * len(header))
            for llm_model in cfg["models"]:
                if llm_model not in llm_table:
                    continue
                row = f"{llm_model:<25}"
                for level in levels:
                    llm_key = metric_map[metric]
                    val = llm_table[llm_model].get(level, {}).get(llm_key)
                    row += f"{val:>10.4f}" if val is not None else f"{'N/A':>10}"
                print(row)

    print("\n" + "=" * 90)

    # Save results
    serializable = {}
    for level in results:
        serializable[level] = {
            model: {k: round(v, 4) for k, v in metrics.items()}
            for model, metrics in results[level].items()
        }
    out_path = ROOT / cfg["results_dir"] / "embed_results_v2.json"
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
