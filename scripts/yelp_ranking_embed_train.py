#!/usr/bin/env python3
"""Train LightGBM + MLP on dual embeddings for next-business ranking.

Pointwise approach: each (user, candidate) pair is a training sample.
Features per pair: user_emb(1024) + cand_emb(1024) + diff(1024) + product(1024)
 + handcrafted(6) = 4102 dims.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evaluator import accuracy_at_k, mrr
from src.yelp.ranking_builder import load_ranking_test_cases

EMBED_DIM = 1024


def build_features(embed_dir: Path, split: str, level: str) -> np.ndarray:
    """Assemble pointwise feature matrix from stored embeddings.

    user embeddings:  (n_cases, 1024)
    cand embeddings:  (n_cases, n_cands, 1024)
    handcrafted:      (n_cases, n_cands, 6)

    Output: (n_cases * n_cands, 4102)
    """
    user = np.load(embed_dir / f"{split}_{level}_user.npy")   # (n_cases, 1024)
    cand = np.load(embed_dir / f"{split}_{level}_cand.npy")   # (n_cases, n_cands, 1024)
    hc = np.load(embed_dir / f"{split}_handcrafted.npy")      # (n_cases, n_cands, 6)

    n_cases, n_cands, _ = cand.shape

    # Broadcast user to match candidate shape
    user_expanded = np.repeat(user[:, np.newaxis, :], n_cands, axis=1)  # (n_cases, n_cands, 1024)

    diff = user_expanded - cand
    product = user_expanded * cand

    # Flatten to pointwise: (n_cases * n_cands, feature_dim)
    flat = lambda arr: arr.reshape(n_cases * n_cands, -1)
    return np.hstack([
        flat(user_expanded),
        flat(cand),
        flat(diff),
        flat(product),
        flat(hc),
    ])


def build_labels(embed_dir: Path, split: str, n_cands: int) -> np.ndarray:
    """Binary labels: 1 for ground truth candidate, 0 otherwise."""
    gt_indices = np.load(embed_dir / f"{split}_gt_indices.npy")
    n_cases = len(gt_indices)
    labels = np.zeros(n_cases * n_cands, dtype=np.float32)
    for i, gt_idx in enumerate(gt_indices):
        labels[i * n_cands + gt_idx] = 1.0
    return labels


def scores_to_rankings(
    scores: np.ndarray, n_cases: int, n_cands: int
) -> list[list[int]]:
    scores_2d = scores.reshape(n_cases, n_cands)
    return [np.argsort(row)[::-1].tolist() for row in scores_2d]


def evaluate_rankings(rankings, gt_indices):
    return {
        "Acc@1": accuracy_at_k(rankings, gt_indices, 1),
        "Acc@5": accuracy_at_k(rankings, gt_indices, 5),
        "MRR": mrr(rankings, gt_indices),
    }


def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]
    embed_dir = ROOT / ds["processed_dir"] / "embeddings_v2"

    test_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    n_cands = len(test_cases[0].candidates)
    test_gt_indices = [tc.ground_truth_idx for tc in test_cases]

    train_gt = np.load(embed_dir / "train_gt_indices.npy")
    n_train = len(train_gt)
    n_test = len(test_cases)
    print(f"Train: {n_train} cases ({n_train * n_cands} samples), Test: {n_test} cases ({n_test * n_cands} samples)")

    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    model_names = ["LightGBM", "MLP"]
    results: dict[str, dict[str, dict]] = {level: {} for level in levels}

    for level in levels:
        print(f"\n{'='*50}")
        print(f"  {level}")
        print(f"{'='*50}")

        X_train = build_features(embed_dir, "train", level)
        X_test = build_features(embed_dir, "test", level)
        y_train = build_labels(embed_dir, "train", n_cands)
        print(f"  Feature dims: {X_train.shape[1]}")

        # --- LightGBM ---
        val_size = int(n_train * 0.15)
        rng = np.random.RandomState(42)
        case_perm = rng.permutation(n_train)
        val_case_idx = set(case_perm[:val_size].tolist())

        val_mask = np.zeros(len(y_train), dtype=bool)
        for ci in val_case_idx:
            val_mask[ci * n_cands : (ci + 1) * n_cands] = True

        lgb_train = lgb.Dataset(X_train[~val_mask], label=y_train[~val_mask])
        lgb_val = lgb.Dataset(
            X_train[val_mask], label=y_train[val_mask], reference=lgb_train
        )

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
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
            lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        lgb_scores = lgb_model.predict(X_test)
        lgb_rankings = scores_to_rankings(lgb_scores, n_test, n_cands)
        results[level]["LightGBM"] = evaluate_rankings(lgb_rankings, test_gt_indices)
        print(
            f"  LightGBM: {lgb_model.best_iteration} rounds, "
            f"Acc@1={results[level]['LightGBM']['Acc@1']:.4f}"
        )

        # --- MLP ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=0.001,
            batch_size=256,
            n_iter_no_change=20,
        )
        mlp.fit(X_train_s, y_train.astype(int))
        mlp_scores = mlp.predict_proba(X_test_s)[:, 1]
        mlp_rankings = scores_to_rankings(mlp_scores, n_test, n_cands)
        results[level]["MLP"] = evaluate_rankings(mlp_rankings, test_gt_indices)
        print(
            f"  MLP: {mlp.n_iter_} epochs, "
            f"Acc@1={results[level]['MLP']['Acc@1']:.4f}"
        )

    # ------------------------------------------------------------------
    # Load comparison results
    # ------------------------------------------------------------------
    llm_metrics_path = ROOT / cfg["results_dir"] / "metrics.json"
    llm_table = {}
    if llm_metrics_path.exists():
        llm_table = json.loads(llm_metrics_path.read_text())

    hc_results_path = ROOT / cfg["results_dir"] / "handcrafted_results.json"
    hc_table = {}
    if hc_results_path.exists():
        hc_table = json.loads(hc_results_path.read_text())

    llm_metric_map = {"Acc@1": "acc_at_1", "Acc@5": "acc_at_5", "MRR": "mrr"}

    print("\n" + "=" * 110)
    print("EMBEDDING-BASED vs HANDCRAFTED vs LLM DIRECT PROMPTING: NEXT-BUSINESS RANKING")
    print("=" * 110)

    for metric in ["Acc@1", "Acc@5", "MRR"]:
        print(f"\n--- {metric} (higher is better) ---")
        header = f"{'Method':<32}" + "".join(f"{lvl:>10}" for lvl in levels)
        print(header)
        print("-" * len(header))

        random_vals = {"Acc@1": 0.05, "Acc@5": 0.25, "MRR": 0.122}
        rv = random_vals[metric]
        print(f"{'Random (1/20)':<32}" + f"{rv:>10.4f}" * len(levels))

        # Embedding-based
        for mn in model_names:
            row = f"{mn + ' (embed-v2)':<32}"
            for level in levels:
                val = results[level][mn][metric]
                row += f"{val:>10.4f}"
            print(row)

        # Handcrafted
        if hc_table:
            for mn in ["LightGBM", "MLP"]:
                has = any(mn in hc_table.get(lvl, {}) for lvl in levels)
                if not has:
                    continue
                row = f"{mn + ' (handcrafted)':<32}"
                for level in levels:
                    val = hc_table.get(level, {}).get(mn, {}).get(metric)
                    row += f"{val:>10.4f}" if val is not None else f"{'N/A':>10}"
                print(row)

        # LLM direct
        if llm_table:
            print("-" * len(header))
            for llm_model in cfg["models"]:
                if llm_model not in llm_table:
                    continue
                row = f"{llm_model:<32}"
                for level in levels:
                    llm_key = llm_metric_map[metric]
                    val = llm_table[llm_model].get(level, {}).get(llm_key)
                    row += f"{val:>10.4f}" if val is not None else f"{'N/A':>10}"
                print(row)

    print("\n" + "=" * 110)

    # Save
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
