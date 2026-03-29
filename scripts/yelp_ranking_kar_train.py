#!/usr/bin/env python3
"""Train LightGBM + MLP on KAR reasoning embeddings for next-business ranking.

Pointwise binary classification: each (user, candidate) pair gets a score.
Features: reasoning_emb(1024) + handcrafted(6) = 1030 dims per pair.
Prints 4-way comparison: KAR vs embed-v2 vs handcrafted vs LLM direct.
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


def build_features(kar_dir: Path, split: str, level: str) -> np.ndarray:
    """Flatten (n_cases, n_cands, dim) into (n_cases*n_cands, dim)."""
    reasoning = np.load(kar_dir / f"{split}_{level}_reasoning.npy")  # (n, c, 1024)
    hc = np.load(kar_dir / f"{split}_handcrafted.npy")               # (n, c, 6)
    n, c, _ = reasoning.shape
    r_flat = reasoning.reshape(n * c, -1)
    h_flat = hc.reshape(n * c, -1)
    return np.hstack([r_flat, h_flat])


def build_labels(kar_dir: Path, split: str, n_cands: int) -> np.ndarray:
    gt = np.load(kar_dir / f"{split}_gt_indices.npy")
    labels = np.zeros(len(gt) * n_cands, dtype=np.float32)
    for i, g in enumerate(gt):
        labels[i * n_cands + g] = 1.0
    return labels


def scores_to_rankings(scores: np.ndarray, n_cases: int, n_cands: int):
    s2d = scores.reshape(n_cases, n_cands)
    return [np.argsort(row)[::-1].tolist() for row in s2d]


def eval_rankings(rankings, gt_indices):
    return {
        "Acc@1": accuracy_at_k(rankings, gt_indices, 1),
        "Acc@5": accuracy_at_k(rankings, gt_indices, 5),
        "MRR": mrr(rankings, gt_indices),
    }


def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]
    kar_dir = ROOT / ds["processed_dir"] / "kar_embeddings"

    test_cases = load_ranking_test_cases(
        ROOT / ds["processed_dir"] / "test_cases.json"
    )
    n_cands = len(test_cases[0].candidates)
    test_gt = [tc.ground_truth_idx for tc in test_cases]

    train_gt = np.load(kar_dir / "train_gt_indices.npy")
    n_train, n_test = len(train_gt), len(test_cases)
    print(
        f"Train: {n_train} cases ({n_train * n_cands} samples), "
        f"Test: {n_test} cases ({n_test * n_cands} samples)"
    )

    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    model_names = ["LightGBM", "MLP"]
    results: dict[str, dict[str, dict]] = {lv: {} for lv in levels}

    for level in levels:
        print(f"\n{'='*50}")
        print(f"  {level}")
        print(f"{'='*50}")

        X_train = build_features(kar_dir, "train", level)
        X_test = build_features(kar_dir, "test", level)
        y_train = build_labels(kar_dir, "train", n_cands)
        print(f"  Feature dims: {X_train.shape[1]}")

        # LightGBM
        val_size = int(n_train * 0.15)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n_train)
        val_set = set(perm[:val_size].tolist())
        val_mask = np.zeros(len(y_train), dtype=bool)
        for ci in val_set:
            val_mask[ci * n_cands : (ci + 1) * n_cands] = True

        lgb_t = lgb.Dataset(X_train[~val_mask], label=y_train[~val_mask])
        lgb_v = lgb.Dataset(
            X_train[val_mask], label=y_train[val_mask], reference=lgb_t
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
            lgb_t,
            num_boost_round=1000,
            valid_sets=[lgb_v],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        lgb_scores = lgb_model.predict(X_test)
        lgb_rank = scores_to_rankings(lgb_scores, n_test, n_cands)
        results[level]["LightGBM"] = eval_rankings(lgb_rank, test_gt)
        print(
            f"  LightGBM: {lgb_model.best_iteration} rounds, "
            f"Acc@1={results[level]['LightGBM']['Acc@1']:.4f}"
        )

        # MLP
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train)
        Xte = scaler.transform(X_test)
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
        mlp.fit(Xtr, y_train.astype(int))
        mlp_scores = mlp.predict_proba(Xte)[:, 1]
        mlp_rank = scores_to_rankings(mlp_scores, n_test, n_cands)
        results[level]["MLP"] = eval_rankings(mlp_rank, test_gt)
        print(
            f"  MLP: {mlp.n_iter_} epochs, "
            f"Acc@1={results[level]['MLP']['Acc@1']:.4f}"
        )

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

    llm_map = {"Acc@1": "acc_at_1", "Acc@5": "acc_at_5", "MRR": "mrr"}

    print("\n" + "=" * 110)
    print("KAR vs EMBED-v2 vs HANDCRAFTED vs LLM DIRECT: NEXT-BUSINESS RANKING")
    print("=" * 110)

    for metric in ["Acc@1", "Acc@5", "MRR"]:
        print(f"\n--- {metric} (higher is better) ---")
        header = f"{'Method':<32}" + "".join(f"{lv:>10}" for lv in levels)
        print(header)
        print("-" * len(header))

        rv = {"Acc@1": 0.05, "Acc@5": 0.25, "MRR": 0.122}[metric]
        print(f"{'Random (1/20)':<32}" + f"{rv:>10.4f}" * len(levels))

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
                    v = llm_table[llm_model].get(lv, {}).get(llm_map[metric])
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
