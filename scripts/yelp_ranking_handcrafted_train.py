#!/usr/bin/env python3
"""Train LightGBM + MLP on handcrafted features for next-business ranking.

Pointwise learning-to-rank: each (user, candidate) pair is a training sample
with binary labels (1 = ground truth, 0 = negative).  At test time, score all
20 candidates per case, rank by predicted probability, compute Acc@1/5, MRR.

Feature hierarchy (cumulative, same as rating task):
  G0: candidate features only   (3 features)
  G1: + city context             (6 features)
  G2: + category profile         (11 features)
  G3: + recent trajectory        (18 features)
  G4: + full spatiotemporal      (23 features)
"""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.yelp.data_loader import ReviewRecord
from src.yelp.ranking_builder import RankingTestCase, load_ranking_test_cases
from src.evaluator import accuracy_at_k, mrr


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_pair_features(
    tc: RankingTestCase, candidate: ReviewRecord, level: str
) -> list[float]:
    """Feature vector for a (user, candidate) pair at given granularity."""
    history = tc.history
    profile = tc.user_profile

    cand_cats = [c.strip() for c in candidate.categories.split(",") if c.strip()]
    cand_cat_set = set(cand_cats)

    features: list[float] = []

    # ── G0: candidate features only ──
    features.append(float(len(cand_cats)))
    features.append(float(len(candidate.business_name)))
    features.append(0.5)  # uninformed prior

    if level == "G0":
        return features

    # ── G1: city features ──
    city_counts: dict[str, int] = {}
    for r in history:
        city_counts[r.city] = city_counts.get(r.city, 0) + 1
    most_common_city = max(city_counts, key=city_counts.get) if city_counts else ""

    features.append(1.0 if candidate.city == most_common_city else 0.0)
    city_reviews = [r for r in history if r.city == candidate.city]
    features.append(len(city_reviews) / len(history) if history else 0.0)
    city_stars = [r.stars for r in city_reviews]
    features.append(float(np.mean(city_stars)) if city_stars else 3.0)

    if level == "G1":
        return features

    # ── G2: category profile features ──
    user_cat_set = set(profile.keys())
    overlap = (
        len(cand_cat_set & user_cat_set) / len(cand_cat_set)
        if cand_cat_set
        else 0.0
    )
    features.append(overlap)

    cat_stars = [
        profile[cat]["avg_stars"] for cat in cand_cat_set if cat in profile
    ]
    all_stars = [r.stars for r in history]
    user_avg = float(np.mean(all_stars)) if all_stars else 3.0
    features.append(float(np.mean(cat_stars)) if cat_stars else user_avg)

    cat_visits = sum(
        profile[cat]["count"] for cat in cand_cat_set if cat in profile
    )
    features.append(float(cat_visits))

    features.append(user_avg)
    features.append(float(np.std(all_stars)) if len(all_stars) > 1 else 0.0)

    if level == "G2":
        return features

    # ── G3: recent trajectory features ──
    MAX_RECENT = 15
    recent = history[-MAX_RECENT:]
    recent_stars = [r.stars for r in recent]

    features.append(float(np.mean(recent_stars)))
    features.append(float(np.std(recent_stars)) if len(recent_stars) > 1 else 0.0)

    recent_cat_match = (
        sum(
            1
            for r in recent
            if {c.strip() for c in r.categories.split(",")} & cand_cat_set
        )
        / len(recent)
        if recent
        else 0.0
    )
    features.append(recent_cat_match)

    if len(recent_stars) >= 3:
        x = np.arange(len(recent_stars), dtype=float)
        slope = float(np.polyfit(x, np.array(recent_stars, dtype=float), 1)[0])
    else:
        slope = 0.0
    features.append(slope)

    streak = 0
    for r in reversed(recent):
        if {c.strip() for c in r.categories.split(",")} & cand_cat_set:
            streak += 1
        else:
            break
    features.append(float(streak))

    recent_cats: set[str] = set()
    for r in recent:
        for c in r.categories.split(","):
            c = c.strip()
            if c:
                recent_cats.add(c)
    features.append(float(len(recent_cats)))

    last5 = recent[-5:] if len(recent) >= 5 else recent
    features.append(float(np.mean([r.stars for r in last5])))

    if level == "G3":
        return features

    # ── G4: spatiotemporal features ──
    lats = [r.latitude for r in history if r.latitude != 0]
    lons = [r.longitude for r in history if r.longitude != 0]

    if lats and lons:
        user_lat = float(np.mean(lats))
        user_lon = float(np.mean(lons))
        dist = haversine_km(user_lat, user_lon, candidate.latitude, candidate.longitude)
        dists_from_centroid = [
            haversine_km(user_lat, user_lon, r.latitude, r.longitude)
            for r in history
            if r.latitude != 0
        ]
        avg_radius = (
            float(np.mean(dists_from_centroid)) if dists_from_centroid else 0.0
        )
        dist_pct = (
            sum(1 for d in dists_from_centroid if d <= dist)
            / len(dists_from_centroid)
            if dists_from_centroid
            else 0.5
        )
    else:
        dist = 0.0
        avg_radius = 0.0
        dist_pct = 0.5

    features.append(dist)
    features.append(avg_radius)
    features.append(dist_pct)

    dates: list[datetime] = []
    for r in history:
        try:
            dates.append(datetime.strptime(r.date[:10], "%Y-%m-%d"))
        except (ValueError, TypeError):
            pass

    if len(dates) >= 2:
        day_gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        features.append(float(np.mean(day_gaps)))
    else:
        features.append(0.0)

    if dates:
        features.append(sum(1 for d in dates if d.weekday() >= 5) / len(dates))
    else:
        features.append(0.0)

    return features


# ------------------------------------------------------------------
# Build pointwise training/test arrays
# ------------------------------------------------------------------

def build_pointwise(cases: list[RankingTestCase], level: str):
    """Return (X, y, group_sizes) for pointwise ranking."""
    X_rows: list[list[float]] = []
    y_rows: list[float] = []
    group_sizes: list[int] = []

    for tc in cases:
        n_cands = len(tc.candidates)
        group_sizes.append(n_cands)
        for i, cand in enumerate(tc.candidates):
            X_rows.append(compute_pair_features(tc, cand, level))
            y_rows.append(1.0 if i == tc.ground_truth_idx else 0.0)

    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.float32),
        group_sizes,
    )


def scores_to_rankings(scores: np.ndarray, group_sizes: list[int]) -> list[list[int]]:
    """Convert flat score array into per-case ranked candidate lists."""
    rankings = []
    offset = 0
    for gs in group_sizes:
        case_scores = scores[offset : offset + gs]
        ranked = np.argsort(case_scores)[::-1].tolist()
        rankings.append(ranked)
        offset += gs
    return rankings


def evaluate_rankings(rankings, gt_indices):
    return {
        "Acc@1": accuracy_at_k(rankings, gt_indices, 1),
        "Acc@5": accuracy_at_k(rankings, gt_indices, 5),
        "MRR": mrr(rankings, gt_indices),
    }


# ------------------------------------------------------------------

def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    train_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    test_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test ranking cases")

    test_gt_indices = [tc.ground_truth_idx for tc in test_cases]

    import lightgbm as lgb
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    model_names = ["LightGBM", "MLP"]
    results: dict[str, dict[str, dict]] = {level: {} for level in levels}

    for level in levels:
        print(f"\n{'='*50}")
        print(f"  {level}")
        print(f"{'='*50}")

        X_train, y_train, train_groups = build_pointwise(train_cases, level)
        X_test, _, test_groups = build_pointwise(test_cases, level)
        print(f"  Features: {X_train.shape[1]} dims, Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        print(f"  Positive rate: {y_train.mean():.3f}")

        # --- LightGBM (binary classification) ---
        val_size = int(len(train_cases) * 0.15)
        rng = np.random.RandomState(42)
        case_perm = rng.permutation(len(train_cases))
        val_case_idx = set(case_perm[:val_size].tolist())

        val_mask = np.zeros(len(y_train), dtype=bool)
        offset = 0
        for i, gs in enumerate(train_groups):
            if i in val_case_idx:
                val_mask[offset : offset + gs] = True
            offset += gs

        lgb_train = lgb.Dataset(X_train[~val_mask], label=y_train[~val_mask])
        lgb_val = lgb.Dataset(
            X_train[val_mask], label=y_train[val_mask], reference=lgb_train
        )

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "verbose": -1,
            "seed": 42,
        }
        lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        lgb_scores = lgb_model.predict(X_test)
        lgb_rankings = scores_to_rankings(lgb_scores, test_groups)
        results[level]["LightGBM"] = evaluate_rankings(lgb_rankings, test_gt_indices)
        print(f"  LightGBM: {lgb_model.best_iteration} rounds, Acc@1={results[level]['LightGBM']['Acc@1']:.4f}")

        # --- MLP (binary classification) ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        mlp = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            learning_rate_init=0.001,
            batch_size=256,
            n_iter_no_change=20,
        )
        mlp.fit(X_train_s, y_train.astype(int))
        mlp_scores = mlp.predict_proba(X_test_s)[:, 1]
        mlp_rankings = scores_to_rankings(mlp_scores, test_groups)
        results[level]["MLP"] = evaluate_rankings(mlp_rankings, test_gt_indices)
        print(f"  MLP: {mlp.n_iter_} epochs, Acc@1={results[level]['MLP']['Acc@1']:.4f}")

    # ------------------------------------------------------------------
    # Load LLM results for comparison
    # ------------------------------------------------------------------
    llm_metrics_path = ROOT / cfg["results_dir"] / "metrics.json"
    llm_table = {}
    if llm_metrics_path.exists():
        llm_table = json.loads(llm_metrics_path.read_text())

    print("\n" + "=" * 100)
    print("HANDCRAFTED vs LLM DIRECT PROMPTING: NEXT-BUSINESS RANKING")
    print("=" * 100)

    metric_keys = ["Acc@1", "Acc@5", "MRR"]
    llm_metric_map = {"Acc@1": "acc_at_1", "Acc@5": "acc_at_5", "MRR": "mrr"}

    for metric in metric_keys:
        print(f"\n--- {metric} (higher is better) ---")
        header = f"{'Method':<32}" + "".join(f"{lvl:>10}" for lvl in levels)
        print(header)
        print("-" * len(header))

        random_vals = {"Acc@1": 0.05, "Acc@5": 0.25, "MRR": 0.122}
        rv = random_vals[metric]
        print(f"{'Random (1/20)':<32}" + f"{rv:>10.4f}" * len(levels))

        for model_name in model_names:
            row = f"{model_name + ' (handcrafted)':<32}"
            for level in levels:
                val = results[level][model_name][metric]
                row += f"{val:>10.4f}"
            print(row)

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

    print("\n" + "=" * 100)

    # Save
    serializable = {}
    for level in results:
        serializable[level] = {
            model: {k: round(v, 4) for k, v in metrics.items()}
            for model, metrics in results[level].items()
        }
    out_path = ROOT / cfg["results_dir"] / "handcrafted_results.json"
    out_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
