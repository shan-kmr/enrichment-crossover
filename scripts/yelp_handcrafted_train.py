#!/usr/bin/env python3
"""Train LightGBM + MLP on handcrafted numeric features (no LLM, no embeddings)
at G0-G4 granularity levels.

Features are cumulative -- each level adds to the previous:
  G0: target business only            (3 features)
  G1: + city context                  (6 features)
  G2: + category profile              (11 features)
  G3: + recent trajectory             (18 features)
  G4: + full spatiotemporal           (23 features)
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

from src.yelp.data_loader import YelpTestCase, load_test_cases


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


def compute_features(tc: YelpTestCase, level: str) -> list[float]:
    """Build feature vector at the given granularity level.

    Mirrors the LLM prompt hierarchy: G0 sees only the target business,
    G1 adds city, G2 adds the category profile, G3 adds recent trajectory,
    G4 adds coordinates and dates.
    """
    history = tc.history
    target = tc.target
    profile = tc.user_profile

    target_cats = [c.strip() for c in target.categories.split(",") if c.strip()]
    target_cat_set = set(target_cats)

    features: list[float] = []

    # ── G0: target business features only ──
    features.append(float(len(target_cats)))
    features.append(float(len(target.business_name)))
    features.append(3.0)  # uninformed prior (midpoint of 1-5)

    if level == "G0":
        return features

    # ── G1: city features ──
    city_counts: dict[str, int] = {}
    for r in history:
        city_counts[r.city] = city_counts.get(r.city, 0) + 1
    most_common_city = max(city_counts, key=city_counts.get) if city_counts else ""

    features.append(1.0 if target.city == most_common_city else 0.0)
    city_reviews = [r for r in history if r.city == target.city]
    features.append(len(city_reviews) / len(history) if history else 0.0)
    city_stars = [r.stars for r in city_reviews]
    features.append(float(np.mean(city_stars)) if city_stars else 3.0)

    if level == "G1":
        return features

    # ── G2: category profile features ──
    user_cat_set = set(profile.keys())
    overlap = len(target_cat_set & user_cat_set) / len(target_cat_set) if target_cat_set else 0.0
    features.append(overlap)

    cat_stars = [profile[cat]["avg_stars"] for cat in target_cat_set if cat in profile]
    all_stars = [r.stars for r in history]
    user_avg = float(np.mean(all_stars)) if all_stars else 3.0
    features.append(float(np.mean(cat_stars)) if cat_stars else user_avg)

    cat_visits = sum(profile[cat]["count"] for cat in target_cat_set if cat in profile)
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
            1 for r in recent
            if {c.strip() for c in r.categories.split(",")} & target_cat_set
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
        if {c.strip() for c in r.categories.split(",")} & target_cat_set:
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
        dist = haversine_km(user_lat, user_lon, target.latitude, target.longitude)
        dists_from_centroid = [
            haversine_km(user_lat, user_lon, r.latitude, r.longitude)
            for r in history
            if r.latitude != 0
        ]
        avg_radius = float(np.mean(dists_from_centroid)) if dists_from_centroid else 0.0
        dist_pct = (
            sum(1 for d in dists_from_centroid if d <= dist) / len(dists_from_centroid)
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


def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    train_cases = load_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    test_cases = load_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test cases")

    train_labels = np.array([tc.target.stars for tc in train_cases])
    test_labels = np.array([tc.target.stars for tc in test_cases])

    import lightgbm as lgb
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    model_names = ["LightGBM", "MLP"]
    results: dict[str, dict[str, dict]] = {level: {} for level in levels}

    for level in levels:
        print(f"\n{'='*50}")
        print(f"  {level}")
        print(f"{'='*50}")

        X_train = np.array(
            [compute_features(tc, level) for tc in train_cases], dtype=np.float32
        )
        X_test = np.array(
            [compute_features(tc, level) for tc in test_cases], dtype=np.float32
        )
        print(f"  Feature dimensions: {X_train.shape[1]}")

        # --- LightGBM ---
        val_size = int(len(train_labels) * 0.15)
        idx = np.random.RandomState(42).permutation(len(train_labels))
        train_idx, val_idx = idx[val_size:], idx[:val_size]

        lgb_train = lgb.Dataset(X_train[train_idx], label=train_labels[train_idx])
        lgb_val = lgb.Dataset(
            X_train[val_idx], label=train_labels[val_idx], reference=lgb_train
        )

        params = {
            "objective": "regression",
            "metric": "mae",
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
        lgb_pred = lgb_model.predict(X_test)
        results[level]["LightGBM"] = evaluate(test_labels, lgb_pred)
        print(
            f"  LightGBM: {lgb_model.best_iteration} rounds, "
            f"MAE={results[level]['LightGBM']['MAE']:.4f}"
        )

        importance = lgb_model.feature_importance(importance_type="gain")
        top_k = min(5, len(importance))
        top_idx = np.argsort(importance)[-top_k:][::-1]
        print(f"  Top feature indices (by gain): {list(top_idx)}")

        # --- MLP ---
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
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
        print(
            f"  MLP: {mlp.n_iter_} epochs, "
            f"MAE={results[level]['MLP']['MAE']:.4f}"
        )

    # ------------------------------------------------------------------
    # Load comparison results
    # ------------------------------------------------------------------
    llm_metrics_path = ROOT / cfg["results_dir"] / "metrics.json"
    llm_table = {}
    if llm_metrics_path.exists():
        llm_table = json.loads(llm_metrics_path.read_text())

    embed_results_path = ROOT / cfg["results_dir"] / "embed_results_v2.json"
    embed_table = {}
    if embed_results_path.exists():
        embed_table = json.loads(embed_results_path.read_text())

    metric_map = {
        "MAE": "mae",
        "RMSE": "rmse",
        "Exact": "exact_acc",
        "Within-1": "within1_acc",
    }

    # ------------------------------------------------------------------
    # Print full comparison table
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("HANDCRAFTED vs EMBEDDING-BASED vs LLM DIRECT PROMPTING: STAR RATING PREDICTION")
    print("=" * 110)

    for metric in ["MAE", "RMSE", "Exact", "Within-1"]:
        lower_better = metric in ("MAE", "RMSE")
        label = f"{metric} ({'lower' if lower_better else 'higher'} is better)"
        print(f"\n--- {label} ---")
        header = f"{'Method':<32}" + "".join(f"{lvl:>10}" for lvl in levels)
        print(header)
        print("-" * len(header))

        for model_name in model_names:
            row = f"{model_name + ' (handcrafted)':<32}"
            for level in levels:
                val = results[level][model_name][metric]
                row += f"{val:>10.4f}"
            print(row)

        if embed_table:
            for model_name in ["LightGBM", "MLP"]:
                has_data = any(model_name in embed_table.get(lvl, {}) for lvl in levels)
                if not has_data:
                    continue
                row = f"{model_name + ' (embed-v2)':<32}"
                for level in levels:
                    val = embed_table.get(level, {}).get(model_name, {}).get(metric)
                    row += f"{val:>10.4f}" if val is not None else f"{'N/A':>10}"
                print(row)

        if llm_table:
            print("-" * len(header))
            for llm_model in cfg["models"]:
                if llm_model not in llm_table:
                    continue
                row = f"{llm_model:<32}"
                for level in levels:
                    llm_key = metric_map[metric]
                    val = llm_table[llm_model].get(level, {}).get(llm_key)
                    row += f"{val:>10.4f}" if val is not None else f"{'N/A':>10}"
                print(row)

    print("\n" + "=" * 110)

    print("\nFeature dimensions by granularity:")
    for level in levels:
        n = len(compute_features(test_cases[0], level))
        print(f"  {level}: {n} features")

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
