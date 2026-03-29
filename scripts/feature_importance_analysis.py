#!/usr/bin/env python3
"""Extract LightGBM feature importances at each granularity level.

Trains LightGBM on the enriched tier for both friendship and next-location
tasks, then groups features by category to show how enrichment feature
importance changes across granularity levels.

Output: results/feature_importance.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm as lgb

from src.gowalla.data_loader import load_test_cases, load_nextloc_cases
from src.gowalla.enrichment import load_category_map, compute_enriched_handcrafted
from src.gowalla.nextloc_features import compute_nextloc_handcrafted

FRIENDSHIP_FEATURE_NAMES = {
    "G0": ["checkins_a", "checkins_b", "checkins_sum"],
    "G1_added": ["same_region", "centroid_dist", "region_match",
                  "same_top_cat*", "entropy_a*", "entropy_b*", "top3_cat_overlap*"],
    "G2_added": ["uniq_loc_a", "uniq_loc_b", "shared_locs", "jaccard_locs",
                  "shared_cats*", "cat_jaccard*", "uniq_cats_a*", "uniq_cats_b*",
                  "nonshared_cat_jaccard*", "cat_js_div*", "concentration_a*", "concentration_b*"],
    "G3_added": ["geo_spread_a", "geo_spread_b", "active_days_a", "active_days_b"],
    "G4_added": ["temporal_cooc", "norm_distance", "weighted_jaccard",
                  "cat_cosine*", "nonshared_cat_cosine*"],
}

NEXTLOC_ENRICHED_FEATURE_NAMES = {
    "G0": ["user_checkins", "user_uniq_locs", "cand_popularity"],
    "G1_added": ["user_region", "dist_to_centroid",
                  "user_top_cat*", "cand_cat_known*", "cand_cat_in_profile*"],
    "G2_added": ["geo_spread", "active_days", "visited_cand", "visits_to_cand", "dist_to_last",
                  "cat_entropy*", "cand_cat_visits*", "cand_cat_ratio*"],
    "G3_added": ["min_dist_recent", "avg_dist_recent", "last_dist_recent",
                  "recent_same_cat*"],
    "G4_added": ["mean_visit_hour", "std_visit_hour", "avg_hour_at_cand"],
}


def get_feature_names(name_map: dict, level: str) -> list[str]:
    names = list(name_map["G0"])
    for g in ["G1", "G2", "G3", "G4"]:
        if level >= g:
            names.extend(name_map.get(f"{g}_added", []))
    return names


def is_enrichment(name: str) -> bool:
    return name.endswith("*")


def analyze_importance(importances: np.ndarray, names: list[str]) -> dict:
    total = importances.sum()
    if total == 0:
        return {"enrichment_share": 0.0, "features": {}}

    enrichment_imp = sum(
        imp for imp, n in zip(importances, names) if is_enrichment(n))

    feat_details = []
    for imp, n in sorted(zip(importances, names), key=lambda x: -x[0]):
        feat_details.append({
            "name": n.rstrip("*"),
            "importance": round(float(imp), 1),
            "share": round(float(imp / total * 100), 2),
            "is_enrichment": is_enrichment(n),
        })

    return {
        "enrichment_share_pct": round(float(enrichment_imp / total * 100), 2),
        "non_enrichment_share_pct": round(float((total - enrichment_imp) / total * 100), 2),
        "n_features": len(names),
        "n_enrichment_features": sum(1 for n in names if is_enrichment(n)),
        "top_5": feat_details[:5],
    }


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    nl = cfg["nextloc"]
    levels = cfg["granularity_levels"]

    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded category map: {len(cat_map)} locations")

    results = {"friendship": {}, "nextloc": {}}

    # ── Friendship ──────────────────────────────────────────────────
    filt_dir = ROOT / enr["filtered_dir"]
    ml_train = filt_dir / "ml_train_cases.json"
    ml_test = filt_dir / "ml_test_cases.json"
    if ml_train.exists() and ml_test.exists():
        f_train = load_test_cases(ml_train)
        f_test = load_test_cases(ml_test)
    else:
        f_train = load_test_cases(filt_dir / "train_cases.json")
        f_test = load_test_cases(filt_dir / "test_cases.json")
    f_train_y = np.array([tc.label for tc in f_train])
    f_test_y = np.array([tc.label for tc in f_test])
    print(f"\nFriendship: {len(f_train)} train, {len(f_test)} test")

    for level in levels:
        X_train = np.array(
            [compute_enriched_handcrafted(tc, level, cat_map) for tc in f_train],
            dtype=np.float32)
        X_test = np.array(
            [compute_enriched_handcrafted(tc, level, cat_map) for tc in f_test],
            dtype=np.float32)

        dtrain = lgb.Dataset(X_train, label=f_train_y)
        dval = lgb.Dataset(X_test, label=f_test_y, reference=dtrain)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 31, "learning_rate": 0.05,
            "feature_fraction": 0.9, "verbose": -1,
        }
        bst = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        names = get_feature_names(FRIENDSHIP_FEATURE_NAMES, level)
        importances = bst.feature_importance(importance_type="gain")

        if len(names) != len(importances):
            print(f"  WARNING: {level} names={len(names)} vs imp={len(importances)}")
            names = [f"f{i}" for i in range(len(importances))]

        analysis = analyze_importance(importances, names)
        results["friendship"][level] = analysis
        print(f"  {level}: enrichment share = {analysis['enrichment_share_pct']:.1f}% "
              f"({analysis['n_enrichment_features']}/{analysis['n_features']} features)")

    # ── Next-Location ──────────────────────────────────────────────
    data_dir = ROOT / nl["data_dir"]
    n_train = load_nextloc_cases(data_dir / "train_cases.json")
    n_test = load_nextloc_cases(data_dir / "test_cases.json")
    n_train_y = np.array([tc.label for tc in n_train])
    n_test_y = np.array([tc.label for tc in n_test])
    print(f"\nNext-Location: {len(n_train)} train, {len(n_test)} test")

    for level in levels:
        X_train = np.array(
            [compute_nextloc_handcrafted(tc, level, "enriched", cat_map=cat_map)
             for tc in n_train], dtype=np.float32)
        X_test = np.array(
            [compute_nextloc_handcrafted(tc, level, "enriched", cat_map=cat_map)
             for tc in n_test], dtype=np.float32)
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        dtrain = lgb.Dataset(X_train, label=n_train_y)
        dval = lgb.Dataset(X_test, label=n_test_y, reference=dtrain)
        params = {
            "objective": "binary", "metric": "binary_logloss",
            "num_leaves": 31, "learning_rate": 0.05,
            "feature_fraction": 0.9, "verbose": -1,
        }
        bst = lgb.train(
            params, dtrain, num_boost_round=300,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        names = get_feature_names(NEXTLOC_ENRICHED_FEATURE_NAMES, level)
        importances = bst.feature_importance(importance_type="gain")

        if len(names) != len(importances):
            print(f"  WARNING: {level} names={len(names)} vs imp={len(importances)}")
            names = [f"f{i}" for i in range(len(importances))]

        analysis = analyze_importance(importances, names)
        results["nextloc"][level] = analysis
        print(f"  {level}: enrichment share = {analysis['enrichment_share_pct']:.1f}% "
              f"({analysis['n_enrichment_features']}/{analysis['n_features']} features)")

    out_path = ROOT / "results" / "feature_importance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
