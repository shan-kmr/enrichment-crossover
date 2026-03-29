#!/usr/bin/env python3
"""Generate 5-panel UMAP progression figure (G0 through G4) for the paper.

Produces: paper/figures/crossover_analysis/umap_progression.png
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore")

from src.gowalla.data_loader import load_test_cases
from src.gowalla.enrichment import load_category_map, compute_enriched_handcrafted

FEATURE_NAMES = {
    "G0": ["checkins_a", "checkins_b", "checkins_sum"],
    "G1_added": [
        "same_region", "centroid_dist", "region_match",
        "same_top_cat*", "entropy_a*", "entropy_b*", "top3_cat_overlap*",
    ],
    "G2_added": [
        "uniq_loc_a", "uniq_loc_b", "shared_locs", "jaccard_locs",
        "shared_cats*", "cat_jaccard*", "uniq_cats_a*", "uniq_cats_b*",
        "nonshared_cat_jacc*", "cat_js_div*", "conc_a*", "conc_b*",
    ],
    "G3_added": ["geo_spread_a", "geo_spread_b", "active_days_a", "active_days_b"],
    "G4_added": [
        "temporal_cooc", "norm_distance", "weighted_jaccard",
        "cat_cosine*", "nonshared_cat_cos*",
    ],
}


def get_names(level: str) -> list[str]:
    names = list(FEATURE_NAMES["G0"])
    for g in ["G1", "G2", "G3", "G4"]:
        if level >= g:
            names.extend(FEATURE_NAMES.get(f"{g}_added", []))
    return names


def enrichment_mask(names: list[str]) -> np.ndarray:
    return np.array([n.endswith("*") for n in names])


def main():
    import umap
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    enr = cfg["enrichment"]
    cat_map = load_category_map(ROOT / enr["categories_path"])
    filt_dir = ROOT / enr["filtered_dir"]

    ml_test = filt_dir / "ml_test_cases.json"
    if ml_test.exists():
        test_cases = load_test_cases(ml_test)
    else:
        test_cases = load_test_cases(filt_dir / "test_cases.json")

    y_test = np.array([tc.label for tc in test_cases])
    print(f"Loaded {len(test_cases)} test cases")

    out_dir = ROOT / "paper" / "figures" / "crossover_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = ["G0", "G1", "G2", "G3", "G4"]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, level in enumerate(levels):
        print(f"  Computing UMAP for {level}...")
        X = np.array(
            [compute_enriched_handcrafted(tc, level, cat_map)
             for tc in test_cases], dtype=np.float32)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        emb = reducer.fit_transform(X_scaled)
        sil = silhouette_score(X_scaled, y_test)

        ax = axes[i]
        pos = y_test == 1
        ax.scatter(emb[~pos, 0], emb[~pos, 1], c="#2196F3",
                   alpha=0.35, s=12, label="Not friends", rasterized=True)
        ax.scatter(emb[pos, 0], emb[pos, 1], c="#E91E63",
                   alpha=0.35, s=12, label="Friends", rasterized=True)
        n_feats = X.shape[1]
        ax.set_title(f"{level} ({n_feats} feat.)\nsil={sil:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.legend(fontsize=7, loc="lower left")
        print(f"    {level}: silhouette={sil:.4f}, dims={n_feats}")

    plt.suptitle("UMAP Progression: Enriched Feature Space Across Granularity Levels (Gowalla Friendship)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out_path = out_dir / "umap_progression.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
