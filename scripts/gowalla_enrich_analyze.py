#!/usr/bin/env python3
"""Full comparison: enriched vs original Gowalla friendship prediction on the SAME filtered subset.

For original LLM results: extracts matching entries from the full result files.
For original ML: retrains on the filtered data inline for a fair comparison.
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
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from src.gowalla.data_loader import FriendshipTestCase, load_test_cases
from src.gowalla.enrichment import load_category_map, compute_enriched_handcrafted


def compute_metrics_from_results(results: list[dict]) -> dict:
    valid = [r for r in results if r["prediction"] is not None]
    if not valid:
        return {"accuracy": 0, "f1": 0, "precision": 0, "recall": 0}
    tp = sum(1 for r in valid if r["prediction"] == 1 and r["label"] == 1)
    tn = sum(1 for r in valid if r["prediction"] == 0 and r["label"] == 0)
    fp = sum(1 for r in valid if r["prediction"] == 1 and r["label"] == 0)
    fn = sum(1 for r in valid if r["prediction"] == 0 and r["label"] == 1)
    acc = (tp + tn) / len(valid) if valid else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": round(acc, 4), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4)}


def compute_metrics_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    acc = (tp + tn) / len(y_true) if len(y_true) else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    return {"accuracy": round(acc, 4), "f1": round(f1, 4),
            "precision": round(prec, 4), "recall": round(rec, 4)}


def compute_orig_handcrafted(tc: FriendshipTestCase, granularity: str) -> list[float]:
    """Original handcrafted features (no categories)."""
    a, b = tc.user_a, tc.user_b
    feats = [float(a.total_checkins), float(b.total_checkins),
             float(a.total_checkins + b.total_checkins)]
    if granularity in ("G1", "G2", "G3", "G4"):
        feats.extend([float(tc.same_region), tc.centroid_distance_km,
                       float(a.primary_region == b.primary_region)])
    if granularity in ("G2", "G3", "G4"):
        feats.extend([float(a.unique_locations), float(b.unique_locations),
                       float(tc.shared_locations), tc.jaccard_locations])
    if granularity in ("G3", "G4"):
        feats.extend([a.geo_spread_km, b.geo_spread_km,
                       float(a.active_days), float(b.active_days)])
    if granularity == "G4":
        feats.extend([float(tc.temporal_co_occurrences),
                       tc.centroid_distance_km / max(a.geo_spread_km + b.geo_spread_km, 0.01),
                       tc.jaccard_locations * float(tc.temporal_co_occurrences + 1)])
    return feats


def train_handcrafted_ml(train_cases, test_cases, levels, feat_fn, label=""):
    train_labels = np.array([tc.label for tc in train_cases])
    test_labels = np.array([tc.label for tc in test_cases])
    results = {}
    for level in levels:
        X_train = np.array([feat_fn(tc, level) for tc in train_cases], dtype=np.float32)
        X_test = np.array([feat_fn(tc, level) for tc in test_cases], dtype=np.float32)

        dtrain = lgb.Dataset(X_train, label=train_labels)
        dval = lgb.Dataset(X_test, label=test_labels, reference=dtrain)
        bst = lgb.train(
            {"objective": "binary", "metric": "binary_logloss", "num_leaves": 31,
             "learning_rate": 0.05, "feature_fraction": 0.9, "verbose": -1},
            dtrain, num_boost_round=300, valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        preds_lgb = (bst.predict(X_test) > 0.5).astype(int)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                            early_stopping=True, n_iter_no_change=20, random_state=42)
        mlp.fit(X_tr_s, train_labels)
        preds_mlp = mlp.predict(X_te_s)

        results[level] = {
            "lgb": compute_metrics_arrays(test_labels, preds_lgb),
            "mlp": compute_metrics_arrays(test_labels, preds_mlp),
        }
        print(f"  {label} {level}: LGB={results[level]['lgb']['accuracy']:.4f}, "
              f"MLP={results[level]['mlp']['accuracy']:.4f} ({X_train.shape[1]} feats)")
    return results


def _val(d, keys, default="NA"):
    v = d
    for k in keys:
        if isinstance(v, dict):
            v = v.get(k, default)
        else:
            return default
    return v


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    ds = cfg["dataset"]
    enr = cfg["enrichment"]
    models = cfg["models"]
    levels = cfg["granularity_levels"]

    filt_dir = ROOT / enr["filtered_dir"]
    enr_results_dir = ROOT / enr["results_dir"]
    orig_baseline_dir = enr_results_dir / "original_baseline"

    cat_map = load_category_map(ROOT / enr["categories_path"])

    # --- Load filtered test cases ---
    filt_test = load_test_cases(filt_dir / "test_cases.json")
    filt_ml_train = load_test_cases(filt_dir / "ml_train_cases.json")
    filt_ml_test = load_test_cases(filt_dir / "ml_test_cases.json")
    print(f"Filtered LLM test: {len(filt_test)} pairs")
    print(f"Filtered ML: {len(filt_ml_train)} train, {len(filt_ml_test)} test\n")

    # --- Load original LLM baseline results (run on same filtered pairs) ---
    orig_llm: dict[str, dict[str, dict]] = {}
    for model in models:
        orig_llm[model] = {}
        for level in levels:
            path = orig_baseline_dir / f"{model}__{level}.json"
            if path.exists():
                orig_llm[model][level] = compute_metrics_from_results(json.loads(path.read_text()))
            else:
                orig_llm[model][level] = {"accuracy": "NA", "f1": "NA"}

    # --- Load enriched LLM results ---
    enr_llm: dict[str, dict[str, dict]] = {}
    for model in models:
        enr_llm[model] = {}
        for level in levels:
            path = enr_results_dir / f"{model}__{level}.json"
            if path.exists():
                enr_llm[model][level] = compute_metrics_from_results(json.loads(path.read_text()))
            else:
                enr_llm[model][level] = {"accuracy": "NA", "f1": "NA"}

    # --- Train original handcrafted ML on filtered data ---
    print("Training original handcrafted ML on filtered data...")
    orig_hc = train_handcrafted_ml(
        filt_ml_train, filt_ml_test, levels, compute_orig_handcrafted, "orig_hc"
    )

    # --- Train enriched handcrafted ML on filtered data ---
    print("\nTraining enriched handcrafted ML on filtered data...")
    enr_hc = train_handcrafted_ml(
        filt_ml_train, filt_ml_test, levels,
        lambda tc, g: compute_enriched_handcrafted(tc, g, cat_map), "enr_hc"
    )

    # --- Load enriched embed-v2 results if available ---
    enr_embed = {}
    embed_path = enr_results_dir / "enriched_embed_results.json"
    if embed_path.exists():
        enr_embed = json.loads(embed_path.read_text())

    # --- Print tables ---
    n_test = len(filt_test)
    n_ml = len(filt_ml_test)

    for metric in ["accuracy", "f1"]:
        label = "Accuracy" if metric == "accuracy" else "F1 Score"
        print(f"\n{'='*105}")
        print(f"GOWALLA ENRICHMENT COMPARISON — {label}  "
              f"(LLM: {n_test} pairs, ML: {n_ml} test / {len(filt_ml_train)} train)")
        print(f"{'='*105}")

        header = f"{'Method':<42}" + "".join(f"{l:>10}" for l in levels)
        print(header)
        print("-" * len(header))
        print(f"{'Random':<42}" + "".join(f"{'0.5000':>10}" for _ in levels))

        # --- Original section ---
        print(f"\n{'── ORIGINAL (raw coordinates) ──':<42}")
        for tag, key in [("LightGBM (handcrafted)", "lgb"), ("MLP (handcrafted)", "mlp")]:
            row = f"  {tag:<40}"
            for level in levels:
                v = _val(orig_hc, [level, key, metric])
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

        print(f"  {'─'*40}")
        for model in models:
            row = f"  {model:<40}"
            for level in levels:
                v = _val(orig_llm, [model, level, metric])
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

        # --- Enriched section ---
        print(f"\n{'── ENRICHED (+ venue categories) ──':<42}")
        for tag, key in [("LightGBM (handcrafted)", "lgb"), ("MLP (handcrafted)", "mlp")]:
            row = f"  {tag:<40}"
            for level in levels:
                v = _val(enr_hc, [level, key, metric])
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

        if enr_embed:
            for tag, key in [("LightGBM (embed-v2)", "lgb_embed"), ("MLP (embed-v2)", "mlp_embed")]:
                row = f"  {tag:<40}"
                for level in levels:
                    v = _val(enr_embed, [level, key, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

        print(f"  {'─'*40}")
        for model in models:
            row = f"  {model:<40}"
            for level in levels:
                v = _val(enr_llm, [model, level, metric])
                row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
            print(row)

    # --- Delta table ---
    print(f"\n{'='*105}")
    print("DELTA: Enriched minus Original  (positive = enrichment helped)")
    print(f"{'='*105}")

    for metric in ["accuracy", "f1"]:
        label = "Acc" if metric == "accuracy" else "F1"
        print(f"\n--- {label} ---")
        header = f"{'Method':<42}" + "".join(f"{l:>10}" for l in levels)
        print(header)
        print("-" * len(header))

        for tag, o_key, e_key in [
            ("LightGBM (handcrafted)", "lgb", "lgb"),
            ("MLP (handcrafted)", "mlp", "mlp"),
        ]:
            row = f"{tag:<42}"
            for level in levels:
                o = _val(orig_hc, [level, o_key, metric])
                e = _val(enr_hc, [level, e_key, metric])
                if isinstance(o, (int, float)) and isinstance(e, (int, float)):
                    d = e - o
                    row += f"{'+' if d >= 0 else ''}{d:>9.4f}"
                else:
                    row += f"{'NA':>10}"
            print(row)

        for model in models:
            row = f"{model:<42}"
            for level in levels:
                o = _val(orig_llm, [model, level, metric])
                e = _val(enr_llm, [model, level, metric])
                if isinstance(o, (int, float)) and isinstance(e, (int, float)):
                    d = e - o
                    row += f"{'+' if d >= 0 else ''}{d:>9.4f}"
                else:
                    row += f"{'NA':>10}"
            print(row)

    print(f"\n{'='*105}")

    # ================================================================
    # HARD NEGATIVES COMPARISON  (gpt-5-mini, gpt-5-nano only)
    # ================================================================
    reasoning_models = ["gpt-5-mini", "gpt-5-nano"]
    hard_neg_dir = ROOT / enr.get("hard_neg_results_dir", "results/gowalla_enriched/hard_neg")
    hard_neg_llm: dict[str, dict[str, dict]] = {}
    for model in reasoning_models:
        hard_neg_llm[model] = {}
        for level in levels:
            path = hard_neg_dir / f"{model}__{level}.json"
            if path.exists():
                hard_neg_llm[model][level] = compute_metrics_from_results(
                    json.loads(path.read_text()))
            else:
                hard_neg_llm[model][level] = {"accuracy": "NA", "f1": "NA"}

    has_hard_neg = any(
        isinstance(_val(hard_neg_llm, [m, l, "accuracy"]), float)
        for m in reasoning_models for l in levels
    )
    if has_hard_neg:
        for metric in ["accuracy", "f1"]:
            label = "Accuracy" if metric == "accuracy" else "F1 Score"
            print(f"\n{'='*105}")
            print(f"HARD NEGATIVES vs EASY NEGATIVES — {label}  "
                  f"(same-region non-friends vs random non-friends)")
            print(f"{'='*105}")
            header = f"{'Method':<42}" + "".join(f"{l:>10}" for l in levels)
            print(header)
            print("-" * len(header))

            for model in reasoning_models:
                row = f"  {model} (easy neg, zero-shot){'':<12}"[:42]
                for level in levels:
                    v = _val(enr_llm, [model, level, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

                row = f"  {model} (hard neg, zero-shot){'':<12}"[:42]
                for level in levels:
                    v = _val(hard_neg_llm, [model, level, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

                row = f"  {model} delta (hard - easy){'':<12}"[:42]
                for level in levels:
                    e = _val(enr_llm, [model, level, metric])
                    h = _val(hard_neg_llm, [model, level, metric])
                    if isinstance(e, (int, float)) and isinstance(h, (int, float)):
                        d = h - e
                        row += f"{'+' if d >= 0 else ''}{d:>9.4f}"
                    else:
                        row += f"{'NA':>10}"
                print(row)
                print()

    # ================================================================
    # FEW-SHOT vs ZERO-SHOT COMPARISON  (gpt-5-mini, gpt-5-nano only)
    # ================================================================
    fewshot_dir = ROOT / enr.get("fewshot_results_dir", "results/gowalla_enriched/fewshot")
    fewshot_llm: dict[str, dict[str, dict]] = {}
    for model in reasoning_models:
        fewshot_llm[model] = {}
        for level in levels:
            path = fewshot_dir / f"{model}__{level}.json"
            if path.exists():
                fewshot_llm[model][level] = compute_metrics_from_results(
                    json.loads(path.read_text()))
            else:
                fewshot_llm[model][level] = {"accuracy": "NA", "f1": "NA"}

    has_fewshot = any(
        isinstance(_val(fewshot_llm, [m, l, "accuracy"]), float)
        for m in reasoning_models for l in levels
    )
    n_ex = enr.get("fewshot_n_examples", 10)
    if has_fewshot:
        for metric in ["accuracy", "f1"]:
            label = "Accuracy" if metric == "accuracy" else "F1 Score"
            print(f"\n{'='*105}")
            print(f"FEW-SHOT ({n_ex} examples) vs ZERO-SHOT — {label}")
            print(f"{'='*105}")
            header = f"{'Method':<42}" + "".join(f"{l:>10}" for l in levels)
            print(header)
            print("-" * len(header))

            for model in reasoning_models:
                row = f"  {model} (zero-shot){'':<20}"[:42]
                for level in levels:
                    v = _val(enr_llm, [model, level, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

                row = f"  {model} ({n_ex}-shot){'':<20}"[:42]
                for level in levels:
                    v = _val(fewshot_llm, [model, level, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

                row = f"  {model} delta (few - zero){'':<12}"[:42]
                for level in levels:
                    z = _val(enr_llm, [model, level, metric])
                    f = _val(fewshot_llm, [model, level, metric])
                    if isinstance(z, (int, float)) and isinstance(f, (int, float)):
                        d = f - z
                        row += f"{'+' if d >= 0 else ''}{d:>9.4f}"
                    else:
                        row += f"{'NA':>10}"
                print(row)
                print()

    print(f"\n{'='*105}")

    # ================================================================
    # FINE-TUNING vs FEW-SHOT vs ZERO-SHOT  (gpt-4o-mini, gpt-4.1-mini)
    # ================================================================
    finetune_dir = ROOT / enr.get("finetune_results_dir", "results/gowalla_enriched/finetune")
    ft_models = enr.get("finetune_models", ["gpt-4o-mini", "gpt-4.1-mini"])
    ft_n_train = enr.get("finetune_n_train", 200)

    finetune_llm: dict[str, dict[str, dict]] = {}
    for model in ft_models:
        finetune_llm[model] = {}
        for level in levels:
            path = finetune_dir / f"{model}__{level}.json"
            if path.exists():
                finetune_llm[model][level] = compute_metrics_from_results(
                    json.loads(path.read_text()))
            else:
                finetune_llm[model][level] = {"accuracy": "NA", "f1": "NA"}

    has_finetune = any(
        isinstance(_val(finetune_llm, [m, l, "accuracy"]), float)
        for m in ft_models for l in levels
    )
    if has_finetune:
        for metric in ["accuracy", "f1"]:
            label = "Accuracy" if metric == "accuracy" else "F1 Score"
            print(f"\n{'='*105}")
            print(f"FINE-TUNED ({ft_n_train} examples) vs ZERO-SHOT vs FEW-SHOT — {label}")
            print(f"{'='*105}")
            header = f"{'Method':<50}" + "".join(f"{l:>10}" for l in levels)
            print(header)
            print("-" * len(header))

            # Supervised ML baselines for context
            for tag, key in [("LightGBM embed-v2 (20K supervised)", "lgb_embed")]:
                row = f"  {tag:<48}"
                for level in levels:
                    v = _val(enr_embed, [level, key, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)
            print(f"  {'─'*48}")

            for model in ft_models:
                # Zero-shot (from enriched LLM if available)
                zs_key = model
                if zs_key in enr_llm:
                    row = f"  {model} (zero-shot){'':<28}"[:50]
                    for level in levels:
                        v = _val(enr_llm, [zs_key, level, metric])
                        row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                    print(row)

                # Fine-tuned
                row = f"  {model} (fine-tuned, {ft_n_train} ex){'':<16}"[:50]
                for level in levels:
                    v = _val(finetune_llm, [model, level, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

                # Delta
                if zs_key in enr_llm:
                    row = f"  {model} delta (FT - zero){'':<22}"[:50]
                    for level in levels:
                        z = _val(enr_llm, [zs_key, level, metric])
                        f = _val(finetune_llm, [model, level, metric])
                        if isinstance(z, (int, float)) and isinstance(f, (int, float)):
                            d = f - z
                            row += f"{'+' if d >= 0 else ''}{d:>9.4f}"
                        else:
                            row += f"{'NA':>10}"
                    print(row)
                print()

            # Also show few-shot reasoning models for comparison
            if has_fewshot:
                print(f"  {'─'*48}")
                print(f"  {'(few-shot reasoning models for comparison)':<50}")
                for model in reasoning_models:
                    row = f"  {model} ({n_ex}-shot){'':<28}"[:50]
                    for level in levels:
                        v = _val(fewshot_llm, [model, level, metric])
                        row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                    print(row)

    print(f"\n{'='*105}")

    # ================================================================
    # LLAMA 3.1 8B vs OpenAI vs ML  (open-source comparison)
    # ================================================================
    llama_dir = enr_results_dir / "llama"
    llama_paradigms = {
        "zero-shot": "llama_zeroshot",
        "2-shot": "llama_fewshot",
        "FT 20K": "llama_finetuned",
    }
    llama_results: dict[str, dict[str, dict]] = {}
    for paradigm, prefix in llama_paradigms.items():
        llama_results[paradigm] = {}
        for level in levels:
            path = llama_dir / f"{prefix}__{level}.json"
            if path.exists():
                llama_results[paradigm][level] = compute_metrics_from_results(
                    json.loads(path.read_text()))
            else:
                llama_results[paradigm][level] = {"accuracy": "NA", "f1": "NA"}

    has_llama = any(
        isinstance(_val(llama_results, [p, l, "accuracy"]), float)
        for p in llama_paradigms for l in levels
    )
    if has_llama:
        for metric in ["accuracy", "f1"]:
            label = "Accuracy" if metric == "accuracy" else "F1 Score"
            print(f"\n{'='*105}")
            print(f"LLAMA 3.1 8B vs OpenAI vs ML — {label}")
            print(f"{'='*105}")
            header = f"{'Method':<50}" + "".join(f"{l:>10}" for l in levels)
            print(header)
            print("-" * len(header))

            for tag, key in [("LightGBM embed-v2 (20K supervised)", "lgb_embed")]:
                row = f"  {tag:<48}"
                for level in levels:
                    v = _val(enr_embed, [level, key, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)
            print(f"  {'─'*48}")

            for paradigm in llama_paradigms:
                row = f"  Llama 3.1 8B ({paradigm}){'':<22}"[:50]
                for level in levels:
                    v = _val(llama_results, [paradigm, level, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

            print(f"  {'─'*48}")
            print(f"  {'(OpenAI models for comparison)':<50}")

            if "gpt-4o-mini" in enr_llm:
                row = f"  gpt-4o-mini (zero-shot){'':<26}"[:50]
                for level in levels:
                    v = _val(enr_llm, ["gpt-4o-mini", level, metric])
                    row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                print(row)

            if has_fewshot:
                for model in reasoning_models:
                    row = f"  {model} ({n_ex}-shot){'':<28}"[:50]
                    for level in levels:
                        v = _val(fewshot_llm, [model, level, metric])
                        row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                    print(row)

            if has_finetune:
                for model in ft_models:
                    row = f"  {model} (FT, {ft_n_train} ex){'':<22}"[:50]
                    for level in levels:
                        v = _val(finetune_llm, [model, level, metric])
                        row += f"{v:>10}" if isinstance(v, str) else f"{v:>10.4f}"
                    print(row)

    print(f"\n{'='*105}")

    # Save all results
    summary = {
        "filtered_test_size": n_test,
        "filtered_ml_test_size": n_ml,
        "filtered_ml_train_size": len(filt_ml_train),
        "original_llm": orig_llm,
        "enriched_llm": enr_llm,
        "original_handcrafted": orig_hc,
        "enriched_handcrafted": enr_hc,
        "enriched_embed": enr_embed,
        "hard_neg_llm": hard_neg_llm,
        "fewshot_llm": fewshot_llm,
        "finetune_llm": finetune_llm,
        "llama_results": llama_results,
    }
    out = enr_results_dir / "enrichment_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nFull comparison saved to {out}")


if __name__ == "__main__":
    main()
