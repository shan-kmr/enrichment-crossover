#!/usr/bin/env python3
"""Generate enriched embeddings + handcrafted features for Gowalla friendship prediction.

Uses venue category information from Overture Maps enrichment.
Dual embedding: User A context + User B context separately (text-embedding-3-large).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from openai import OpenAI

from src.gowalla.data_loader import UserProfile, load_test_cases
from src.gowalla.enrichment import (
    load_category_map, get_category, format_category,
    user_category_profile, compute_enriched_handcrafted,
)

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
EMBED_BATCH = 100
MAX_HISTORY = 15


def build_user_text(profile: UserProfile, granularity: str, cat_map: dict) -> str:
    parts: list[str] = []
    cat_prof = user_category_profile(profile, cat_map)

    if granularity in ("G1", "G2", "G3", "G4"):
        parts.append(f"Region: {profile.primary_region}")
        top_str = ", ".join(
            f"{name} ({pct:.0f}%)"
            for name, _, pct in cat_prof["top_categories"][:5]
        )
        if top_str:
            parts.append(f"Top venues: {top_str}")

    if granularity in ("G2", "G3", "G4"):
        parts.append(
            f"Stats: {profile.total_checkins} check-ins, "
            f"{profile.unique_locations} unique locations, "
            f"{cat_prof['unique_categories']} venue categories, "
            f"spread {profile.geo_spread_km:.1f} km, "
            f"{profile.active_days} active days"
        )

    if granularity == "G3":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [
            f"{format_category(get_category(c.location_id, cat_map))} "
            f"({c.latitude:.4f},{c.longitude:.4f})"
            for c in recent
        ]
        parts.append("Recent: " + "; ".join(lines))

    if granularity == "G4":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [
            f"[{c.timestamp}] {format_category(get_category(c.location_id, cat_map))} "
            f"({c.latitude:.4f},{c.longitude:.4f})"
            for c in recent
        ]
        parts.append("Recent: " + "; ".join(lines))

    if not parts:
        parts.append(f"User with {profile.total_checkins} check-ins")

    return ". ".join(parts)


def embed_texts(client: OpenAI, texts: list[str], cache_dir: Path) -> list[list[float]]:
    results: list[list[float] | None] = [None] * len(texts)
    uncached: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        key = hashlib.sha256(f"{EMBED_MODEL}:{EMBED_DIM}:{text}".encode()).hexdigest()
        path = cache_dir / f"{key}.json"
        if path.exists():
            results[i] = json.loads(path.read_text())
        else:
            uncached.append((i, text))

    if uncached:
        for bs in tqdm(range(0, len(uncached), EMBED_BATCH), desc="    embed", leave=False):
            batch = uncached[bs : bs + EMBED_BATCH]
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=[t for _, t in batch],
                dimensions=EMBED_DIM,
            )
            for (idx, text), emb in zip(batch, resp.data):
                vec = emb.embedding
                results[idx] = vec
                key = hashlib.sha256(f"{EMBED_MODEL}:{EMBED_DIM}:{text}".encode()).hexdigest()
                (cache_dir / f"{key}.json").write_text(json.dumps(vec))
            time.sleep(0.1)

    return results


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    ds = cfg["dataset"]
    enr = cfg["enrichment"]
    levels = cfg["granularity_levels"]

    cat_map = load_category_map(ROOT / enr["categories_path"])
    print(f"Loaded category map: {len(cat_map)} locations")

    filt_dir = ROOT / enr["filtered_dir"]
    ml_train = filt_dir / "ml_train_cases.json"
    ml_test = filt_dir / "ml_test_cases.json"
    if ml_train.exists() and ml_test.exists():
        train_cases = load_test_cases(ml_train)
        test_cases = load_test_cases(ml_test)
        print(f"Loaded filtered ML datasets: {len(train_cases)} train, {len(test_cases)} test")
    else:
        train_cases = load_test_cases(filt_dir / "train_cases.json")
        test_cases = load_test_cases(filt_dir / "test_cases.json")
        print(f"Loaded filtered {len(train_cases)} train, {len(test_cases)} test cases")

    client = OpenAI()
    emb_dir = ROOT / enr["embeddings_dir"]
    emb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "results" / "embed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split, cases in [("train", train_cases), ("test", test_cases)]:
        labels = np.array([tc.label for tc in cases])
        np.save(emb_dir / f"{split}_labels.npy", labels)

        for level in levels:
            ua_path = emb_dir / f"{split}_{level}_user_a.npy"
            ub_path = emb_dir / f"{split}_{level}_user_b.npy"
            hc_path = emb_dir / f"{split}_{level}_handcrafted.npy"

            if ua_path.exists() and ub_path.exists() and hc_path.exists():
                print(f"  {split}/{level}: exists, skipping")
                continue

            print(f"\n  {split}/{level}: generating enriched embeddings + handcrafted...")

            texts_a = [build_user_text(tc.user_a, level, cat_map) for tc in cases]
            texts_b = [build_user_text(tc.user_b, level, cat_map) for tc in cases]

            print(f"    Embedding User A texts ({len(texts_a)})...")
            embs_a = embed_texts(client, texts_a, cache_dir)
            print(f"    Embedding User B texts ({len(texts_b)})...")
            embs_b = embed_texts(client, texts_b, cache_dir)

            np.save(ua_path, np.array(embs_a, dtype=np.float32))
            np.save(ub_path, np.array(embs_b, dtype=np.float32))

            hc = np.array(
                [compute_enriched_handcrafted(tc, level, cat_map) for tc in cases],
                dtype=np.float32,
            )
            np.save(hc_path, hc)
            print(f"    Saved: {ua_path.name}, {ub_path.name}, {hc_path.name} ({hc.shape[1]} feats)")

    print(f"\nDone. Enriched embeddings at {emb_dir}/")


if __name__ == "__main__":
    main()
