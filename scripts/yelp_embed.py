#!/usr/bin/env python3
"""Generate OpenAI embeddings (v2): text-embedding-3-large, dual user/biz split,
plus handcrafted numerical features."""

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

from src.yelp.data_loader import YelpTestCase, load_test_cases
from src.yelp.prompt_builder import build_prompt

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Context text builders
# ---------------------------------------------------------------------------

def build_user_context_text(test_case: YelpTestCase, granularity: str) -> str:
    """User context only (no target business, no task instruction)."""
    messages = build_prompt(test_case, granularity)
    user_content = messages[1]["content"]
    parts = user_content.split("\n\n")
    user_parts = [
        p for p in parts
        if not p.startswith("Target business:") and not p.startswith("Predict")
    ]
    return "\n\n".join(user_parts) if user_parts else "No user context available."


def build_business_text(test_case: YelpTestCase, granularity: str) -> str:
    """Target business description only."""
    messages = build_prompt(test_case, granularity)
    user_content = messages[1]["content"]
    for part in user_content.split("\n\n"):
        if part.startswith("Target business:"):
            return part
    include_coords = granularity == "G4"
    coords = f" at ({test_case.target.latitude:.4f}, {test_case.target.longitude:.4f})" if include_coords else ""
    return f'Target business: "{test_case.target.business_name}" (Categories: {test_case.target.categories}){coords}'


# ---------------------------------------------------------------------------
# Handcrafted features
# ---------------------------------------------------------------------------

def compute_handcrafted(tc: YelpTestCase) -> list[float]:
    history = tc.history
    if not history:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    stars = [r.stars for r in history]
    user_avg = float(np.mean(stars))
    user_count = float(len(history))
    user_std = float(np.std(stars)) if len(stars) > 1 else 0.0

    target_cats = {c.strip() for c in tc.target.categories.split(",") if c.strip()}
    user_cats = set(tc.user_profile.keys())
    overlap = len(target_cats & user_cats) / len(target_cats) if target_cats else 0.0

    cat_stars = [
        tc.user_profile[cat]["avg_stars"]
        for cat in target_cats if cat in tc.user_profile
    ]
    cat_avg = float(np.mean(cat_stars)) if cat_stars else user_avg

    city_counts: dict[str, int] = {}
    for r in history:
        city_counts[r.city] = city_counts.get(r.city, 0) + 1
    most_common_city = max(city_counts, key=city_counts.get) if city_counts else ""
    city_match = 1.0 if tc.target.city == most_common_city else 0.0

    return [user_avg, user_count, user_std, overlap, cat_avg, city_match]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

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
        for batch_start in tqdm(range(0, len(uncached), BATCH_SIZE), desc="    batches", leave=False):
            batch = uncached[batch_start : batch_start + BATCH_SIZE]
            batch_texts = [t for _, t in batch]

            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=batch_texts,
                dimensions=EMBED_DIM,
            )

            for (idx, text), emb_data in zip(batch, response.data):
                embedding = emb_data.embedding
                results[idx] = embedding
                key = hashlib.sha256(f"{EMBED_MODEL}:{EMBED_DIM}:{text}".encode()).hexdigest()
                (cache_dir / f"{key}.json").write_text(json.dumps(embedding))

            time.sleep(0.1)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    test_cases = load_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    train_cases = load_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test cases")
    print(f"Embedding model: {EMBED_MODEL}, dim: {EMBED_DIM}")

    client = OpenAI()
    embed_dir = ROOT / ds["processed_dir"] / "embeddings_v2"
    embed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "results" / "embed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split_name, cases in [("train", train_cases), ("test", test_cases)]:
        labels = np.array([tc.target.stars for tc in cases])
        np.save(embed_dir / f"{split_name}_labels.npy", labels)

        # Handcrafted features (granularity-independent)
        hc_path = embed_dir / f"{split_name}_handcrafted.npy"
        if not hc_path.exists():
            print(f"  Computing {split_name} handcrafted features...")
            hc = np.array([compute_handcrafted(tc) for tc in cases], dtype=np.float32)
            np.save(hc_path, hc)
            print(f"    Saved {hc.shape}")
        else:
            print(f"  {split_name}/handcrafted: already exists, skipping")

        # Dual embeddings per granularity level
        for level in levels:
            user_path = embed_dir / f"{split_name}_{level}_user.npy"
            biz_path = embed_dir / f"{split_name}_{level}_biz.npy"

            if user_path.exists() and biz_path.exists():
                print(f"  {split_name}/{level}: already exists, skipping")
                continue

            print(f"  Embedding {split_name}/{level} ({len(cases)} cases)...")

            user_texts = [build_user_context_text(tc, level) for tc in cases]
            biz_texts = [build_business_text(tc, level) for tc in cases]

            print(f"    User context embeddings...")
            user_embs = embed_texts(client, user_texts, cache_dir)
            np.save(user_path, np.array(user_embs, dtype=np.float32))

            print(f"    Business embeddings...")
            biz_embs = embed_texts(client, biz_texts, cache_dir)
            np.save(biz_path, np.array(biz_embs, dtype=np.float32))

    print(f"\nDone. Embeddings at {embed_dir}/")


if __name__ == "__main__":
    main()
