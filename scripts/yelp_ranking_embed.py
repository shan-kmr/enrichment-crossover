#!/usr/bin/env python3
"""Generate dual OpenAI embeddings for ranking task: user context + each candidate.

For each (case, granularity) produces:
  - 1 user context embedding (varies by granularity)
  - N candidate embeddings (same across granularities except G4 which adds coords)

Outputs saved to data/processed/yelp_ranking/embeddings_v2/ as .npy files.
Also computes handcrafted features per (user, candidate) pair.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY in your environment (see .env.example)"

from openai import OpenAI

from src.yelp.data_loader import ReviewRecord
from src.yelp.ranking_builder import RankingTestCase, load_ranking_test_cases

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
BATCH_SIZE = 100
MAX_HISTORY = 15


# ---------------------------------------------------------------------------
# Text builders
# ---------------------------------------------------------------------------

def build_user_context_text(tc: RankingTestCase, granularity: str) -> str:
    """User context only (no candidates, no task instruction)."""
    parts: list[str] = []

    if granularity in ("G1", "G2", "G3", "G4"):
        city = tc.ground_truth.city
        if not city and tc.history:
            city = tc.history[-1].city
        parts.append(f"User located in {city or 'Unknown'}.")

    if granularity in ("G2", "G3", "G4") and tc.user_profile:
        sorted_cats = sorted(tc.user_profile.items(), key=lambda x: -x[1]["count"])
        lines = [
            f"{cat}: {info['count']} visits, avg {info['avg_stars']} stars"
            for cat, info in sorted_cats[:15]
        ]
        parts.append("Category history: " + "; ".join(lines))

    if granularity == "G3" and tc.history:
        recent = tc.history[-MAX_HISTORY:]
        lines = [
            f"{r.business_name} ({r.categories}) -> {r.stars} stars"
            for r in recent
        ]
        parts.append("Recent visits: " + "; ".join(lines))

    if granularity == "G4" and tc.history:
        recent = tc.history[-MAX_HISTORY:]
        lines = [
            f"[{r.date}] {r.business_name} ({r.categories}) at ({r.latitude:.4f},{r.longitude:.4f}) -> {r.stars} stars"
            for r in recent
        ]
        parts.append("Recent visits: " + "; ".join(lines))

    return " ".join(parts) if parts else "No user context available."


def build_candidate_text(candidate: ReviewRecord, granularity: str) -> str:
    """Single candidate business description."""
    coords = (
        f" at ({candidate.latitude:.4f}, {candidate.longitude:.4f})"
        if granularity == "G4"
        else ""
    )
    return f'"{candidate.business_name}" (Categories: {candidate.categories}){coords}'


# ---------------------------------------------------------------------------
# Handcrafted features (per user-candidate pair)
# ---------------------------------------------------------------------------

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


def compute_handcrafted(tc: RankingTestCase, candidate: ReviewRecord) -> list[float]:
    history = tc.history
    if not history:
        return [0.0] * 6

    stars = [r.stars for r in history]
    user_avg = float(np.mean(stars))
    user_count = float(len(history))
    user_std = float(np.std(stars)) if len(stars) > 1 else 0.0

    cand_cats = {c.strip() for c in candidate.categories.split(",") if c.strip()}
    user_cats = set(tc.user_profile.keys())
    overlap = len(cand_cats & user_cats) / len(cand_cats) if cand_cats else 0.0

    cat_stars = [
        tc.user_profile[cat]["avg_stars"]
        for cat in cand_cats
        if cat in tc.user_profile
    ]
    cat_avg = float(np.mean(cat_stars)) if cat_stars else user_avg

    city_counts: dict[str, int] = {}
    for r in history:
        city_counts[r.city] = city_counts.get(r.city, 0) + 1
    most_common_city = max(city_counts, key=city_counts.get) if city_counts else ""
    city_match = 1.0 if candidate.city == most_common_city else 0.0

    return [user_avg, user_count, user_std, overlap, cat_avg, city_match]


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def embed_texts(
    client: OpenAI, texts: list[str], cache_dir: Path
) -> list[list[float]]:
    results: list[list[float] | None] = [None] * len(texts)
    uncached: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        key = hashlib.sha256(
            f"{EMBED_MODEL}:{EMBED_DIM}:{text}".encode()
        ).hexdigest()
        path = cache_dir / f"{key}.json"
        if path.exists():
            results[i] = json.loads(path.read_text())
        else:
            uncached.append((i, text))

    if uncached:
        for batch_start in tqdm(
            range(0, len(uncached), BATCH_SIZE), desc="    batches", leave=False
        ):
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
                key = hashlib.sha256(
                    f"{EMBED_MODEL}:{EMBED_DIM}:{text}".encode()
                ).hexdigest()
                (cache_dir / f"{key}.json").write_text(json.dumps(embedding))

            time.sleep(0.1)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    test_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
    train_cases = load_ranking_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    print(f"Loaded {len(train_cases)} train, {len(test_cases)} test ranking cases")
    print(f"Embedding model: {EMBED_MODEL}, dim: {EMBED_DIM}")

    client = OpenAI()
    embed_dir = ROOT / ds["processed_dir"] / "embeddings_v2"
    embed_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "results" / "embed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split_name, cases in [("train", train_cases), ("test", test_cases)]:
        n_cands = len(cases[0].candidates)

        # Labels: ground truth index per case
        gt_indices = np.array([tc.ground_truth_idx for tc in cases])
        np.save(embed_dir / f"{split_name}_gt_indices.npy", gt_indices)

        # Handcrafted features: shape (n_cases, n_cands, 6)
        hc_path = embed_dir / f"{split_name}_handcrafted.npy"
        if not hc_path.exists():
            print(f"  Computing {split_name} handcrafted features...")
            hc = np.array(
                [
                    [compute_handcrafted(tc, cand) for cand in tc.candidates]
                    for tc in cases
                ],
                dtype=np.float32,
            )
            np.save(hc_path, hc)
            print(f"    Saved {hc.shape}")
        else:
            print(f"  {split_name}/handcrafted: exists, skipping")

        for level in levels:
            user_path = embed_dir / f"{split_name}_{level}_user.npy"
            cand_path = embed_dir / f"{split_name}_{level}_cand.npy"

            if user_path.exists() and cand_path.exists():
                print(f"  {split_name}/{level}: exists, skipping")
                continue

            print(f"  Embedding {split_name}/{level}...")

            # User context embeddings: 1 per case
            user_texts = [build_user_context_text(tc, level) for tc in cases]
            print(f"    User context ({len(user_texts)} texts)...")
            user_embs = embed_texts(client, user_texts, cache_dir)
            np.save(user_path, np.array(user_embs, dtype=np.float32))

            # Candidate embeddings: n_cands per case (flattened, then reshaped)
            cand_texts = [
                build_candidate_text(cand, level)
                for tc in cases
                for cand in tc.candidates
            ]
            print(f"    Candidate ({len(cand_texts)} texts)...")
            cand_embs_flat = embed_texts(client, cand_texts, cache_dir)
            cand_embs = np.array(cand_embs_flat, dtype=np.float32).reshape(
                len(cases), n_cands, EMBED_DIM
            )
            np.save(cand_path, cand_embs)

    print(f"\nDone. Embeddings at {embed_dir}/")


if __name__ == "__main__":
    main()
