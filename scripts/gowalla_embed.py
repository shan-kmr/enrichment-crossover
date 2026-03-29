#!/usr/bin/env python3
"""Generate embeddings + handcrafted features for Gowalla friendship prediction.

For each (user_pair, granularity), embed User A's context and User B's context
separately with text-embedding-3-large, and compute handcrafted pair features.
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

from src.gowalla.data_loader import FriendshipTestCase, UserProfile, load_test_cases

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
EMBED_BATCH = 100
MAX_HISTORY = 15


def build_user_text(profile: UserProfile, granularity: str) -> str:
    parts: list[str] = []
    if granularity in ("G1", "G2", "G3", "G4"):
        parts.append(f"Region: {profile.primary_region}")
    if granularity in ("G2", "G3", "G4"):
        parts.append(
            f"Stats: {profile.total_checkins} check-ins, "
            f"{profile.unique_locations} unique locations, "
            f"spread {profile.geo_spread_km:.1f} km, "
            f"{profile.active_days} active days"
        )
    if granularity == "G3":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [f"Loc{c.location_id} ({c.latitude:.4f},{c.longitude:.4f})" for c in recent]
        parts.append("Recent: " + "; ".join(lines))
    if granularity == "G4":
        recent = profile.checkins[-MAX_HISTORY:]
        lines = [f"[{c.timestamp}] Loc{c.location_id} ({c.latitude:.4f},{c.longitude:.4f})" for c in recent]
        parts.append("Recent: " + "; ".join(lines))
    if not parts:
        parts.append(f"User with {profile.total_checkins} check-ins")
    return ". ".join(parts)


def compute_handcrafted(tc: FriendshipTestCase, granularity: str) -> list[float]:
    a, b = tc.user_a, tc.user_b

    # G0: basic pair stats (3 features)
    feats = [
        float(a.total_checkins),
        float(b.total_checkins),
        float(a.total_checkins + b.total_checkins),
    ]

    if granularity in ("G1", "G2", "G3", "G4"):
        # G1: region-level (+ 3 = 6)
        feats.extend([
            float(tc.same_region),
            tc.centroid_distance_km,
            float(a.primary_region == b.primary_region),
        ])

    if granularity in ("G2", "G3", "G4"):
        # G2: overlap stats (+ 4 = 10)
        feats.extend([
            float(a.unique_locations),
            float(b.unique_locations),
            float(tc.shared_locations),
            tc.jaccard_locations,
        ])

    if granularity in ("G3", "G4"):
        # G3: trajectory features (+ 4 = 14)
        feats.extend([
            a.geo_spread_km,
            b.geo_spread_km,
            float(a.active_days),
            float(b.active_days),
        ])

    if granularity == "G4":
        # G4: temporal co-occurrence (+ 3 = 17)
        feats.extend([
            float(tc.temporal_co_occurrences),
            tc.centroid_distance_km / max(a.geo_spread_km + b.geo_spread_km, 0.01),
            tc.jaccard_locations * float(tc.temporal_co_occurrences + 1),
        ])

    return feats


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
    levels = cfg["granularity_levels"]

    ml_train = ROOT / ds["processed_dir"] / "ml_train_cases.json"
    ml_test = ROOT / ds["processed_dir"] / "ml_test_cases.json"
    if ml_train.exists() and ml_test.exists():
        train_cases = load_test_cases(ml_train)
        test_cases = load_test_cases(ml_test)
        print(f"Loaded ML datasets: {len(train_cases)} train, {len(test_cases)} test")
    else:
        train_cases = load_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
        test_cases = load_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")
        print(f"Loaded {len(train_cases)} train, {len(test_cases)} test cases")

    client = OpenAI()
    emb_dir = ROOT / ds["processed_dir"] / "embeddings_v2"
    emb_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = ROOT / "results" / "embed_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split, cases in [("train", train_cases), ("test", test_cases)]:
        # Labels
        labels = np.array([tc.label for tc in cases])
        np.save(emb_dir / f"{split}_labels.npy", labels)

        for level in levels:
            ua_path = emb_dir / f"{split}_{level}_user_a.npy"
            ub_path = emb_dir / f"{split}_{level}_user_b.npy"
            hc_path = emb_dir / f"{split}_{level}_handcrafted.npy"

            if ua_path.exists() and ub_path.exists() and hc_path.exists():
                print(f"  {split}/{level}: exists, skipping")
                continue

            print(f"\n  {split}/{level}: generating embeddings + handcrafted...")

            texts_a = [build_user_text(tc.user_a, level) for tc in cases]
            texts_b = [build_user_text(tc.user_b, level) for tc in cases]

            print(f"    Embedding User A texts ({len(texts_a)})...")
            embs_a = embed_texts(client, texts_a, cache_dir)
            print(f"    Embedding User B texts ({len(texts_b)})...")
            embs_b = embed_texts(client, texts_b, cache_dir)

            np.save(ua_path, np.array(embs_a, dtype=np.float32))
            np.save(ub_path, np.array(embs_b, dtype=np.float32))

            hc = np.array([compute_handcrafted(tc, level) for tc in cases], dtype=np.float32)
            np.save(hc_path, hc)
            print(f"    Saved: {ua_path.name}, {ub_path.name}, {hc_path.name} ({hc.shape[1]} feats)")

    print(f"\nDone. Embeddings at {emb_dir}/")


if __name__ == "__main__":
    main()
