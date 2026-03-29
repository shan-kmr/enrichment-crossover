#!/usr/bin/env python3
"""Collect KAR batch results for rating prediction, embed reasoning, save .npy.

Usage:
  python3 scripts/yelp_kar_batch_collect.py --status   # check batch progress
  python3 scripts/yelp_kar_batch_collect.py --wait      # poll until done, then process
  python3 scripts/yelp_kar_batch_collect.py             # process if already done
"""

from __future__ import annotations

import argparse
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

from src.yelp.data_loader import load_test_cases

EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
EMBED_BATCH = 100


def compute_handcrafted(tc) -> list[float]:
    history = tc.history
    if not history:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    stars = [r.stars for r in history]
    user_avg = float(np.mean(stars))
    user_std = float(np.std(stars)) if len(stars) > 1 else 0.0

    target_cats = {c.strip() for c in tc.target.categories.split(",") if c.strip()}
    user_cats = set(tc.user_profile.keys())
    overlap = len(target_cats & user_cats) / len(target_cats) if target_cats else 0.0

    cat_stars = [
        tc.user_profile[cat]["avg_stars"]
        for cat in target_cats
        if cat in tc.user_profile
    ]
    cat_avg = float(np.mean(cat_stars)) if cat_stars else user_avg

    city_counts: dict[str, int] = {}
    for r in history:
        city_counts[r.city] = city_counts.get(r.city, 0) + 1
    most_common = max(city_counts, key=city_counts.get) if city_counts else ""
    city_match = 1.0 if tc.target.city == most_common else 0.0

    return [user_avg, float(len(history)), user_std, overlap, cat_avg, city_match]


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


def check_batches(client: OpenAI, batch_ids: list[str]) -> bool:
    all_complete = True
    for bid in batch_ids:
        b = client.batches.retrieve(bid)
        rc = b.request_counts
        print(f"  {bid}: {b.status}  ({rc.completed}/{rc.total} done, {rc.failed} failed)")
        if b.status not in ("completed", "failed", "expired", "cancelled"):
            all_complete = False
    return all_complete


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true", help="Just print batch status")
    parser.add_argument("--wait", action="store_true", help="Poll until complete, then process")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    batch_dir = ROOT / "results" / "kar_batches"
    info = json.loads((batch_dir / "rating_batch_info.json").read_text())
    batch_ids = info["batch_ids"]

    client = OpenAI()

    print("Batch status:")
    all_done = check_batches(client, batch_ids)

    if args.status:
        return

    if not all_done:
        if args.wait:
            print("\nPolling every 60s until complete...")
            while not all_done:
                time.sleep(60)
                all_done = check_batches(client, batch_ids)
        else:
            print("\nBatches not ready. Use --wait to poll, or run --status to check.")
            return

    # --- Download results ---
    print("\nDownloading batch results...")
    results_map: dict[str, str] = {}
    for bid in batch_ids:
        b = client.batches.retrieve(bid)
        if b.status != "completed":
            print(f"  WARNING: batch {bid} status={b.status}, skipping")
            continue
        content = client.files.content(b.output_file_id)
        for line in content.text.strip().split("\n"):
            row = json.loads(line)
            cid = row["custom_id"]
            resp = row["response"]
            if resp["status_code"] == 200:
                text = resp["body"]["choices"][0]["message"]["content"] or "No reasoning."
            else:
                text = "No reasoning."
            results_map[cid] = text

    print(f"Collected {len(results_map)} reasoning outputs")

    # --- Load cases ---
    train_cases = load_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    test_cases = load_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")

    embed_client = OpenAI()
    kar_dir = ROOT / ds["processed_dir"] / "kar_embeddings"
    kar_dir.mkdir(parents=True, exist_ok=True)
    embed_cache = ROOT / "results" / "embed_cache"
    embed_cache.mkdir(parents=True, exist_ok=True)

    # --- Save labels + handcrafted + embeddings ---
    for split, cases in [("train", train_cases), ("test", test_cases)]:
        labels = np.array([tc.target.stars for tc in cases])
        np.save(kar_dir / f"{split}_labels.npy", labels)

        hc_path = kar_dir / f"{split}_handcrafted.npy"
        if not hc_path.exists():
            print(f"  Computing {split} handcrafted features...")
            hc = np.array([compute_handcrafted(tc) for tc in cases], dtype=np.float32)
            np.save(hc_path, hc)
        else:
            print(f"  {split}/handcrafted: exists")

        for level in levels:
            out_path = kar_dir / f"{split}_{level}_reasoning.npy"
            if out_path.exists():
                print(f"  {split}/{level}: exists, skipping")
                continue

            reasoning_texts = []
            missing = 0
            for i in range(len(cases)):
                key = f"{split}|{level}|{i:05d}"
                text = results_map.get(key)
                if text is None:
                    missing += 1
                    text = "No reasoning."
                reasoning_texts.append(text)

            if missing:
                print(f"  WARNING: {missing}/{len(cases)} missing for {split}/{level}")

            print(f"  Embedding {split}/{level}: {len(reasoning_texts)} texts...")
            embs = embed_texts(embed_client, reasoning_texts, embed_cache)
            np.save(out_path, np.array(embs, dtype=np.float32))
            print(f"    Saved {out_path.name}")

    print(f"\nDone. KAR embeddings at {kar_dir}/")
    print(f"Next: python3 scripts/yelp_kar_train.py")


if __name__ == "__main__":
    main()
