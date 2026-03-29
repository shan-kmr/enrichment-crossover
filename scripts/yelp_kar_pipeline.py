#!/usr/bin/env python3
"""KAR-style reasoning embeddings for Yelp star rating prediction.

For each (user, target, granularity), ask gpt-5-nano for brief reasoning
about what rating the user would give, then embed that reasoning with
text-embedding-3-large.  Saves .npy arrays for the training script.

Uses concurrent API calls for speed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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

REASONING_MODEL = "gpt-5-nano"
REASONING_MAX_TOKENS = 2048
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
EMBED_BATCH = 100
CONCURRENT_WORKERS = 20
KAR_TRAIN_LIMIT = 500  # cap training cases for speed (set None for all)

KAR_SYSTEM = (
    "You are an expert at analyzing user preferences. "
    "Given information about a user and a target business, "
    "reason briefly (2-3 sentences) about what star rating this user "
    "would likely give and why. Consider their history, category "
    "preferences, and alignment with the business."
)


# ---------------------------------------------------------------------------
# Disk-based reasoning cache (thread-safe for independent keys)
# ---------------------------------------------------------------------------

_CACHE_DIR: Path = ROOT / "results" / "kar_cache"


def _cache_key(model: str, messages: list[dict], max_tokens: int) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": 0.0, "max_tokens": max_tokens},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _read_cache(key: str) -> dict | None:
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _write_cache(key: str, data: dict) -> None:
    (_CACHE_DIR / f"{key}.json").write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# Single reasoning call (used by thread pool)
# ---------------------------------------------------------------------------

_oai_client: OpenAI | None = None


def _init_worker():
    global _oai_client
    _oai_client = OpenAI()


def _call_reasoning(idx: int, messages: list[dict]) -> tuple[int, str]:
    key = _cache_key(REASONING_MODEL, messages, REASONING_MAX_TOKENS)
    cached = _read_cache(key)
    if cached and cached.get("content"):
        return idx, cached["content"]

    for attempt in range(5):
        try:
            resp = _oai_client.chat.completions.create(
                model=REASONING_MODEL,
                messages=messages,
                max_completion_tokens=REASONING_MAX_TOKENS,
                timeout=120.0,
            )
            content = resp.choices[0].message.content
            if not content:
                continue  # don't cache empty; retry
            _write_cache(key, {"content": content, "model": resp.model})
            return idx, content
        except Exception as e:
            delay = 30 * (2 ** attempt)
            print(f"  Worker error (attempt {attempt+1}/5): {e}, retrying in {delay}s")
            time.sleep(delay)

    return idx, "No reasoning."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_kar_prompt(tc: YelpTestCase, granularity: str) -> list[dict[str, str]]:
    messages = build_prompt(tc, granularity)
    messages[0]["content"] = KAR_SYSTEM
    messages[1]["content"] = messages[1]["content"].replace(
        "Predict the star rating (1-5) this user would give. Return ONLY a single integer.",
        "Reason briefly about what star rating this user would give to this business and why.",
    )
    return messages


def compute_handcrafted(tc: YelpTestCase) -> list[float]:
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
    most_common_city = max(city_counts, key=city_counts.get) if city_counts else ""
    city_match = 1.0 if tc.target.city == most_common_city else 0.0

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
        for bs in tqdm(
            range(0, len(uncached), EMBED_BATCH),
            desc="    embed",
            leave=False,
        ):
            batch = uncached[bs : bs + EMBED_BATCH]
            resp = client.embeddings.create(
                model=EMBED_MODEL,
                input=[t for _, t in batch],
                dimensions=EMBED_DIM,
            )
            for (idx, text), emb in zip(batch, resp.data):
                vec = emb.embedding
                results[idx] = vec
                key = hashlib.sha256(
                    f"{EMBED_MODEL}:{EMBED_DIM}:{text}".encode()
                ).hexdigest()
                (cache_dir / f"{key}.json").write_text(json.dumps(vec))
            time.sleep(0.1)

    return results


# ---------------------------------------------------------------------------
# Concurrent reasoning for a list of cases
# ---------------------------------------------------------------------------

def reason_concurrently(cases: list[YelpTestCase], level: str) -> list[str]:
    tasks = [(i, build_kar_prompt(tc, level)) for i, tc in enumerate(cases)]
    results = ["No reasoning."] * len(cases)

    with ThreadPoolExecutor(
        max_workers=CONCURRENT_WORKERS, initializer=_init_worker
    ) as pool:
        futures = {
            pool.submit(_call_reasoning, idx, msgs): idx
            for idx, msgs in tasks
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"    {level} reasoning"
        ):
            idx, text = future.result()
            results[idx] = text

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((ROOT / "yelp_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    train_cases = load_test_cases(ROOT / ds["processed_dir"] / "train_cases.json")
    test_cases = load_test_cases(ROOT / ds["processed_dir"] / "test_cases.json")

    if KAR_TRAIN_LIMIT and len(train_cases) > KAR_TRAIN_LIMIT:
        train_cases = train_cases[:KAR_TRAIN_LIMIT]
        print(f"Capped training cases to {KAR_TRAIN_LIMIT}")

    print(f"Using {len(train_cases)} train, {len(test_cases)} test cases")
    print(f"Concurrency: {CONCURRENT_WORKERS} workers")

    embed_client = OpenAI()

    kar_dir = ROOT / ds["processed_dir"] / "kar_embeddings"
    kar_dir.mkdir(parents=True, exist_ok=True)
    embed_cache = ROOT / "results" / "embed_cache"
    embed_cache.mkdir(parents=True, exist_ok=True)

    for split, cases in [("train", train_cases), ("test", test_cases)]:
        labels = np.array([tc.target.stars for tc in cases])
        np.save(kar_dir / f"{split}_labels.npy", labels)

        hc_path = kar_dir / f"{split}_handcrafted.npy"
        if not hc_path.exists():
            print(f"  Computing {split} handcrafted features...")
            hc = np.array(
                [compute_handcrafted(tc) for tc in cases], dtype=np.float32
            )
            np.save(hc_path, hc)
        else:
            print(f"  {split}/handcrafted: exists")

        for level in levels:
            out_path = kar_dir / f"{split}_{level}_reasoning.npy"
            if out_path.exists():
                print(f"  {split}/{level}: exists, skipping")
                continue

            print(f"\n  KAR reasoning: {split}/{level} ({len(cases)} cases)")
            reasoning_texts = reason_concurrently(cases, level)

            print(f"    Embedding {len(reasoning_texts)} reasoning texts...")
            embs = embed_texts(embed_client, reasoning_texts, embed_cache)
            np.save(out_path, np.array(embs, dtype=np.float32))
            print(f"    Saved {out_path.name}")

    print(f"\nDone. KAR embeddings at {kar_dir}/")


if __name__ == "__main__":
    main()
