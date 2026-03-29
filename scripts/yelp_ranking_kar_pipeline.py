#!/usr/bin/env python3
"""KAR-style reasoning embeddings for Yelp next-business ranking.

For each (user, candidate, granularity), ask gpt-5-nano for brief reasoning
about how likely the user is to visit that candidate, then embed the
reasoning with text-embedding-3-large.

Uses concurrent API calls for speed.

Outputs per granularity:
  {split}_{level}_reasoning.npy  -- shape (n_cases, n_cands, 1024)
  {split}_handcrafted.npy        -- shape (n_cases, n_cands, 6)
  {split}_gt_indices.npy         -- shape (n_cases,)
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

from src.yelp.data_loader import ReviewRecord
from src.yelp.ranking_builder import RankingTestCase, load_ranking_test_cases

REASONING_MODEL = "gpt-5-nano"
REASONING_MAX_TOKENS = 2048
EMBED_MODEL = "text-embedding-3-large"
EMBED_DIM = 1024
EMBED_BATCH = 100
CONCURRENT_WORKERS = 20
MAX_HISTORY = 15
KAR_TRAIN_LIMIT = 200  # cap training cases (each has 20 candidates)

KAR_SYSTEM = (
    "You are an expert at predicting user behavior. "
    "Given information about a user and a candidate business, "
    "reason briefly (1-2 sentences) about how likely this user is "
    "to visit this business next and why."
)


# ---------------------------------------------------------------------------
# Disk-based reasoning cache
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


def _call_reasoning(flat_idx: int, messages: list[dict]) -> tuple[int, str]:
    key = _cache_key(REASONING_MODEL, messages, REASONING_MAX_TOKENS)
    cached = _read_cache(key)
    if cached and cached.get("content"):
        return flat_idx, cached["content"]

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
            return flat_idx, content
        except Exception as e:
            delay = 30 * (2 ** attempt)
            print(f"  Worker error (attempt {attempt+1}/5): {e}, retrying in {delay}s")
            time.sleep(delay)

    return flat_idx, "No reasoning."


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _user_context(tc: RankingTestCase, granularity: str) -> str:
    parts: list[str] = []

    if granularity in ("G1", "G2", "G3", "G4"):
        city = tc.ground_truth.city
        if not city and tc.history:
            city = tc.history[-1].city
        parts.append(f"The user is located in {city or 'Unknown'}.")

    if granularity in ("G2", "G3", "G4") and tc.user_profile:
        sorted_cats = sorted(tc.user_profile.items(), key=lambda x: -x[1]["count"])
        lines = [
            f"  - {cat}: {info['count']} visits, avg {info['avg_stars']} stars"
            for cat, info in sorted_cats[:15]
        ]
        parts.append("The user's review history by category:\n" + "\n".join(lines))

    if granularity == "G3" and tc.history:
        recent = tc.history[-MAX_HISTORY:]
        lines = [
            f"  {i}. {r.business_name} ({r.categories}) -> {r.stars} stars"
            for i, r in enumerate(recent, 1)
        ]
        parts.append(f"Recent {len(recent)} visits:\n" + "\n".join(lines))

    if granularity == "G4" and tc.history:
        recent = tc.history[-MAX_HISTORY:]
        lines = [
            f"  {i}. [{r.date}] {r.business_name} ({r.categories}) "
            f"at ({r.latitude:.4f}, {r.longitude:.4f}) -> {r.stars} stars"
            for i, r in enumerate(recent, 1)
        ]
        parts.append(f"Recent {len(recent)} visits:\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "No user context available."


def _candidate_desc(cand: ReviewRecord, granularity: str) -> str:
    coords = (
        f" at ({cand.latitude:.4f}, {cand.longitude:.4f})"
        if granularity == "G4"
        else ""
    )
    return f'"{cand.business_name}" (Categories: {cand.categories}){coords}'


def build_kar_prompt(
    tc: RankingTestCase, cand: ReviewRecord, granularity: str
) -> list[dict[str, str]]:
    user_ctx = _user_context(tc, granularity)
    cand_desc = _candidate_desc(cand, granularity)
    user_msg = (
        f"{user_ctx}\n\n"
        f"Candidate business: {cand_desc}\n\n"
        f"Reason briefly about how likely this user is to visit this business next."
    )
    return [
        {"role": "system", "content": KAR_SYSTEM},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Handcrafted features (per user-candidate pair)
# ---------------------------------------------------------------------------

def compute_handcrafted(tc: RankingTestCase, cand: ReviewRecord) -> list[float]:
    history = tc.history
    if not history:
        return [0.0] * 6

    stars = [r.stars for r in history]
    user_avg = float(np.mean(stars))
    user_std = float(np.std(stars)) if len(stars) > 1 else 0.0

    cand_cats = {c.strip() for c in cand.categories.split(",") if c.strip()}
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
    most_common = max(city_counts, key=city_counts.get) if city_counts else ""
    city_match = 1.0 if cand.city == most_common else 0.0

    return [user_avg, float(len(history)), user_std, overlap, cat_avg, city_match]


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_texts(client: OpenAI, texts: list[str], cache_dir: Path) -> list[list[float]]:
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
# Main
# ---------------------------------------------------------------------------

def main():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((ROOT / "yelp_ranking_config.yaml").read_text())
    ds = cfg["dataset"]
    levels = cfg["granularity_levels"]

    train_cases = load_ranking_test_cases(
        ROOT / ds["processed_dir"] / "train_cases.json"
    )
    test_cases = load_ranking_test_cases(
        ROOT / ds["processed_dir"] / "test_cases.json"
    )

    if KAR_TRAIN_LIMIT and len(train_cases) > KAR_TRAIN_LIMIT:
        train_cases = train_cases[:KAR_TRAIN_LIMIT]
        print(f"Capped training cases to {KAR_TRAIN_LIMIT}")

    print(f"Using {len(train_cases)} train, {len(test_cases)} test ranking cases")
    print(f"Concurrency: {CONCURRENT_WORKERS} workers")

    embed_client = OpenAI()

    kar_dir = ROOT / ds["processed_dir"] / "kar_embeddings"
    kar_dir.mkdir(parents=True, exist_ok=True)
    embed_cache = ROOT / "results" / "embed_cache"
    embed_cache.mkdir(parents=True, exist_ok=True)

    n_cands = len(train_cases[0].candidates)

    for split, cases in [("train", train_cases), ("test", test_cases)]:
        gt = np.array([tc.ground_truth_idx for tc in cases])
        np.save(kar_dir / f"{split}_gt_indices.npy", gt)

        hc_path = kar_dir / f"{split}_handcrafted.npy"
        if not hc_path.exists():
            print(f"  Computing {split} handcrafted features...")
            hc = np.array(
                [
                    [compute_handcrafted(tc, c) for c in tc.candidates]
                    for tc in cases
                ],
                dtype=np.float32,
            )
            np.save(hc_path, hc)
            print(f"    Saved {hc.shape}")
        else:
            print(f"  {split}/handcrafted: exists")

        for level in levels:
            out_path = kar_dir / f"{split}_{level}_reasoning.npy"
            if out_path.exists():
                print(f"  {split}/{level}: exists, skipping")
                continue

            total = len(cases) * n_cands
            print(f"\n  KAR reasoning: {split}/{level} ({total} pairs)")

            # Build all (case, candidate) prompts with flat indices
            tasks: list[tuple[int, list[dict]]] = []
            for ci, tc in enumerate(cases):
                for cj, cand in enumerate(tc.candidates):
                    flat = ci * n_cands + cj
                    msgs = build_kar_prompt(tc, cand, level)
                    tasks.append((flat, msgs))

            reasoning_flat = ["No reasoning."] * total

            with ThreadPoolExecutor(
                max_workers=CONCURRENT_WORKERS, initializer=_init_worker
            ) as pool:
                futures = {
                    pool.submit(_call_reasoning, idx, msgs): idx
                    for idx, msgs in tasks
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc=f"    {level} reasoning",
                ):
                    idx, text = future.result()
                    reasoning_flat[idx] = text

            print(f"    Embedding {len(reasoning_flat)} reasoning texts...")
            embs_flat = embed_texts(embed_client, reasoning_flat, embed_cache)
            embs = np.array(embs_flat, dtype=np.float32).reshape(
                len(cases), n_cands, EMBED_DIM
            )
            np.save(out_path, embs)
            print(f"    Saved {out_path.name}")

    print(f"\nDone. KAR embeddings at {kar_dir}/")


if __name__ == "__main__":
    main()
