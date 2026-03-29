#!/usr/bin/env python3
"""Download and preprocess Gowalla data. Build train/test friendship pairs."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.gowalla.data_loader import (
    load_edges, load_checkins, build_user_profile,
    compute_pair_features, FriendshipTestCase, save_test_cases,
)


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    ds = cfg["dataset"]

    edges_path = ROOT / ds["edges_path"]
    checkins_path = ROOT / ds["checkins_path"]

    if not edges_path.exists() or not checkins_path.exists():
        print("Gowalla data not found. Download with:")
        print(f"  mkdir -p {edges_path.parent}")
        print(f"  wget -O {edges_path}.gz https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz")
        print(f"  wget -O {checkins_path}.gz https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz")
        print(f"  gunzip {edges_path}.gz {checkins_path}.gz")
        return

    print("Loading edges...")
    edges = load_edges(edges_path)
    print(f"  {len(edges)} undirected edges")

    print("Loading check-ins...")
    user_checkins = load_checkins(checkins_path)
    print(f"  {len(user_checkins)} users, {sum(len(v) for v in user_checkins.values())} check-ins")

    min_ci = ds["min_user_checkins"]
    eligible = {uid for uid, cis in user_checkins.items() if len(cis) >= min_ci}
    print(f"  {len(eligible)} users with >= {min_ci} check-ins")

    print("Building user profiles...")
    profiles = {uid: build_user_profile(uid, user_checkins[uid]) for uid in eligible}

    # Filter edges to eligible users
    eligible_edges = [(a, b) for a, b in edges if a in eligible and b in eligible]
    print(f"  {len(eligible_edges)} eligible friendship edges")

    random.seed(42)
    random.shuffle(eligible_edges)

    n_test_pos = ds["test_pairs"] // 2
    n_train_pos = ds["train_pairs"] // 2
    n_ml_test_pos = ds.get("ml_test_pairs", ds["test_pairs"]) // 2
    n_ml_train_pos = ds.get("ml_train_pairs", ds["train_pairs"]) // 2

    # LLM test/train use first slice; ML uses a larger separate slice
    test_edges = eligible_edges[:n_test_pos]
    train_edges = eligible_edges[n_test_pos : n_test_pos + n_train_pos]

    # ML gets its own larger pool (non-overlapping with LLM test)
    ml_offset = n_test_pos
    ml_test_edges = eligible_edges[ml_offset : ml_offset + n_ml_test_pos]
    ml_train_edges = eligible_edges[ml_offset + n_ml_test_pos : ml_offset + n_ml_test_pos + n_ml_train_pos]

    # Build friend lookup for negative sampling
    friend_set: dict[int, set[int]] = {}
    for a, b in edges:
        friend_set.setdefault(a, set()).add(b)
        friend_set.setdefault(b, set()).add(a)

    eligible_list = list(eligible)

    def sample_negatives(pos_edges: list[tuple[int, int]], n: int, seed: int = 0) -> list[tuple[int, int]]:
        rng = random.Random(42 + seed)
        negs = []
        seen: set[tuple[int, int]] = set()
        attempts = 0
        while len(negs) < n and attempts < n * 100:
            a = rng.choice(eligible_list)
            b = rng.choice(eligible_list)
            if a == b:
                attempts += 1
                continue
            pair = (min(a, b), max(a, b))
            if b in friend_set.get(a, set()):
                attempts += 1
                continue
            if pair not in seen:
                seen.add(pair)
                negs.append(pair)
            attempts += 1
        return negs[:n]

    def build_cases(pos: list, neg: list) -> list[FriendshipTestCase]:
        cases = []
        for a, b in pos:
            pf = compute_pair_features(profiles[a], profiles[b])
            tc = FriendshipTestCase(
                user_a=profiles[a], user_b=profiles[b], label=1, **pf
            )
            cases.append(tc)
        for a, b in neg:
            pf = compute_pair_features(profiles[a], profiles[b])
            tc = FriendshipTestCase(
                user_a=profiles[a], user_b=profiles[b], label=0, **pf
            )
            cases.append(tc)
        random.shuffle(cases)
        return cases

    out_dir = ROOT / ds["processed_dir"]

    # LLM test/train (keep existing if already there)
    llm_test_path = out_dir / "test_cases.json"
    llm_train_path = out_dir / "train_cases.json"
    if not llm_test_path.exists():
        print("Building LLM test cases...")
        test_neg = sample_negatives(test_edges, n_test_pos, seed=1)
        test_cases = build_cases(test_edges, test_neg)
        save_test_cases(test_cases, llm_test_path)
        p = sum(1 for tc in test_cases if tc.label == 1)
        print(f"  Saved {len(test_cases)} LLM test cases ({p} pos)")
    else:
        print(f"  LLM test_cases.json exists, skipping")

    if not llm_train_path.exists():
        print("Building LLM train cases...")
        train_neg = sample_negatives(train_edges, n_train_pos, seed=2)
        train_cases = build_cases(train_edges, train_neg)
        save_test_cases(train_cases, llm_train_path)
        p = sum(1 for tc in train_cases if tc.label == 1)
        print(f"  Saved {len(train_cases)} LLM train cases ({p} pos)")
    else:
        print(f"  LLM train_cases.json exists, skipping")

    # ML test/train (larger)
    print(f"\nBuilding ML datasets ({n_ml_test_pos*2} test, {n_ml_train_pos*2} train)...")
    print("  Sampling ML test negatives...")
    ml_test_neg = sample_negatives(ml_test_edges, n_ml_test_pos, seed=3)
    print("  Building ML test cases...")
    ml_test_cases = build_cases(ml_test_edges, ml_test_neg)
    save_test_cases(ml_test_cases, out_dir / "ml_test_cases.json")
    p = sum(1 for tc in ml_test_cases if tc.label == 1)
    print(f"  Saved {len(ml_test_cases)} ML test cases ({p} pos, {len(ml_test_cases)-p} neg)")

    print("  Sampling ML train negatives...")
    ml_train_neg = sample_negatives(ml_train_edges, n_ml_train_pos, seed=4)
    print("  Building ML train cases...")
    ml_train_cases = build_cases(ml_train_edges, ml_train_neg)
    save_test_cases(ml_train_cases, out_dir / "ml_train_cases.json")
    p = sum(1 for tc in ml_train_cases if tc.label == 1)
    print(f"  Saved {len(ml_train_cases)} ML train cases ({p} pos, {len(ml_train_cases)-p} neg)")

    print(f"\nOutput: {out_dir}/")


if __name__ == "__main__":
    main()
