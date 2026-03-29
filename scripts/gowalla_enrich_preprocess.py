#!/usr/bin/env python3
"""Build balanced train/test friendship pairs from ONLY well-geocoded users.

Filters to users with ≥50% known Overture categories, then generates
balanced positive/negative pairs at the same sizes as the original experiment.
"""

from __future__ import annotations

import json
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
from src.gowalla.enrichment import get_category


def main():
    cfg = yaml.safe_load((ROOT / "gowalla_config.yaml").read_text())
    ds = cfg["dataset"]
    enr = cfg["enrichment"]
    min_known = enr.get("min_known_ratio", 0.50)

    cat_map = json.loads((ROOT / enr["categories_path"]).read_text())
    print(f"Loaded category map: {len(cat_map):,} locations")

    edges_path = ROOT / ds["edges_path"]
    checkins_path = ROOT / ds["checkins_path"]

    print("Loading edges...")
    edges = load_edges(edges_path)
    print(f"  {len(edges):,} undirected edges")

    print("Loading check-ins...")
    user_checkins = load_checkins(checkins_path)
    print(f"  {len(user_checkins):,} users")

    min_ci = ds["min_user_checkins"]
    eligible_base = {uid for uid, cis in user_checkins.items() if len(cis) >= min_ci}
    print(f"  {len(eligible_base):,} users with ≥{min_ci} check-ins")

    # Filter to well-geocoded users
    eligible = set()
    for uid in eligible_base:
        cis = user_checkins[uid]
        known = sum(1 for c in cis if get_category(c.location_id, cat_map) != "unknown")
        if known / len(cis) >= min_known:
            eligible.add(uid)
    print(f"  {len(eligible):,} users with ≥{min_known*100:.0f}% known categories")

    print("Building user profiles...")
    profiles = {uid: build_user_profile(uid, user_checkins[uid]) for uid in eligible}

    eligible_edges = [(a, b) for a, b in edges if a in eligible and b in eligible]
    print(f"  {len(eligible_edges):,} friendship edges between well-geocoded users")

    random.seed(42)
    random.shuffle(eligible_edges)

    n_test_pos = ds["test_pairs"] // 2
    n_train_pos = ds["train_pairs"] // 2
    n_ml_test_pos = ds.get("ml_test_pairs", ds["test_pairs"]) // 2
    n_ml_train_pos = ds.get("ml_train_pairs", ds["train_pairs"]) // 2

    test_edges = eligible_edges[:n_test_pos]
    train_edges = eligible_edges[n_test_pos : n_test_pos + n_train_pos]
    ml_offset = n_test_pos + n_train_pos
    ml_test_edges = eligible_edges[ml_offset : ml_offset + n_ml_test_pos]
    ml_train_edges = eligible_edges[ml_offset + n_ml_test_pos : ml_offset + n_ml_test_pos + n_ml_train_pos]

    print(f"  Allocated: {len(test_edges)} LLM test, {len(train_edges)} LLM train, "
          f"{len(ml_test_edges)} ML test, {len(ml_train_edges)} ML train positive edges")

    # Build friend lookup for negative sampling
    friend_set: dict[int, set[int]] = {}
    for a, b in edges:
        friend_set.setdefault(a, set()).add(b)
        friend_set.setdefault(b, set()).add(a)

    eligible_list = list(eligible)

    def sample_negatives(n: int, seed: int) -> list[tuple[int, int]]:
        rng = random.Random(42 + seed)
        negs: list[tuple[int, int]] = []
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
            cases.append(FriendshipTestCase(user_a=profiles[a], user_b=profiles[b], label=1, **pf))
        for a, b in neg:
            pf = compute_pair_features(profiles[a], profiles[b])
            cases.append(FriendshipTestCase(user_a=profiles[a], user_b=profiles[b], label=0, **pf))
        random.shuffle(cases)
        return cases

    out_dir = ROOT / enr["filtered_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, pos_edges, n_neg, seed in [
        ("test_cases.json", test_edges, n_test_pos, 1),
        ("train_cases.json", train_edges, n_train_pos, 2),
        ("ml_test_cases.json", ml_test_edges, n_ml_test_pos, 3),
        ("ml_train_cases.json", ml_train_edges, n_ml_train_pos, 4),
    ]:
        print(f"\n  {name}: {len(pos_edges)} pos + {n_neg} neg...")
        neg_edges = sample_negatives(n_neg, seed)
        cases = build_cases(pos_edges, neg_edges)
        save_test_cases(cases, out_dir / name)
        pos = sum(1 for tc in cases if tc.label == 1)
        print(f"    Saved {len(cases)} pairs ({pos} pos, {len(cases)-pos} neg)")

    print(f"\nOutput: {out_dir}/")
    print(f"All users have ≥{min_known*100:.0f}% Overture category coverage. Ready for enriched experiments.")


if __name__ == "__main__":
    main()
