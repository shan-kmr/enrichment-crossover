#!/usr/bin/env python3
"""Build hard-negative test set: negatives are same-region non-friends.

Uses the same eligible (well-geocoded) user pool as the enriched experiment,
but samples negative pairs where both users share the same primary_region.
This makes the friendship prediction task significantly harder.
"""

from __future__ import annotations

import random
import json
import sys
from collections import defaultdict
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

    print("Loading edges...")
    edges = load_edges(ROOT / ds["edges_path"])
    print(f"  {len(edges):,} undirected edges")

    print("Loading check-ins...")
    user_checkins = load_checkins(ROOT / ds["checkins_path"])
    print(f"  {len(user_checkins):,} users")

    min_ci = ds["min_user_checkins"]
    eligible_base = {uid for uid, cis in user_checkins.items() if len(cis) >= min_ci}

    eligible = set()
    for uid in eligible_base:
        cis = user_checkins[uid]
        known = sum(1 for c in cis if get_category(c.location_id, cat_map) != "unknown")
        if known / len(cis) >= min_known:
            eligible.add(uid)
    print(f"  {len(eligible):,} well-geocoded users (≥{min_known*100:.0f}% known)")

    print("Building user profiles...")
    profiles = {uid: build_user_profile(uid, user_checkins[uid]) for uid in eligible}

    region_users: dict[str, list[int]] = defaultdict(list)
    for uid, prof in profiles.items():
        region_users[prof.primary_region].append(uid)

    big_regions = {r: uids for r, uids in region_users.items() if len(uids) >= 5}
    print(f"  {len(big_regions)} regions with ≥5 users "
          f"(total {sum(len(v) for v in big_regions.values()):,} users)")

    friend_set: dict[int, set[int]] = {}
    for a, b in edges:
        friend_set.setdefault(a, set()).add(b)
        friend_set.setdefault(b, set()).add(a)

    eligible_edges = [(a, b) for a, b in edges
                      if a in eligible and b in eligible]
    random.seed(42)
    random.shuffle(eligible_edges)

    n_test_pos = ds["test_pairs"] // 2
    test_pos_edges = eligible_edges[:n_test_pos]
    print(f"\n  Positive test edges: {len(test_pos_edges)}")

    # Sample hard negatives: same primary_region, not friends
    big_region_users = set()
    for uids in big_regions.values():
        big_region_users.update(uids)

    rng = random.Random(99)
    hard_negs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    attempts = 0
    target = n_test_pos

    region_list = list(big_regions.keys())
    while len(hard_negs) < target and attempts < target * 500:
        region = rng.choice(region_list)
        uids = big_regions[region]
        if len(uids) < 2:
            attempts += 1
            continue
        a, b = rng.sample(uids, 2)
        pair = (min(a, b), max(a, b))
        if pair in seen:
            attempts += 1
            continue
        if b in friend_set.get(a, set()):
            attempts += 1
            continue
        seen.add(pair)
        hard_negs.append((a, b))
        attempts += 1

    print(f"  Hard negatives sampled: {len(hard_negs)} (same-region non-friends)")

    # Build test cases
    cases: list[FriendshipTestCase] = []
    for a, b in test_pos_edges:
        pf = compute_pair_features(profiles[a], profiles[b])
        cases.append(FriendshipTestCase(user_a=profiles[a], user_b=profiles[b], label=1, **pf))
    for a, b in hard_negs:
        pf = compute_pair_features(profiles[a], profiles[b])
        cases.append(FriendshipTestCase(user_a=profiles[a], user_b=profiles[b], label=0, **pf))

    random.shuffle(cases)

    out_dir = ROOT / enr["hard_neg_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_test_cases(cases, out_dir / "test_cases.json")

    pos = sum(1 for tc in cases if tc.label == 1)
    neg = len(cases) - pos
    print(f"\n  Saved {len(cases)} hard-neg test cases ({pos} pos, {neg} neg) to {out_dir}/")

    # Stats on hard negatives vs easy negatives
    hn_dists = [tc.centroid_distance_km for tc in cases if tc.label == 0]
    hp_dists = [tc.centroid_distance_km for tc in cases if tc.label == 1]
    print(f"\n  Hard-neg centroid distances: mean={sum(hn_dists)/max(len(hn_dists),1):.1f} km, "
          f"median={sorted(hn_dists)[len(hn_dists)//2]:.1f} km")
    print(f"  Positive centroid distances: mean={sum(hp_dists)/max(len(hp_dists),1):.1f} km, "
          f"median={sorted(hp_dists)[len(hp_dists)//2]:.1f} km")


if __name__ == "__main__":
    main()
