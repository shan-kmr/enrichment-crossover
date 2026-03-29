"""Compute Acc@K and MRR from experiment results."""

from __future__ import annotations

import numpy as np


def accuracy_at_k(rankings: list[list[int] | None], ground_truths: list[int], k: int) -> float:
    """Fraction of test cases where ground truth appears in the top-k of the ranking."""
    hits = 0
    valid = 0
    for ranking, gt in zip(rankings, ground_truths):
        if ranking is None:
            continue
        valid += 1
        if gt in ranking[:k]:
            hits += 1
    return hits / valid if valid > 0 else 0.0


def mrr(rankings: list[list[int] | None], ground_truths: list[int]) -> float:
    """Mean Reciprocal Rank: average of 1/rank for the ground truth item."""
    reciprocals = []
    for ranking, gt in zip(rankings, ground_truths):
        if ranking is None:
            continue
        if gt in ranking:
            rank = ranking.index(gt) + 1
            reciprocals.append(1.0 / rank)
        else:
            reciprocals.append(0.0)
    return float(np.mean(reciprocals)) if reciprocals else 0.0


def evaluate_condition(results: list[dict]) -> dict:
    """Compute all metrics for one (model, granularity) condition."""
    rankings = [r["ranking"] for r in results]
    ground_truths = [r["ground_truth_idx"] for r in results]

    n_valid = sum(1 for r in rankings if r is not None)
    n_failed = len(rankings) - n_valid

    total_tokens = sum(
        r.get("usage", {}).get("total_tokens", 0)
        for r in results if r.get("usage")
    )
    cached_count = sum(1 for r in results if r.get("cached"))

    return {
        "acc_at_1": accuracy_at_k(rankings, ground_truths, 1),
        "acc_at_5": accuracy_at_k(rankings, ground_truths, 5),
        "mrr": mrr(rankings, ground_truths),
        "n_total": len(results),
        "n_valid": n_valid,
        "n_parse_failures": n_failed,
        "total_tokens": total_tokens,
        "cached_responses": cached_count,
    }


def bootstrap_ci(
    rankings: list[list[int] | None],
    ground_truths: list[int],
    metric_fn,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Bootstrap confidence interval for a metric."""
    rng = np.random.RandomState(seed)
    n = len(rankings)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sampled_r = [rankings[i] for i in idx]
        sampled_g = [ground_truths[i] for i in idx]
        scores.append(metric_fn(sampled_r, sampled_g))
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(scores, 100 * alpha))
    hi = float(np.percentile(scores, 100 * (1 - alpha)))
    return lo, hi
