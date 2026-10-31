"""Ranking metrics (section 7). All are computed against *one* held-out item
per user, which is how the splits are materialised.

Hand-checkable cases live in ``tests/test_recsys_metrics.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], target: int, k: int) -> float:
    """1.0 when ``target`` is inside the first ``k`` (single ground truth)."""
    return float(target in list(ranked)[:k])


def ndcg_at_k(ranked: Sequence[int], target: int, k: int) -> float:
    """Single ground truth: log2-discounted gain, ideal = 1."""
    ranked = list(ranked)[:k]
    for position, item in enumerate(ranked, start=1):
        if item == target:
            return float(np.log2(2.0) / np.log2(position + 1))
    return 0.0


def hit_rate_at_k(ranked: Sequence[int], target: int, k: int) -> float:
    """Binary hit inside the first ``k``."""
    return recall_at_k(ranked, target, k)


def item_coverage(ranked_lists: Sequence[Sequence[int]], catalog_size: int) -> float:
    """Distinct recommended items / catalogue size."""
    if catalog_size == 0:
        return 0.0
    distinct: set[int] = set()
    for ranked in ranked_lists:
        distinct.update(int(i) for i in ranked)
    return len(distinct) / float(catalog_size)


def intra_list_diversity(
    ranked: Sequence[int], distance_fn, k: int | None = None
) -> float:
    """Mean pairwise *distance* inside a ranked list.

    ``distance_fn(a, b) -> float in [0, 1]`` is supplied by the caller - the
    theme Jaccard distance per the spec (higher = more diverse).
    """
    ranked = list(ranked)[: k if k is not None else len(ranked)]
    if len(ranked) < 2:
        return 0.0
    pairs = [
        (ranked[i], ranked[j])
        for i in range(len(ranked))
        for j in range(i + 1, len(ranked))
    ]
    return float(np.mean([float(distance_fn(a, b)) for a, b in pairs]))


def theme_dissimilarity_fn(theme_sets: dict[int, frozenset[str]]):
    """Jaccard-distance helper: 1 - |A∩B| / |A∪B|; missing themes -> full distance."""
    def distance(a: int, b: int) -> float:
        if a == b:
            return 0.0
        sa, sb = theme_sets.get(a, frozenset()), theme_sets.get(b, frozenset())
        union = sa | sb
        if not union:
            return 1.0
        return 1.0 - len(sa & sb) / len(union)

    return distance


__all__ = [
    "hit_rate_at_k",
    "intra_list_diversity",
    "item_coverage",
    "ndcg_at_k",
    "recall_at_k",
    "theme_dissimilarity_fn",
]
