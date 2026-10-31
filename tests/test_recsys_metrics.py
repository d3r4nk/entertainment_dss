"\"\"Hand-computed metric cases (section 7/11)."""

from __future__ import annotations

import numpy as np
import pytest

from entdec.recsys.metrics import (
    hit_rate_at_k,
    intra_list_diversity,
    item_coverage,
    ndcg_at_k,
    recall_at_k,
    theme_dissimilarity_fn,
)


def test_recall_and_hit_rate() -> None:
    ranked = [3, 7, 1, 9]
    assert recall_at_k(ranked, 7, 2) == 1.0
    assert recall_at_k(ranked, 7, 1) == 0.0
    assert recall_at_k(ranked, 7, 0) == 0.0
    assert hit_rate_at_k(ranked, 9, 4) == 1.0
    assert hit_rate_at_k(ranked, 9, 3) == 0.0


def test_ndcg_hand_value() -> None:
    # single ground truth at position 2: 1/log2(3) ~ 0.6309
    assert ndcg_at_k([5, 7, 9], 7, 10) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_k([5, 7, 9], 5, 10) == pytest.approx(1.0)
    assert ndcg_at_k([5, 7, 9], 42, 10) == 0.0
    # truncated before hit
    assert ndcg_at_k([5, 7, 9], 9, 2) == 0.0


def test_coverage_counts_distinct_items() -> None:
    lists = [[1, 2], [2, 3]]
    assert item_coverage(lists, catalog_size=4) == pytest.approx(3 / 4)
    assert item_coverage(lists, catalog_size=0) == 0.0


def test_intra_list_diversity_jaccard() -> None:
    sets = {1: frozenset({"a", "b"}), 2: frozenset({"a", "b"}), 3: frozenset({"c"})}
    fn = theme_dissimilarity_fn(sets)
    assert intra_list_diversity([1, 2], fn) == pytest.approx(0.0)   # identical themes
    assert intra_list_diversity([1, 3], fn) == pytest.approx(1.0)   # disjoint themes
    # A={a,b}, C={c}: 1 - 0/3 = 1.0; A,A,C: mean(0, 1, 1) = 2/3
    assert intra_list_diversity([1, 1, 3], fn) == pytest.approx(2 / 3)
    assert intra_list_diversity([1], fn) == 0.0
