"""Tests for :mod:`entdec.data.enrich`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entdec.data.enrich import (
    TAG_COLUMNS,
    cooccurrence_similarity,
    propagate_tags,
)


def _batch(rows: list[tuple[int, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["uid", "app_id", "is_recommended"])


def test_cooccurrence_is_cosine_normalised() -> None:
    item_ids = np.array([1, 2, 3])
    users_item1 = [(u, 1, True) for u in range(10)]
    users_item2 = [(u, 2, True) for u in range(10)]  # identical audience
    users_item3 = [(u, 3, True) for u in range(10, 20)]  # disjoint audience
    sim = cooccurrence_similarity(
        [_batch(users_item1 + users_item2 + users_item3)], item_ids, min_support=5
    )
    dense = sim.toarray()
    assert dense[0, 1] == pytest.approx(1.0, abs=1e-5)
    assert dense[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert (np.diag(dense) == 0).all()


def test_cooccurrence_gates_pure_popularity_with_lift() -> None:
    """A blockbuster co-reviewed with everything must not look similar to anything."""
    item_ids = np.array([1, 2, 3])
    rows = []
    # Item 1 is reviewed by everybody (100 users).
    rows += [(user, 1, True) for user in range(100)]
    # Items 2 and 3 are reviewed by 10 users each, disjoint audiences.
    rows += [(user, 2, True) for user in range(10)]
    rows += [(user, 3, True) for user in range(70, 80)]
    sim = cooccurrence_similarity([_batch(rows)], item_ids, min_support=5, min_lift=1.2)
    dense = sim.toarray()
    assert dense[0, 1] == 0.0  # lift(1,2) = 10*100/(100*10) = 1.0 < 1.2
    assert dense[0, 2] == 0.0


def test_cooccurrence_requires_min_support() -> None:
    item_ids = np.array([1, 2])
    rows = [(0, 1, True), (0, 2, True)]
    sim = cooccurrence_similarity([_batch(rows)], item_ids, min_support=5)
    assert sim.nnz == 0


def test_cooccurrence_ignores_negative_reviews() -> None:
    item_ids = np.array([1, 2])
    rows = [(1, 1, True), (1, 2, False), (2, 1, True), (2, 2, False)]
    sim = cooccurrence_similarity([_batch(rows)], item_ids)
    assert sim.nnz == 0


def test_cooccurrence_drops_power_users() -> None:
    item_ids = np.array([1, 2, 3, 4])
    rows = [(99, item, True) for item in item_ids]  # one user reviews everything
    sim = cooccurrence_similarity([_batch(rows)], item_ids, max_items_per_user=2, min_support=1)
    assert sim.nnz == 0


def test_cooccurrence_keeps_users_consistent_across_batches() -> None:
    """The same user split over two batches must still count as one user."""
    item_ids = np.array([1, 2])
    first = _batch([(7, 1, True), (8, 1, True), (9, 1, True), (10, 1, True), (11, 1, True)])
    second = _batch([(7, 2, True), (8, 2, True), (9, 2, True), (10, 2, True), (11, 2, True)])
    sim = cooccurrence_similarity([first, second], item_ids, min_support=5)
    assert sim.nnz == 2  # symmetric pair (0,1) and (1,0)
    assert sim.toarray()[0, 1] == pytest.approx(1.0, abs=1e-5)


def test_cooccurrence_handles_unknown_items() -> None:
    item_ids = np.array([1, 2])
    rows = [(1, 42, True), (2, 1, True)]
    sim = cooccurrence_similarity([_batch(rows)], item_ids)
    assert sim.shape == (2, 2)


def test_cooccurrence_empty_input() -> None:
    sim = cooccurrence_similarity([], np.array([1, 2]))
    assert sim.shape == (2, 2)


def _tagged() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "app_id": [1, 1, 2, 2],
            "tag": ["Roguelike", "Action", "Puzzle", "Co-op"],
            "rank": [1, 2, 1, 2],
            "weight": [1.0, 0.5, 1.0, 0.5],
            "source": ["steam"] * 4,
        }
    )


def test_propagate_tags_fills_only_untagged_items() -> None:
    item_ids = np.array([1, 2, 3])
    # Item 3 shares its audience with item 1 only.
    rows = []
    for user in range(10):
        rows += [(user, 1, True), (user, 3, True)]
    for user in range(10, 20):
        rows += [(user, 2, True)]
    sim = cooccurrence_similarity([_batch(rows)], item_ids, min_support=5)
    propagated = propagate_tags(_tagged(), sim, item_ids, min_similarity=0.01)
    assert set(propagated["app_id"]) == {3}
    assert set(propagated["tag"]) == {"Roguelike", "Action"}
    assert set(propagated["source"]) == {"propagated"}
    assert set(propagated.columns) == set(TAG_COLUMNS)
    assert propagated.iloc[0]["weight"] == pytest.approx(1.0)


def test_propagate_tags_respects_min_similarity() -> None:
    item_ids = np.array([1, 2])
    rows = [(0, 1, True), (0, 2, True)]
    sim = cooccurrence_similarity([_batch(rows)], item_ids, min_support=1)
    assert propagate_tags(_tagged(), sim, item_ids, min_similarity=0.99).empty


def test_propagate_tags_noop_when_everything_is_tagged() -> None:
    item_ids = np.array([1, 2])
    sim = cooccurrence_similarity([_batch([(0, 1, True), (0, 2, True)])], item_ids, min_support=1)
    table = _tagged()
    assert propagate_tags(table, sim, item_ids).empty


def test_propagate_tags_empty_input() -> None:
    assert propagate_tags(pd.DataFrame(columns=list(TAG_COLUMNS)), None, np.array([1])).empty
