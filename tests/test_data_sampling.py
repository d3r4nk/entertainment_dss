"""Tests for :mod:`entdec.data.sampling`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entdec.config import ProfileConfig
from entdec.data.sampling import (
    accumulate_counts,
    counts_to_series,
    ids_to_python,
    select_items,
    select_users,
    user_stats_frame,
)


def profile(**kwargs) -> ProfileConfig:
    base = {
        "name": "test",
        "max_items": 3,
        "max_users": 2,
        "min_movie_ratings": 2,
        "min_game_reviews": 2,
        "min_interactions_per_user": 2,
        "min_positive_per_user": 3,
    }
    base.update(kwargs)
    return ProfileConfig(**base)


def test_accumulate_counts_grows() -> None:
    acc = accumulate_counts(None, np.array([1, 1, 3]))
    acc = accumulate_counts(acc, np.array([7, 3]))
    assert acc.tolist() == [0, 2, 0, 2, 0, 0, 0, 1]


def test_accumulate_counts_empty_input() -> None:
    assert accumulate_counts(None, np.array([], dtype=int)).size == 0
    acc = accumulate_counts(None, np.array([2]))
    assert accumulate_counts(acc, np.array([], dtype=int)).tolist() == [0, 0, 1]


def test_counts_to_series_drops_zero_ids() -> None:
    series = counts_to_series(np.array([0, 4, 0, 2]))
    assert series.to_dict() == {1: 4, 3: 2}


def test_select_items_respects_min_count_and_cap() -> None:
    counts = pd.Series({1: 10, 2: 10, 3: 9, 4: 1, 5: 50})
    chosen = select_items(counts, max_items=3, min_count=5)
    assert chosen.tolist() == [1, 2, 5]


def test_select_items_is_deterministic_on_ties() -> None:
    counts = pd.Series({5: 10, 3: 10, 4: 10})
    assert select_items(counts, max_items=2, min_count=1).tolist() == [3, 4]


def test_select_items_empty_when_threshold_too_high() -> None:
    assert select_items(pd.Series({1: 2, 2: 3}), max_items=10, min_count=100).size == 0


def test_user_stats_frame() -> None:
    frame = pd.DataFrame({"uid": [1, 1, 1, 2, 2], "is_positive": [True, False, True, True, True]})
    stats = user_stats_frame(frame).set_index("uid")
    assert stats.loc[1, "n_interactions"] == 3
    assert stats.loc[1, "n_positives"] == 2
    assert stats.loc[2, "n_positives"] == 2


def test_select_users_filters_and_orders() -> None:
    stats = pd.DataFrame(
        {
            "uid": [1, 2, 3, 4],
            "n_interactions": [10, 10, 10, 2],
            "n_positives": [9, 5, 3, 9],
        }
    )
    cfg = profile(max_users=2, min_interactions_per_user=3, min_positive_per_user=4)
    assert select_users(stats, cfg).tolist() == [1, 2]


def test_select_users_none_eligible() -> None:
    stats = pd.DataFrame({"uid": [1], "n_interactions": [1], "n_positives": [1]})
    cfg = profile(min_interactions_per_user=50)
    assert select_users(stats, cfg).size == 0


def test_ids_to_python() -> None:
    assert ids_to_python(np.array([1, 2])) == {1, 2}


@pytest.mark.parametrize("bad", [np.array([], dtype=int)])
def test_accumulate_counts_handles_empty(bad: np.ndarray) -> None:
    assert accumulate_counts(None, bad).size == 0
