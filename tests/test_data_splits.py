"""Tests for :mod:`entdec.data.splits`."""

from __future__ import annotations

import pandas as pd

from entdec.data.splits import assign_splits, split_sizes, validate_no_leakage


def _frame(rows: list[tuple[int, int, int, bool]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["uid", "item_id", "ts", "is_positive"])


def test_last_positive_is_test_second_last_is_valid() -> None:
    frame = _frame(
        [
            (1, 10, 100, True),
            (1, 11, 200, True),
            (1, 12, 300, True),
            (1, 13, 400, True),
        ]
    )
    out = assign_splits(frame).set_index("item_id")["split"]
    assert out.loc[13] == "test"
    assert out.loc[12] == "valid"
    assert out.loc[10] == "train"
    assert out.loc[11] == "train"


def test_negatives_never_enter_eval_sets() -> None:
    frame = _frame(
        [
            (1, 10, 100, True),
            (1, 11, 200, True),
            (1, 12, 500, False),  # newest row, but negative -> stays in train
        ]
    )
    out = assign_splits(frame).set_index("item_id")
    assert out.loc[12, "split"] == "train"
    assert out.loc[11, "split"] == "test"
    assert out.loc[10, "split"] == "valid"


def test_splits_are_per_user() -> None:
    frame = _frame(
        [
            (1, 10, 100, True),
            (2, 20, 110, True),
            (1, 11, 200, True),
            (2, 21, 210, True),
        ]
    )
    out = assign_splits(frame)
    per_user = out.groupby("uid")["split"].apply(lambda s: sorted(s.tolist()))
    assert per_user.loc[1] == ["test", "valid"]
    assert per_user.loc[2] == ["test", "valid"]


def test_single_positive_user_gets_test_only() -> None:
    out = assign_splits(_frame([(1, 10, 100, True)]))
    assert out["split"].tolist() == ["test"]


def test_out_of_order_input_is_sorted_by_timestamp() -> None:
    frame = _frame([(1, 12, 300, True), (1, 10, 100, True), (1, 11, 200, True)])
    out = assign_splits(frame).set_index("item_id")["split"]
    assert out.loc[12] == "test"
    assert out.loc[11] == "valid"
    assert out.loc[10] == "train"


def test_empty_frame() -> None:
    frame = _frame([])
    out = assign_splits(frame)
    assert out.empty
    assert "split" in out.columns


def test_split_sizes() -> None:
    out = assign_splits(_frame([(1, 10, 1, True), (1, 11, 2, True), (1, 12, 3, True)]))
    assert split_sizes(out) == {"train": 1, "valid": 1, "test": 1}


def test_no_leakage_detects_overlap() -> None:
    frame = pd.DataFrame(
        {
            "uid": [1, 1, 1],
            "item_id": [10, 10, 11],
            "split": ["train", "test", "train"],
        }
    )
    leaked = validate_no_leakage(frame)
    assert len(leaked) == 1
    assert leaked.iloc[0]["leaked_items"] == [10]


def test_clean_split_reports_no_leakage() -> None:
    frame = pd.DataFrame({"uid": [1, 1], "item_id": [10, 11], "split": ["train", "test"]})
    assert validate_no_leakage(frame).empty


def test_no_leakage_ignores_negatives_in_train_only_users() -> None:
    frame = pd.DataFrame(
        {"uid": [1, 1, 2], "item_id": [10, 11, 20], "split": ["train", "test", "train"]}
    )
    assert validate_no_leakage(frame).empty


def test_constrained_catalog_excludes_train_items() -> None:
    """Sanity check of the evaluation contract: train ids are hidden from ranking."""
    frame = pd.DataFrame(
        {
            "uid": [1, 1, 1, 1],
            "item_id": [10, 11, 12, 13],
            "ts": [1, 2, 3, 4],
            "is_positive": [True, True, True, True],
        }
    )
    out = assign_splits(frame)
    train_items = set(out.loc[out["split"] == "train", "item_id"])
    test_item = int(out.loc[out["split"] == "test", "item_id"].iloc[0])
    assert test_item not in train_items
