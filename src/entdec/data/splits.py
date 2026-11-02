"""Per-user time based splits (PROJECT_SPEC.md section 7).

For every user, positive interactions are ordered by timestamp ascending:

* the last positive goes to **test**,
* the second to last goes to **validation**,
* everything else (including every non-positive interaction) is **train**.

Negative-feedback rows are never used as test or validation targets.
"""

from __future__ import annotations

import pandas as pd

from entdec.logging_setup import get_logger

logger = get_logger(__name__)

SPLIT_CATEGORIES = ("train", "valid", "test")


def assign_splits(
    interactions: pd.DataFrame,
    uid_col: str = "uid",
    ts_col: str = "ts",
    flag_col: str = "is_positive",
) -> pd.DataFrame:
    """Add a ``split`` column to an interaction frame.

    Args:
        interactions: Frame with user id, timestamp and a positive flag.
        uid_col: User id column name.
        ts_col: Timestamp column name (numeric, ascending = newer).
        flag_col: Boolean column marking positive interactions.

    Returns:
        A copy of ``interactions`` with a categorical ``split`` column.
    """
    if interactions.empty:
        frame = interactions.copy()
        frame["split"] = pd.Categorical([], categories=list(SPLIT_CATEGORIES))
        return frame

    frame = interactions.sort_values([uid_col, ts_col], kind="stable").reset_index(drop=True)
    labels = pd.Series("train", index=frame.index, dtype="object")

    positives = frame.index[frame[flag_col].to_numpy(dtype=bool)]
    if len(positives):
        # cumcount(ascending=False) is 0 for the newest positive of each user.
        recency_rank = (
            frame.loc[positives, [uid_col]]
            .groupby(uid_col, sort=False)
            .cumcount(ascending=False)
            .to_numpy()
        )
        test_idx = positives[recency_rank == 0]
        valid_idx = positives[recency_rank == 1]
        labels.loc[test_idx] = "test"
        labels.loc[valid_idx] = "valid"

    frame["split"] = pd.Categorical(labels.to_numpy(), categories=list(SPLIT_CATEGORIES))
    logger.info(
        "splits assigned: %s",
        {name: int((frame["split"] == name).sum()) for name in SPLIT_CATEGORIES},
    )
    return frame


def validate_no_leakage(
    interactions: pd.DataFrame,
    uid_col: str = "uid",
    item_col: str = "item_id",
    split_col: str = "split",
) -> pd.DataFrame:
    """Return users whose train items also appear in their test/validation sets.

    Ranking the full catalog minus the *train* items means a train item leaking
    into test would silently inflate the metrics.

    Returns:
        Frame with ``uid`` and ``leaked_items`` for every offending user
        (empty when the split is clean).
    """
    evals = interactions[interactions[split_col].isin(["test", "valid"])]
    trains = interactions[interactions[split_col] == "train"]
    if evals.empty or trains.empty:
        return pd.DataFrame(columns=[uid_col, "leaked_items"])

    merged = evals.merge(trains, on=[uid_col, item_col], suffixes=("", "_train"))
    if merged.empty:
        return pd.DataFrame(columns=[uid_col, "leaked_items"])
    leaked = (
        merged.groupby(uid_col, sort=False)[item_col]
        .agg(lambda values: sorted(set(values)))
        .reset_index()
        .rename(columns={item_col: "leaked_items"})
    )
    logger.warning("%d users have train/test item overlap", len(leaked))
    return leaked


def split_sizes(interactions: pd.DataFrame, split_col: str = "split") -> dict[str, int]:
    """Count rows per split."""
    counts = interactions[split_col].value_counts()
    return {name: int(counts.get(name, 0)) for name in SPLIT_CATEGORIES}


__all__ = ["SPLIT_CATEGORIES", "assign_splits", "split_sizes", "validate_no_leakage"]
