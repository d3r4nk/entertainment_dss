"""Sampling helpers (PROJECT_SPEC.md section 3).

The pipeline is deterministic: given the same raw data and profile, the same
items and users are selected. Ties are broken by ascending id.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from entdec.config import ProfileConfig
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

ID_TYPES = (np.int32, np.int64, np.uint32, np.uint64)


def accumulate_counts(acc: np.ndarray | None, ids: np.ndarray) -> np.ndarray:
    """Add a batch of ids into a counting array, growing it when needed.

    ``np.bincount`` grows to the largest id seen, so no id range has to be
    known up front and no per-user dictionary is needed for 14M+ users.

    Args:
        acc: Running counts (``int64``) or ``None`` for the first batch.
        ids: Non-negative integer ids.

    Returns:
        The updated counts array (index == id).
    """
    if ids.size == 0:
        return acc if acc is not None else np.zeros(0, dtype=np.int64)
    ids = np.asarray(ids)
    if ids.dtype not in ID_TYPES:
        ids = ids.astype(np.int64)
    batch_counts = np.bincount(ids, minlength=int(ids.max()) + 1)
    if acc is None or acc.size < batch_counts.size:
        grown = np.zeros(batch_counts.size, dtype=np.int64)
        if acc is not None:
            grown[: acc.size] = acc
        acc = grown
    acc[: batch_counts.size] += batch_counts
    return acc


def counts_to_series(acc: np.ndarray | None, name: str = "count") -> pd.Series:
    """Convert an id-indexed counts array into a dense Series over observed ids."""
    if acc is None or acc.size == 0:
        return pd.Series(dtype="int64", name=name)
    observed = np.nonzero(acc)[0]
    return pd.Series(acc[observed], index=observed, name=name)


def select_items(counts: pd.Series, max_items: int, min_count: int) -> np.ndarray:
    """Select the most frequent items, deterministically.

    Args:
        counts: Interaction count per item id.
        max_items: Maximum number of items to keep.
        min_count: Minimum number of interactions an item must have.

    Returns:
        Sorted array of selected item ids.
    """
    eligible = counts[counts >= min_count]
    if eligible.empty:
        logger.warning("no item has >= %d interactions", min_count)
        return np.zeros(0, dtype=np.int64)
    ordered = eligible.sort_values(ascending=False, kind="stable")
    # Secondary sort by id ascending for determinism on equal counts.
    frame = ordered.rename("count").reset_index()
    frame.columns = ["item", "count"]
    frame = frame.sort_values(["count", "item"], ascending=[False, True], kind="stable")
    chosen = frame["item"].head(max_items).to_numpy()
    logger.info(
        "selected %d items (>= %d interactions) out of %d eligible",
        chosen.size,
        min_count,
        eligible.size,
    )
    return np.sort(chosen)


def user_stats_frame(interactions: pd.DataFrame, uid_col: str = "uid") -> pd.DataFrame:
    """Per-user interaction and positive-interaction counts.

    Args:
        interactions: Frame holding at least ``uid_col`` and ``is_positive``.
        uid_col: Name of the user id column.

    Returns:
        Frame indexed by ``uid`` with ``n_interactions`` and ``n_positives``.
    """
    grouped = interactions.groupby(uid_col, sort=False)["is_positive"]
    stats = pd.DataFrame(
        {"n_interactions": grouped.size(), "n_positives": grouped.sum()}
    ).reset_index()
    stats["n_positives"] = stats["n_positives"].astype("int64")
    stats["n_interactions"] = stats["n_interactions"].astype("int64")
    return stats


def select_users(
    stats: pd.DataFrame,
    profile: ProfileConfig,
    uid_col: str = "uid",
) -> np.ndarray:
    """Select the users kept in the sampled graph.

    Users must have at least ``min_interactions_per_user`` interactions and
    ``min_positive_per_user`` positive interactions; among those, the most
    active (by positives, then interactions, then lowest id) win.

    Args:
        stats: Output of :func:`user_stats_frame`.
        profile: Active sampling profile.
        uid_col: Name of the user id column.

    Returns:
        Sorted array of selected user ids.
    """
    eligible = stats[
        (stats["n_interactions"] >= profile.min_interactions_per_user)
        & (stats["n_positives"] >= profile.min_positive_per_user)
    ]
    if eligible.empty:
        logger.warning(
            "no user has >= %d interactions and >= %d positives",
            profile.min_interactions_per_user,
            profile.min_positive_per_user,
        )
        return np.zeros(0, dtype=np.int64)
    ordered = eligible.sort_values(
        ["n_positives", "n_interactions", uid_col], ascending=[False, False, True], kind="stable"
    )
    chosen = ordered[uid_col].head(profile.max_users).to_numpy()
    logger.info(
        "selected %d users out of %d eligible (checked %d)",
        chosen.size,
        eligible.size,
        len(stats),
    )
    return np.sort(chosen)


def ids_to_python(values: Iterable[int] | np.ndarray) -> set[int]:
    """Materialise ids into a plain Python set for fast membership tests."""
    return {int(value) for value in np.asarray(values).tolist()}


__all__ = [
    "accumulate_counts",
    "counts_to_series",
    "ids_to_python",
    "select_items",
    "select_users",
    "user_stats_frame",
]
