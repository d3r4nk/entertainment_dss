"""Metadata completion for games whose Steam entry is empty.

``games_metadata.json`` has no tags and no description for a large share of the
most popular titles (Portal 2, Dota 2, Hades, Stardew Valley, ...): 1,244 of the
50,872 entries are empty overall, but ~63% of the top 500 games by review count
are affected. Without tags those items would have no ``:Theme`` edges and no
usable text, which breaks the theme/KG layer and the chatbot demo.

The fix uses **only the local dataset**: items are described by *who reviewed
them*. The user-game matrix of all reviews for the sampled games is turned into
an item-item cosine similarity, and the tags of the most similar *tagged*
neighbours are transferred to the untagged item, weighted by similarity.

Every row produced here is marked ``source = "propagated"`` so the origin of a
tag is always auditable, and Steam tags are never overwritten.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from entdec.config import Config
from entdec.data.io import iter_parquet_batches
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

TAG_COLUMNS = ("app_id", "tag", "rank", "weight", "source")


def cooccurrence_similarity(
    batches: Iterable[pd.DataFrame],
    item_ids: np.ndarray,
    *,
    item_col: str = "app_id",
    uid_col: str = "uid",
    flag_col: str | None = "is_recommended",
    min_items_per_user: int = 2,
    max_items_per_user: int = 100,
    min_support: int = 10,
    min_lift: float = 1.0,
) -> sparse.csr_matrix:
    """Item-item association score derived from co-review behaviour.

    Popular items are co-reviewed with everything, so raw co-occurrence counts
    are useless here. The score is the cosine of the binary user-item vectors
    (popularity normalised), **gated by lift**:

    ``lift(i, j) = co(i, j) * n_users / (n_users_i * n_users_j)``

    A pair only survives when ``lift >= min_lift`` (the two items attract an
    audience that is *over*-represented, not just a big one) and the pair is
    supported by at least ``min_support`` users.

    Args:
        batches: Stream of interaction frames (columns: uid, item, optional flag).
        item_ids: Catalog ids; the matrix follows this order.
        item_col: Name of the item id column.
        uid_col: Name of the user id column.
        flag_col: Optional boolean column; when given only ``True`` rows count.
        min_items_per_user: Users with a single item carry no pair information.
        max_items_per_user: Users above this are dropped; they review almost
            everything and would make every pair look similar.
        min_support: Minimum number of shared users for a pair to count.
        min_lift: Minimum lift for a pair to count.

    Returns:
        ``(n_items, n_items)`` float32 CSR matrix of scores with a zeroed
        diagonal.
    """
    item_position = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=np.asarray(item_ids))
    user_parts: list[np.ndarray] = []
    item_parts: list[np.ndarray] = []

    for batch in batches:
        frame = batch
        if flag_col is not None and flag_col in frame.columns:
            frame = frame[frame[flag_col].to_numpy(dtype=bool)]
        if frame.empty:
            continue
        codes = item_position.reindex(frame[item_col].to_numpy()).to_numpy()
        mask = ~pd.isna(codes)
        if not mask.any():
            continue
        # Keep the raw user ids: factorizing per batch would merge distinct
        # users that happen to share a small per-batch index.
        user_parts.append(frame[uid_col].to_numpy()[mask].astype(np.int64))
        item_parts.append(codes[mask].astype(np.int32))

    empty = sparse.csr_matrix((len(item_ids), len(item_ids)), dtype=np.float32)
    if not item_parts:
        logger.warning("no interactions available for similarity")
        return empty

    users = pd.factorize(np.concatenate(user_parts))[0]
    items = np.concatenate(item_parts)

    matrix = sparse.csr_matrix(
        (np.ones(len(users), dtype=np.float32), (users, items)),
        shape=(int(users.max()) + 1, len(item_ids)),
    )
    counts_per_user = np.asarray(matrix.sum(axis=1)).ravel()
    keep = np.flatnonzero(
        (counts_per_user >= min_items_per_user) & (counts_per_user <= max_items_per_user)
    )
    matrix = matrix[keep]
    n_users = matrix.shape[0]
    if n_users == 0:
        logger.warning("no usable users after the per-user item filter")
        return empty

    co = (matrix.T @ matrix).tocsr().astype(np.float64)
    # The diagonal of X^T X is the number of users holding the item; a row sum
    # would count co-reviews instead.
    support = np.asarray(co.diagonal()).ravel()
    indptr, indices, data = co.indptr, co.indices, co.data
    # Row index of every stored element, needed for the pairwise denominators.
    rows = np.repeat(np.arange(co.shape[0], dtype=np.int64), np.diff(indptr))
    denominators = support[rows] * support[indices]
    valid = denominators > 0

    cosine = np.zeros_like(data)
    cosine[valid] = data[valid] / np.sqrt(denominators[valid])
    lift = np.zeros_like(data)
    lift[valid] = data[valid] * n_users / denominators[valid]

    score = np.where((lift >= min_lift) & (data >= min_support), cosine, 0.0)
    sparse_score = sparse.csr_matrix((score, indices, indptr), shape=co.shape).astype(np.float32)
    sparse_score.setdiag(0.0)
    sparse_score.eliminate_zeros()
    logger.info(
        "co-review similarity: %d users, %d candidate pairs (density %.3f%%, min_lift=%.2f)",
        n_users,
        sparse_score.nnz,
        100.0 * sparse_score.nnz / max(co.shape[0] ** 2, 1),
        min_lift,
    )
    return sparse_score


def propagate_tags(
    tagged: pd.DataFrame,
    similarity: sparse.csr_matrix,
    item_ids: np.ndarray,
    *,
    top_neighbours: int = 8,
    top_tags: int = 10,
    min_similarity: float = 0.05,
) -> pd.DataFrame:
    """Transfer tags from the most similar tagged items to untagged items.

    Args:
        tagged: Tag table with ``app_id``, ``tag``, ``rank``, ``weight``.
        similarity: Output of :func:`cooccurrence_similarity`.
        item_ids: Catalog order used to build ``similarity``.
        top_neighbours: How many neighbours contribute.
        top_tags: How many tags to keep per completed item.
        min_similarity: Ignore neighbours below this cosine similarity.

    Returns:
        Rows with :data:`TAG_COLUMNS` for items that had no tags at all.
    """
    empty = pd.DataFrame(columns=list(TAG_COLUMNS))
    if tagged.empty or similarity.shape[0] == 0:
        return empty

    position = pd.Series(np.arange(len(item_ids), dtype=np.int32), index=np.asarray(item_ids))
    tagged = tagged.copy()
    tagged["position"] = tagged["app_id"].map(position)
    tagged = tagged[tagged["position"].notna()]
    if tagged.empty:
        return empty

    tagged_positions = np.unique(tagged["position"].to_numpy(dtype=np.int32))
    tagged_mask = np.zeros(similarity.shape[0], dtype=bool)
    tagged_mask[tagged_positions] = True
    untagged_positions = np.flatnonzero(~tagged_mask)
    if untagged_positions.size == 0:
        return empty

    # tag -> position -> weight lookup for fast aggregation
    by_position = {
        int(pos): grp[["tag", "weight"]].to_numpy()
        for pos, grp in tagged.groupby("position", sort=False)
    }
    weights_by_position = {
        int(pos): dict(zip(values[:, 0].tolist(), values[:, 1].astype(float).tolist(), strict=False))
        for pos, values in by_position.items()
    }

    rows: list[dict[str, object]] = []
    for pos in untagged_positions:
        row = similarity.getrow(int(pos))
        if row.nnz == 0:
            continue
        neighbour_positions = row.indices
        neighbour_sims = row.data
        # Keep only tagged neighbours above the threshold.
        keep = tagged_mask[neighbour_positions] & (neighbour_sims >= min_similarity)
        if not keep.any():
            continue
        neighbour_positions = neighbour_positions[keep]
        neighbour_sims = neighbour_sims[keep]
        order = np.argsort(-neighbour_sims)[:top_neighbours]
        neighbour_positions = neighbour_positions[order]
        neighbour_sims = neighbour_sims[order]
        total = float(neighbour_sims.sum())
        if total <= 0:
            continue

        aggregated: dict[str, float] = {}
        for neighbour_pos, weight in zip(neighbour_positions, neighbour_sims, strict=False):
            for tag, tag_weight in weights_by_position.get(int(neighbour_pos), {}).items():
                aggregated[tag] = aggregated.get(tag, 0.0) + float(weight) / total * float(tag_weight)
        if not aggregated:
            continue
        best = sorted(aggregated.items(), key=lambda item: (-item[1], item[0]))[:top_tags]
        top_weight = best[0][1] or 1.0
        for rank, (tag, weight) in enumerate(best, start=1):
            rows.append(
                {
                    "app_id": int(item_ids[pos]),
                    "tag": tag,
                    "rank": rank,
                    "weight": round(float(weight) / top_weight, 5),
                    "source": "propagated",
                }
            )

    frame = pd.DataFrame(rows, columns=list(TAG_COLUMNS)) if rows else empty
    logger.info(
        "propagated tags for %d of %d untagged items",
        frame["app_id"].nunique() if not frame.empty else 0,
        untagged_positions.size,
    )
    return frame


def complete_game_tags(
    cfg: Config,
    temp_path: Path,
    item_ids: np.ndarray,
    tag_table: pd.DataFrame,
) -> pd.DataFrame:
    """Fill missing game tags from co-review neighbours (see module docstring).

    Returns:
        The tag table with propagated rows appended, or the input unchanged when
        completion is disabled or impossible.
    """
    if not bool(cfg.get("data_completion.game_tag_propagation", True)):
        logger.info("game tag propagation disabled in settings")
        return tag_table

    tagged = tag_table.copy()
    if "source" not in tagged.columns:
        tagged["source"] = "steam"
    missing_before = set(np.asarray(item_ids).tolist()) - set(tagged["app_id"].tolist())
    if not missing_before:
        logger.info("every sampled game already has Steam tags - nothing to propagate")
        return tagged

    logger.info("%d of %d sampled games have no Steam tags", len(missing_before), len(item_ids))
    similarity = cooccurrence_similarity(
        iter_parquet_batches(temp_path, ["uid", "app_id", "is_recommended"]),
        item_ids,
        max_items_per_user=int(cfg.get("data_completion.max_items_per_user", 100)),
        min_support=int(cfg.get("data_completion.min_support", 10)),
        min_lift=float(cfg.get("data_completion.min_lift", 1.0)),
    )
    propagated = propagate_tags(
        tagged,
        similarity,
        item_ids,
        top_neighbours=int(cfg.get("data_completion.top_neighbours", 8)),
        top_tags=int(cfg.get("taxonomy.game_top_themes", 10)),
        min_similarity=float(cfg.get("data_completion.min_similarity", 0.05)),
    )
    combined = pd.concat([tagged, propagated], ignore_index=True)
    combined["rank"] = combined["rank"].astype("int16")
    combined["weight"] = combined["weight"].astype("float32")
    combined = combined.sort_values(["app_id", "rank"], ignore_index=True)
    missing_after = set(np.asarray(item_ids).tolist()) - set(combined["app_id"].tolist())
    logger.info(
        "games still without tags after propagation: %d of %d", len(missing_after), len(item_ids)
    )
    return combined


__all__ = [
    "TAG_COLUMNS",
    "complete_game_tags",
    "cooccurrence_similarity",
    "propagate_tags",
]
