"""Popularity recommender (section 7): most positive train interactions."""

from __future__ import annotations

import numpy as np

from entdec.logging_setup import get_logger
from entdec.recsys.common import DomainData, ModelOutput

logger = get_logger(__name__)


def popularity_scores(data: DomainData) -> np.ndarray:
    """Positive-interaction counts over the catalogue (train split only)."""

    matrix = data.matrix.tocsr()
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    return counts.astype(np.float64)


def recommend(data: DomainData, user: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, scores) of the top-k popular items for one user."""
    scores = popularity_scores(data)
    from entdec.recsys.common import exclude_seen, top_k_positions

    scores = exclude_seen(scores, data, user)
    positions = top_k_positions(scores, k)
    return np.array(positions), scores[positions]


def model_output(data: DomainData, user: int) -> ModelOutput:
    """Full-catalogue scores for the evaluation loop."""
    return ModelOutput(name="popularity", scores=popularity_scores(data))


def popularity_per_item(data: DomainData) -> dict[int, int]:
    """Raw item id -> positive count (used for cold-start fallbacks)."""
    scores = popularity_scores(data)
    return {raw: int(scores[pos]) for raw, pos in data.catalog_index.items()}


__all__ = ["model_output", "popularity_per_item", "popularity_scores", "recommend"]
