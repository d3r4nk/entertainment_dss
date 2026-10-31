"""Content-based recommender (section 7): cosine against the mean embedding of
the items the user liked in train."""

from __future__ import annotations

import numpy as np

from entdec.logging_setup import get_logger
from entdec.nlp.ollama_client import l2_normalize
from entdec.recsys.common import DomainData, ModelOutput

logger = get_logger(__name__)


def user_profile(data: DomainData, user: int) -> np.ndarray | None:
    """L2-normalised mean of the liked items' embeddings (None when empty)."""
    if data.embeddings is None:
        return None
    liked = data.likes.get(int(user), [])
    positions = [data.item_pos(raw) for raw in liked]
    positions = [p for p in positions if p is not None]
    if not positions:
        return None
    vectors = data.embeddings[positions]
    profile = vectors.mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm == 0.0:
        return None
    return profile / norm


def scores(data: DomainData, user: int) -> np.ndarray:
    profile = user_profile(data, user)
    if profile is None or data.embeddings is None:
        return np.zeros(len(data.catalog), dtype=np.float64)
    matrix = l2_normalize(data.embeddings.astype(np.float32))
    return np.asarray(matrix @ profile, dtype=np.float64)


def model_output(data: DomainData, user: int) -> ModelOutput:
    return ModelOutput(name="content", scores=scores(data, user))


__all__ = ["model_output", "scores", "user_profile"]
