"""ItemKNN (section 7): cosine similarity on the sparse item x user matrix.

``score(user, item) = max over liked items j of cos(item, j)`` (a max is more
robust than a sum when users like large clusters of items).
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from entdec.logging_setup import get_logger
from entdec.recsys.common import DomainData, ModelOutput

logger = get_logger(__name__)


def item_similarity(matrix: sp.csr_matrix) -> sp.csr_matrix:
    """Cosine similarity between catalogue columns (items) of the user x item matrix."""
    item_user = matrix.T.tocsr()  # items x users
    norms = np.sqrt(np.asarray(item_user.multiply(item_user).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    normalised = item_user.multiply(1.0 / norms[:, None]).tocsr()
    similarity = (normalised @ normalised.T).tocsr()
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    return similarity


class ItemKNNModel:
    """Stateful wrapper so the similarity matrix is computed once."""

    def __init__(self, data: DomainData) -> None:
        self.data = data
        self.similarity = item_similarity(data.matrix)

    def scores(self, user: int) -> np.ndarray:
        data = self.data
        liked = data.likes.get(int(user), [])
        positions = [data.item_pos(raw) for raw in liked]
        positions = [p for p in positions if p is not None]
        if not positions:
            return np.zeros(len(data.catalog), dtype=np.float64)
        profile = np.zeros(len(data.catalog), dtype=np.float32)
        profile[positions] = 1.0
        return np.asarray(self.similarity @ profile).ravel().astype(np.float64)


def model_output(model: ItemKNNModel, user: int) -> ModelOutput:
    return ModelOutput(name="itemknn", scores=model.scores(user))


__all__ = ["ItemKNNModel", "item_similarity", "model_output"]
