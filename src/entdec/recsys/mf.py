"""Matrix factorization (section 7): implicit ALS when importable, else
sklearn TruncatedSVD on the binary user x item matrix."""

from __future__ import annotations

import numpy as np

from entdec.logging_setup import get_logger
from entdec.recsys.common import DomainData, ModelOutput

logger = get_logger(__name__)

ALS_FACTORS = 64
ALS_REGULARISATION = 0.05
ALS_ITERATIONS = 15
SVD_FACTORS = 64


class MFModel:
    """Latent-factor model with a ``user_factors x item_factors.T`` scorer."""

    def __init__(self, data: DomainData, factors: int = ALS_FACTORS) -> None:
        self.data = data
        self.method = "als"
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self._user_pos = {int(raw): pos for pos, raw in enumerate(data.user_ids)}
        self.factors = factors
        try:
            import implicit  # type: ignore

            self._fit_als(implicit)
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            logger.info("implicit ALS unavailable (%s) - using TruncatedSVD", type(exc).__name__)
            self.method = "svd"
            self._fit_svd()

    def _fit_als(self, implicit: object) -> None:
        matrix = self.data.matrix.tocsr()
        als = implicit.als.AlternatingLeastSquares(  # type: ignore[attr-defined]
            factors=self.factors,
            regularization=ALS_REGULARISATION,
            iterations=ALS_ITERATIONS,
            random_state=42,
        )
        als.fit(matrix)  # implicit expects users as rows, items as columns
        self.item_factors = np.asarray(als.item_factors, dtype=np.float32)
        self.user_factors = np.asarray(als.user_factors, dtype=np.float32)

    def _fit_svd(self) -> None:
        from sklearn.decomposition import TruncatedSVD

        matrix = self.data.matrix.tocsr()
        factors = min(SVD_FACTORS, min(matrix.shape) - 1)
        svd = TruncatedSVD(n_components=factors, random_state=42)
        self.item_factors = svd.components_.T.astype(np.float32)  # items x k
        user_latent = matrix @ svd.components_.T  # users x k
        self.user_factors = user_latent.astype(np.float32)

    def scores(self, user: int) -> np.ndarray:
        pos = self._user_pos.get(int(user))
        if pos is None or self.user_factors is None or self.item_factors is None:
            return np.zeros(len(self.data.catalog), dtype=np.float64)
        user_vec = self.user_factors[pos]
        return np.asarray(self.item_factors @ user_vec, dtype=np.float64)


def fit(data: DomainData, factors: int = ALS_FACTORS) -> MFModel:
    """Train one MF model over a domain."""
    return MFModel(data, factors=factors)


def model_output(model: MFModel, user: int) -> ModelOutput:
    return ModelOutput(name="mf", scores=model.scores(user))


__all__ = ["ALS_FACTORS", "ALS_ITERATIONS", "ALS_REGULARISATION", "MFModel", "fit", "model_output"]
