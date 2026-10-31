"""Hybrid recommender (section 7): weighted sum of min-max normalised scores.

Weights are tuned by grid search on the validation split
(:func:`tune_weights`) and persisted in ``config/settings.yaml`` under
``hybrid_weights.<profile>`` so the loaded settings are used at inference time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from entdec.logging_setup import get_logger
from entdec.recsys import content, kg_scoring, popularity
from entdec.recsys import mf as mf_mod
from entdec.recsys.common import DomainData, ModelOutput, exclude_seen, top_k_positions
from entdec.recsys.itemknn import ItemKNNModel

logger = get_logger(__name__)

MODEL_NAMES = ("popularity", "itemknn", "mf", "content", "kg")


@dataclass
class BaseScores:
    """All base models' raw scores for one user."""

    user: int
    parts: dict[str, np.ndarray]


def min_max_normalise(vector: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant vector maps to zeros."""
    finite = np.isfinite(vector)
    if not finite.any():
        return np.zeros_like(vector, dtype=np.float64)
    lo = vector[finite].min()
    hi = vector[finite].max()
    if hi <= lo:
        return np.zeros_like(vector, dtype=np.float64)
    out = np.zeros_like(vector, dtype=np.float64)
    out[finite] = (vector[finite] - lo) / (hi - lo)
    return out


def raw_scores(data: DomainData, user: int, fitted: Mapping[str, object]) -> BaseScores:
    """Compute every base model's raw scores for one user."""
    parts: dict[str, np.ndarray] = {}
    parts["popularity"] = popularity.popularity_scores(data)
    knn = fitted.get("itemknn")
    if isinstance(knn, ItemKNNModel):
        parts["itemknn"] = knn.scores(user)
    else:
        parts["itemknn"] = np.zeros(len(data.catalog))
    mf_model = fitted.get("mf")
    if isinstance(mf_model, mf_mod.MFModel):
        parts["mf"] = mf_model.scores(user)
    else:
        parts["mf"] = np.zeros(len(data.catalog))
    parts["content"] = content.scores(data, user)
    parts["kg"] = kg_scoring.kg_scores(data, user)
    return BaseScores(user=user, parts=parts)


def normalised_scores(
    data: DomainData, user: int, fitted: Mapping[str, object]
) -> dict[str, np.ndarray]:
    """Raw scores -> min-max normalised [0, 1] parts."""
    base = raw_scores(data, user, fitted)
    return {name: min_max_normalise(vector) for name, vector in base.parts.items()}


def blend(parts: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Weighted sum of already-normalised parts; missing weights are 0."""
    names = next(iter(parts.values())).shape[0] if parts else 0
    out = np.zeros(names, dtype=np.float64)
    for name, vector in parts.items():
        out += float(weights.get(name, 0.0)) * vector
    return out


class HybridModel:
    """Scores a user with the configured weights (validation-tuned)."""

    def __init__(
        self,
        data: DomainData,
        weights: Mapping[str, float],
        *,
        fitted: Mapping[str, object] | None = None,
    ) -> None:
        self.data = data
        self.weights = {name: float(weights.get(name, 0.0)) for name in MODEL_NAMES}
        self.fitted = dict(fitted or {})

    def scores(self, user: int) -> np.ndarray:
        parts = normalised_scores(self.data, user, self.fitted)
        return blend(parts, self.weights)

    def recommend(self, user: int, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
        scores = exclude_seen(self.scores(user), self.data, user)
        positions = top_k_positions(scores, k)
        return np.array(positions), scores[positions]

    def weights_array(self) -> np.ndarray:
        return np.array([self.weights[name] for name in MODEL_NAMES], dtype=np.float64)


# --------------------------------------------------------------------------- #
# weight tuning on validation
# --------------------------------------------------------------------------- #
def _grid(weights_cfg: Mapping[str, float]) -> Iterable[dict[str, float]]:
    """Coarse deterministic simplex over the five model weights (sums to ~1)."""
    for popularity_weight in (0.0, 0.05, 0.1):
        for knn in (0.0, 0.1, 0.2):
            rest1 = round(1.0 - popularity_weight - knn, 2)
            if rest1 < 0:
                continue
            for mf_weight in (0.0, rest1 / 4, rest1 / 2, rest1):
                rest2 = round(rest1 - mf_weight, 2)
                if rest2 < 0:
                    continue
                for content_frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                    content_weight = round(rest2 * content_frac, 2)
                    kg_weight = round(rest2 - content_weight, 2)
                    if kg_weight < 0:
                        continue
                    candidate = {
                        "popularity": popularity_weight,
                        "itemknn": knn,
                        "mf": round(mf_weight, 2),
                        "content": content_weight,
                        "kg": kg_weight,
                    }
                    if 0.98 <= sum(candidate.values()) <= 1.02:
                        yield candidate


@dataclass
class TuneResult:
    """Best weights plus the validation score they achieved."""

    weights: dict[str, float]
    ndcg: float
    n_candidates: int


def tune_weights(
    data: DomainData,
    evaluate_fn: Callable[[DomainData, Mapping[str, float]], float],
    *,
    max_candidates: int = 400,
) -> TuneResult:
    """Grid-search hybrid weights; ``evaluate_fn`` returns a validation metric."""
    best: tuple[float, dict[str, float]] | None = None
    candidates = 0
    for weights in _grid({}):
        if candidates >= max_candidates:
            break
        score = float(evaluate_fn(data, weights))
        if best is None or score > best[0]:
            best = (score, dict(weights))
        candidates += 1
    if best is None:
        best = (0.0, {"popularity": 1.0})
    logger.info("tuned hybrid weights: %s (ndcg@10=%.4f)", best[1], best[0])
    return TuneResult(weights=best[1], ndcg=best[0], n_candidates=candidates)


def model_output(model: HybridModel, user: int) -> ModelOutput:
    return ModelOutput(name="hybrid", scores=model.scores(user))


__all__ = [
    "MODEL_NAMES",
    "BaseScores",
    "HybridModel",
    "TuneResult",
    "blend",
    "min_max_normalise",
    "model_output",
    "normalised_scores",
    "raw_scores",
    "tune_weights",
]
