"""KG-based scoring (section 7): user profile over Themes/Genres from liked
items, scored by weighted overlap against the item theme vectors."""

from __future__ import annotations

import numpy as np

from entdec.logging_setup import get_logger
from entdec.recsys.common import DomainData, ModelOutput

logger = get_logger(__name__)


def item_theme_vectors(data: DomainData) -> dict[int, dict[str, float]]:
    """Raw item id -> {theme: weight}."""
    out: dict[int, dict[str, float]] = {}
    for row in data.themes.itertuples(index=False):
        out.setdefault(int(row.item_id), {})[str(row.theme)] = float(row.weight)
    return out


def user_theme_profile(data: DomainData, user: int) -> dict[str, float]:
    """Theme -> normalised weight aggregated over the items the user liked."""
    vectors = item_theme_vectors(data)
    profile: dict[str, float] = {}
    for raw in data.likes.get(int(user), []):
        for theme, weight in vectors.get(int(raw), {}).items():
            profile[theme] = profile.get(theme, 0.0) + weight
    if profile:
        total = sum(profile.values())
        if total > 0:
            profile = {theme: weight / total for theme, weight in profile.items()}
    return profile


def user_genre_profile(data: DomainData, user: int) -> dict[str, float]:
    """Genre -> count/sum aggregated over liked items (used by the hybrid KG score)."""
    frame = data.genres[data.genres["item_id"].isin(list(data.likes.get(int(user), [])))]
    if frame.empty:
        return {}
    counts = frame["genre"].value_counts().astype(float)
    total = counts.sum()
    return {str(genre): count / total for genre, count in counts.items()}


def kg_scores(data: DomainData, user: int) -> np.ndarray:
    """Weighted-overlap score between the user profile and each item's themes."""
    item_vectors = item_theme_vectors(data)
    profile = user_theme_profile(data, user)
    out = np.zeros(len(data.catalog), dtype=np.float64)
    if not profile:
        return out
    for pos, raw in enumerate(data.catalog):
        vector = item_vectors.get(int(raw))
        if not vector:
            continue
        out[pos] = sum(
            weight * profile[theme] for theme, weight in vector.items() if theme in profile
        )
    return out


def model_output(data: DomainData, user: int) -> ModelOutput:
    return ModelOutput(name="kg", scores=kg_scores(data, user))


__all__ = ["item_theme_vectors", "kg_scores", "model_output", "user_genre_profile", "user_theme_profile"]
