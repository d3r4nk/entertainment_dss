"""Canonical taxonomy shared by both domains: genres and clustered themes."""

from __future__ import annotations

from entdec.taxonomy.genres import (
    CANONICAL_GENRES,
    GENRE_MAP,
    map_genres,
    map_raw_genre,
    normalise_genre,
)
from entdec.taxonomy.themes import (
    THEME_MAP_CSV,
    build_theme_map,
    load_theme_map,
    normalise_tag,
)

__all__ = [
    "CANONICAL_GENRES",
    "GENRE_MAP",
    "THEME_MAP_CSV",
    "build_theme_map",
    "load_theme_map",
    "map_genres",
    "map_raw_genre",
    "normalise_genre",
    "normalise_tag",
]
