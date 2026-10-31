"""Namespaced node ids.

MovieLens and Steam ids live in different ranges, so every node id is prefixed:
``ml_`` for the movie domain and ``st_`` for the game domain, as specified in
section 5 (``ml_<id>`` / ``st_<id>``).
"""

from __future__ import annotations

import re

MOVIE_PREFIX = "ml_"
GAME_PREFIX = "st_"
USER_PREFIX = {"movie": "ml_", "game": "st_"}
GENRE_PREFIX = "gn_"
THEME_PREFIX = "th_"
PLATFORM_PREFIX = "pf_"

_SLUG = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lower-case slug used inside genre/theme/platform ids."""
    return _SLUG.sub("_", str(text).strip().lower()).strip("_")


def movie_node_id(movie_id: int) -> str:
    """``1`` -> ``ml_1``."""
    return f"{MOVIE_PREFIX}{int(movie_id)}"


def game_node_id(app_id: int) -> str:
    """``730`` -> ``st_730``."""
    return f"{GAME_PREFIX}{int(app_id)}"


def user_node_id(domain: str, uid: int) -> str:
    """``("movie", 7)`` -> ``ml_7``; ``("game", 7)`` -> ``st_7``."""
    prefix = USER_PREFIX.get(domain)
    if prefix is None:
        raise ValueError(f"unknown domain {domain!r}")
    return f"{prefix}{int(uid)}"


def genre_node_id(name: str) -> str:
    """``"Sci-Fi"`` -> ``gn_sci_fi``."""
    return f"{GENRE_PREFIX}{slugify(name)}"


def theme_node_id(name: str) -> str:
    """``"great soundtrack"`` -> ``th_great_soundtrack``."""
    return f"{THEME_PREFIX}{slugify(name)}"


def platform_node_id(name: str) -> str:
    """``"Windows"`` -> ``pf_windows``."""
    return f"{PLATFORM_PREFIX}{slugify(name)}"


def item_node_id(domain: str, item_id: int) -> str:
    """Dispatch to the movie/game id helper."""
    return movie_node_id(item_id) if domain == "movie" else game_node_id(item_id)


def parse_item_id(node_id: str) -> tuple[str, int]:
    """Inverse of :func:`item_node_id`."""
    if node_id.startswith(MOVIE_PREFIX):
        return "movie", int(node_id[len(MOVIE_PREFIX) :])
    if node_id.startswith(GAME_PREFIX):
        return "game", int(node_id[len(GAME_PREFIX) :])
    raise ValueError(f"not an item node id: {node_id!r}")


__all__ = [
    "GAME_PREFIX",
    "GENRE_PREFIX",
    "MOVIE_PREFIX",
    "PLATFORM_PREFIX",
    "THEME_PREFIX",
    "game_node_id",
    "genre_node_id",
    "item_node_id",
    "movie_node_id",
    "parse_item_id",
    "platform_node_id",
    "slugify",
    "theme_node_id",
    "user_node_id",
]
