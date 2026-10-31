"""Canonical genre taxonomy (PROJECT_SPEC.md section 6).

A single hand-written map from the raw MovieLens genre strings and Steam tags
to a shared genre set, so movies and games become comparable on a coarse axis.

The convention: every key is a lower-cased, whitespace-normalised raw genre and
every value is a canonical genre from :data:`CANONICAL_GENRES`.
"""

from __future__ import annotations

import re

CANONICAL_GENRES: tuple[str, ...] = (
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Horror",
    "Indie",
    "Massively Multiplayer",
    "Music",
    "Mystery",
    "Platformer",
    "Puzzle",
    "Racing",
    "RPG",
    "Romance",
    "Sci-Fi",
    "Simulation",
    "Sports",
    "Strategy",
    "Thriller",
    "War",
    "Western",
)

# raw genre (lower case) -> canonical genre
GENRE_MAP: dict[str, str] = {
    # --- MovieLens genres -----------------------------------------------------
    "action": "Action",
    "adventure": "Adventure",
    "animation": "Animation",
    "children": "Family",
    "comedy": "Comedy",
    "crime": "Crime",
    "documentary": "Documentary",
    "drama": "Drama",
    "family": "Family",
    "fantasy": "Fantasy",
    "film-noir": "Crime",
    "film noir": "Crime",
    "horror": "Horror",
    "musical": "Music",
    "music": "Music",
    "mystery": "Mystery",
    "romance": "Romance",
    "sci-fi": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "thriller": "Thriller",
    "war": "War",
    "western": "Western",
    # "IMAX" and "(no genres listed)" are deliberately unmapped.
    # --- Steam tags -----------------------------------------------------------
    "action-adventure": "Adventure",
    "anime": "Animation",
    "atmospheric": "Adventure",
    "base building": "Strategy",
    "beat 'em up": "Action",
    "building": "Simulation",
    "bullet hell": "Action",
    "card game": "Strategy",
    "casual": "Indie",
    "city builder": "Strategy",
    "classic": "Adventure",
    "clicker": "Simulation",
    "co-op": "Adventure",
    "colony sim": "Strategy",
    "cute": "Family",
    "cyberpunk": "Sci-Fi",
    "dungeons & dragons": "RPG",
    "early access": "Indie",
    "education": "Documentary",
    "exploration": "Adventure",
    "family friendly": "Family",
    "fighting": "Action",
    "first-person": "Action",
    "free to play": "Indie",
    "funny": "Comedy",
    "gore": "Horror",
    "hack and slash": "Action",
    "historical": "War",
    "indie": "Indie",
    "jrpg": "RPG",
    "life sim": "Simulation",
    "local co-op": "Adventure",
    "local multiplayer": "Adventure",
    "mature": "Drama",
    "military": "War",
    "mmorpg": "Massively Multiplayer",
    "moba": "Strategy",
    "multiplayer": "Massively Multiplayer",
    "music-based procedural generation": "Music",
    "open world": "Adventure",
    "parkour": "Action",
    "party": "Family",
    "pixel graphics": "Indie",
    "platformer": "Platformer",
    "point & click": "Puzzle",
    "post-apocalyptic": "Sci-Fi",
    "psychological horror": "Horror",
    "puzzle": "Puzzle",
    "puzzle platformer": "Puzzle",
    "racing": "Racing",
    "realistic": "Simulation",
    "relaxing": "Family",
    "retro": "Indie",
    "roguelike": "RPG",
    "roguelite": "RPG",
    "rpg": "RPG",
    "sandbox": "Simulation",
    "sexual content": "Romance",
    "shooter": "Action",
    "simulation": "Simulation",
    "singleplayer": "Adventure",
    "souls-like": "RPG",
    "space": "Sci-Fi",
    "sports": "Sports",
    "stealth": "Action",
    "strategy": "Strategy",
    "survival": "Adventure",
    "survival horror": "Horror",
    "tactical": "Strategy",
    "third person": "Action",
    "tower defense": "Strategy",
    "turn-based": "Strategy",
    "turn-based strategy": "Strategy",
    "visual novel": "Adventure",
    "vr": "Adventure",
    "violent": "Action",
    "world war ii": "War",
    "zombies": "Horror",
}

_NON_ALNUM = re.compile(r"[^a-z0-9\-'&+ ]+")
_WHITESPACE = re.compile(r"\s+")


def normalise_genre(raw: str) -> str:
    """Normalise a raw genre/tag for lookup in :data:`GENRE_MAP`."""
    if not raw:
        return ""
    text = str(raw).strip().lower().replace("_", " ")
    text = text.replace("&", "&").replace("’", "'")
    text = _NON_ALNUM.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def map_raw_genre(raw: str) -> str | None:
    """Map one raw genre to its canonical name, or ``None`` when unknown."""
    return GENRE_MAP.get(normalise_genre(raw))


def map_genres(raw_genres: list[str], limit: int | None = None) -> list[str]:
    """Map a list of raw genres, preserving order and dropping duplicates."""
    mapped: list[str] = []
    for raw in raw_genres or []:
        canonical = map_raw_genre(raw)
        if canonical and canonical not in mapped:
            mapped.append(canonical)
    if limit is not None:
        mapped = mapped[:limit]
    return mapped


def unmapped_genres(raw_genres: list[str]) -> list[str]:
    """Return the raw genres that :data:`GENRE_MAP` does not cover (for auditing)."""
    return [raw for raw in raw_genres or [] if map_raw_genre(raw) is None]


__all__ = [
    "CANONICAL_GENRES",
    "GENRE_MAP",
    "map_genres",
    "map_raw_genre",
    "normalise_genre",
    "unmapped_genres",
]
