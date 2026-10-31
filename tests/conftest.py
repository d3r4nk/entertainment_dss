"""Shared pytest fixtures.

All unit tests run against tiny synthetic fixtures: no network, no GPU, no
Docker. Real-data integration tests are marked ``integration``.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from entdec.config import (
    DEFAULT_SETTINGS_PATH,
    Config,
    EnvSettings,
    ProfileConfig,
    build_paths,
    load_yaml,
)

MovieRow = tuple[int, str, str]
RatingRow = tuple[int, int, float, int]
GameRow = tuple[Any, ...]
RecRow = tuple[Any, ...]


# --------------------------------------------------------------------------- #
# generic helpers
# --------------------------------------------------------------------------- #
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (in place) and return it."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def write_csv(path: Path, header: Sequence[str], rows: Iterable[Sequence[Any]]) -> Path:
    """Write a CSV file, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    """Write newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def write_parquet(path: Path, frame: Any) -> Path:
    """Write a pandas DataFrame to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# config fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., Config]:
    """Factory building a :class:`Config` whose paths point into ``tmp_path``."""

    def _make(
        profile: str = "tiny",
        overrides: dict[str, Any] | None = None,
        name: str = "settings.yaml",
    ) -> Config:
        data = load_yaml(DEFAULT_SETTINGS_PATH)
        if overrides:
            deep_merge(data, overrides)
        settings_path = tmp_path / name
        settings_path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

        data_dir = tmp_path / "data"
        env = EnvSettings(
            entdec_data_dir=data_dir,
            entdec_processed_dir=data_dir / "processed",
            entdec_reports_dir=tmp_path / "reports",
            entdec_cache_dir=tmp_path / ".cache",
        )
        profile_cfg = ProfileConfig(name=profile, **data["profiles"][profile])
        cfg = Config(
            env=env,
            yaml_data=data,
            profile=profile_cfg,
            paths=build_paths(env, profile),
            settings_path=settings_path,
        )
        cfg.paths.ensure()
        (data_dir / "movie_data").mkdir(parents=True, exist_ok=True)
        (data_dir / "game_data").mkdir(parents=True, exist_ok=True)
        return cfg

    return _make


# --------------------------------------------------------------------------- #
# synthetic raw datasets
# --------------------------------------------------------------------------- #
def _synthetic_movies() -> tuple[list[MovieRow], list[RatingRow]]:
    movies: list[MovieRow] = [
        (1, "Toy Story (1995)", "Adventure|Animation|Children|Comedy|Fantasy"),
        (2, "Jumanji (1995)", "Adventure|Children|Fantasy"),
        (3, "Grumpier Old Men (1995)", "Comedy|Romance"),
        (4, "Waiting to Exhale (1995)", "Comedy|Drama|Romance"),
        (5, "Father of the Bride Part II (1995)", "Comedy"),
        (6, "Heat (1995)", "Action|Crime|Thriller"),
        (7, "Sabrina (1995)", "Comedy|Romance"),
        (8, "Tom and Huck (1995)", "Adventure|Children"),
        (9, "Sudden Death (1995)", "Action"),
        (10, "GoldenEye (1995)", "Action|Adventure|Thriller"),
    ]
    ratings: list[RatingRow] = []
    ts = 1_000_000_000
    for user in range(1, 9):
        for offset, movie in enumerate(range(1, 11)):
            score = 5.0 if (user + movie) % 3 else 2.0
            ratings.append((user, movie, score, ts + offset * 1000 + user))
    return movies, ratings


def _synthetic_games() -> tuple[list[GameRow], list[RecRow], list[dict[str, Any]]]:
    titles = [
        "Portal 2",
        "Half-Life 2",
        "Stardew Valley",
        "Hades",
        "The Witcher 3: Wild Hunt",
        "Cyberpunk 2077",
        "Terraria",
        "Celeste",
        "Doom Eternal",
        "Civilization VI",
    ]
    tagsets = [
        ["Puzzle", "Sci-fi", "Story Rich", "Co-op"],
        ["Action", "Sci-fi", "Story Rich"],
        ["Farming Sim", "Relaxing", "Pixel Graphics", "Indie"],
        ["Roguelike", "Action", "Indie", "Great Soundtrack"],
        ["Open World", "RPG", "Fantasy", "Story Rich"],
        ["Open World", "RPG", "Cyberpunk", "Action"],
        ["Sandbox", "Indie", "Pixel Graphics", "Survival"],
        ["Platformer", "Indie", "Difficult", "Great Soundtrack"],
        ["FPS", "Action", "Gore"],
        ["Strategy", "Turn-Based", "Historical"],
    ]
    games: list[GameRow] = []
    metadata: list[dict[str, Any]] = []
    for index, title in enumerate(titles):
        app_id = 1000 + index
        games.append(
            (
                app_id,
                title,
                f"20{10 + index:02d}-03-15",
                "true",
                "false",
                "false",
                "Very Positive",
                90 - index,
                5000 - index * 100,
                19.99,
                29.99,
                0.0,
                "true",
            )
        )
        metadata.append(
            {
                "app_id": app_id,
                "description": f"{title} is a video game about {tagsets[index][0].lower()}.",
                "tags": tagsets[index],
            }
        )
    recs: list[RecRow] = []
    ts = 1_500_000_000
    for user in range(1, 9):
        for index in range(10):
            app_id = 1000 + index
            recs.append(
                (
                    app_id,
                    0,
                    0,
                    "2021-05-05",
                    "true" if (user + index) % 3 else "false",
                    float(10 + index),
                    user,
                    user * 100 + index,
                )
            )
            ts += 1
    return games, recs, metadata


@pytest.fixture
def raw_datasets(tmp_path: Path) -> Path:
    """Create a tiny raw dataset tree mirroring the real folder layout."""
    movie_dir = tmp_path / "data" / "movie_data"
    game_dir = tmp_path / "data" / "game_data"

    movies, ratings = _synthetic_movies()
    write_csv(movie_dir / "movies.csv", ["movieId", "title", "genres"], movies)
    write_csv(movie_dir / "ratings.csv", ["userId", "movieId", "rating", "timestamp"], ratings)
    write_csv(
        movie_dir / "links.csv",
        ["movieId", "imdbId", "tmdbId"],
        [(mid, 100 + mid, 200 + mid) for mid, _, _ in movies],
    )
    write_csv(
        movie_dir / "tags.csv",
        ["userId", "movieId", "tag", "timestamp"],
        [
            (1, 1, "pixar", 1_000_000_000),
            (2, 1, "pixar", 1_000_000_100),
            (3, 6, "heist", 1_000_000_200),
        ],
    )
    write_csv(
        movie_dir / "genome-tags.csv",
        ["tagId", "tag"],
        [(1, "toys"), (2, "friendship"), (3, "heist"), (4, "violence"), (5, "romance")],
    )
    genome_rows = []
    for mid, _, _ in movies:
        for tag_id in range(1, 6):
            genome_rows.append((mid, tag_id, round(0.9 - 0.1 * tag_id, 3)))
    write_csv(movie_dir / "genome-scores.csv", ["movieId", "tagId", "relevance"], genome_rows)

    games, recs, metadata = _synthetic_games()
    write_csv(
        game_dir / "games.csv",
        [
            "app_id",
            "title",
            "date_release",
            "win",
            "mac",
            "linux",
            "rating",
            "positive_ratio",
            "user_reviews",
            "price_final",
            "price_original",
            "discount",
            "steam_deck",
        ],
        games,
    )
    write_jsonl(game_dir / "games_metadata.json", metadata)
    write_csv(
        game_dir / "users.csv",
        ["user_id", "products", "reviews"],
        [(user, 10, 10) for user in range(1, 9)],
    )
    write_csv(
        game_dir / "recommendations.csv",
        ["app_id", "helpful", "funny", "date", "is_recommended", "hours", "user_id", "review_id"],
        recs,
    )
    return tmp_path / "data"


@pytest.fixture
def synthetic_config(make_config: Callable[..., Config], raw_datasets: Path) -> Config:
    """A ``tiny`` profile wired to the synthetic raw tree.

    Thresholds are lowered so the 10-item / 8-user fixtures survive sampling.
    """
    overrides = {
        "profiles": {
            "tiny": {
                "max_items": 6,
                "max_users": 5,
                "min_movie_ratings": 2,
                "min_game_reviews": 2,
                "min_interactions_per_user": 2,
                "min_positive_per_user": 3,
            }
        }
    }
    return make_config("tiny", overrides=overrides)
