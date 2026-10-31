"""Tests for :mod:`entdec.taxonomy`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entdec.config import Config
from entdec.data import movielens, steam
from entdec.taxonomy import genres as genre_mod
from entdec.taxonomy import pipeline as tax_pipeline
from entdec.taxonomy import themes as theme_mod


# --------------------------------------------------------------------------- #
# fake Ollama client
# --------------------------------------------------------------------------- #
class FakeEmbedder:
    """Deterministic embedder: tags of the same group share a direction."""

    def __init__(self, groups: dict[str, list[str]] | None = None) -> None:
        self.groups = groups or {}
        self.calls = 0

    def _vector(self, text: str) -> np.ndarray:
        for index, (_, members) in enumerate(sorted(self.groups.items())):
            if text in members:
                vector = np.zeros(8, dtype=np.float32)
                vector[index % 8] = 1.0
                vector[(index + 3) % 8] = 0.05
                return vector / np.linalg.norm(vector)
        # Unknown tags get a small deterministic direction.
        seed = abs(hash(text)) % 8
        vector = np.zeros(8, dtype=np.float32)
        vector[seed] = 1.0
        return vector

    def embed_batch(
        self, texts, batch_size: int = 24, dimensions: int | None = None, **_: object
    ) -> np.ndarray:
        self.calls += 1
        return np.vstack([self._vector(str(text)) for text in texts])


@pytest.fixture
def tagged_config(make_config, raw_datasets) -> Config:
    """Synthetic project with the data + a tiny clustering-friendly taxonomy config."""
    cfg = make_config(
        "tiny",
        overrides={
            "profiles": {
                "tiny": {
                    "max_items": 6,
                    "max_users": 5,
                    "min_movie_ratings": 2,
                    "min_game_reviews": 2,
                    "min_interactions_per_user": 2,
                    "min_positive_per_user": 3,
                }
            },
            "taxonomy": {"themes": 4, "min_tag_frequency": 1},
        },
    )
    movielens.run(cfg, force=True)
    steam.run(cfg, force=True)
    return cfg


# --------------------------------------------------------------------------- #
# genres
# --------------------------------------------------------------------------- #
def test_spec_genres_are_present() -> None:
    required = {
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Drama",
        "Fantasy",
        "Horror",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Family",
        "Indie",
        "RPG",
        "Strategy",
        "Simulation",
        "Sports",
        "Puzzle",
    }
    assert required <= set(genre_mod.CANONICAL_GENRES)


def test_genre_map_targets_are_canonical() -> None:
    unknown = {
        value for value in genre_mod.GENRE_MAP.values() if value not in genre_mod.CANONICAL_GENRES
    }
    assert not unknown, f"genre map points to unknown canonical genres: {sorted(unknown)}"


def test_normalise_genre() -> None:
    assert genre_mod.normalise_genre("  Sci-Fi ") == "sci-fi"
    assert genre_mod.normalise_genre("Film Noir") == "film noir"
    assert genre_mod.normalise_genre("Sci_Fi") == "sci fi"


def test_map_raw_genre_known_and_unknown() -> None:
    assert genre_mod.map_raw_genre("Children") == "Family"
    assert genre_mod.map_raw_genre("IMAX") is None
    assert genre_mod.map_raw_genre("(no genres listed)") is None


def test_map_genres_deduplicates_and_preserves_order() -> None:
    assert genre_mod.map_genres(["Drama", "Comedy", "Comedy", "IMAX"]) == ["Drama", "Comedy"]


def test_map_genres_limit() -> None:
    assert genre_mod.map_genres(["Drama", "Comedy", "Fantasy"], limit=2) == ["Drama", "Comedy"]


def test_unmapped_genres_is_an_audit() -> None:
    unmapped = genre_mod.unmapped_genres(["Drama", "IMAX", "Story Rich"])
    assert set(unmapped) == {"IMAX", "Story Rich"}


# --------------------------------------------------------------------------- #
# tag normalisation
# --------------------------------------------------------------------------- #
def test_normalise_tag_lowercases_and_strips_punctuation() -> None:
    assert theme_mod.normalise_tag("Sci-Fi & Fantasy!") == "sci-fi & fantasy"
    assert theme_mod.normalise_tag("  Multiple   Spaces  ") == "multiple spaces"


def test_normalise_tag_applies_aliases() -> None:
    assert theme_mod.normalise_tag("Co-op") == "co-op"
    assert theme_mod.normalise_tag("MMO") == "massively multiplayer"
    assert theme_mod.normalise_tag("Top-Down") == "top-down"


def test_normalise_tag_empty() -> None:
    assert theme_mod.normalise_tag(None) == ""
    assert theme_mod.normalise_tag("!!!") == ""


# --------------------------------------------------------------------------- #
# collection + clustering
# --------------------------------------------------------------------------- #
def test_collect_source_tags(tagged_config: Config) -> None:
    tags = theme_mod.collect_source_tags(tagged_config)
    assert {"source", "raw_tag", "normalised", "frequency"} == set(tags.columns)
    assert set(tags["source"]) == {"movie", "game"}
    assert (tags["frequency"] > 0).all()


def test_cluster_tags_separates_groups() -> None:
    vectors = np.zeros((6, 3), dtype=np.float32)
    vectors[0:3, 0] = 1.0
    vectors[0, 1] = 0.02
    vectors[3:6, 2] = 1.0
    labels = theme_mod.cluster_tags(
        vectors, ["a", "b", "c", "d", "e", "f"], np.ones(6, dtype=int), n_themes=2
    )
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_cluster_tags_single_cluster() -> None:
    labels = theme_mod.cluster_tags(np.ones((3, 2)), ["a", "b", "c"], np.ones(3), n_themes=1)
    assert labels.tolist() == [0, 0, 0]


def test_name_clusters_prefers_frequent_then_central() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.0], [0.1, 1.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    frequencies = np.array([1, 9, 5, 5])
    names = theme_mod.name_clusters(labels, ["rare", "common", "b", "c"], frequencies, vectors)
    assert names[0] == "common"
    assert names[1] in {"b", "c"}


def test_assign_by_similarity_maps_to_nearest_cluster() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.array([0, 1])
    query = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
    assert theme_mod.assign_by_similarity(vectors, labels, query).tolist() == [0, 1]


# --------------------------------------------------------------------------- #
# theme map build
# --------------------------------------------------------------------------- #
def _groups_from_cfg(cfg: Config) -> dict[str, list[str]]:
    tags = theme_mod.collect_source_tags(cfg)
    unique = sorted(tags["raw_tag"].str.lower().unique().tolist())
    midpoint = len(unique) // 2
    return {"a": unique[:midpoint], "b": unique[midpoint:]}


def test_build_theme_map_writes_csv_and_covers_every_tag(tagged_config: Config) -> None:
    fake = FakeEmbedder(_groups_from_cfg(tagged_config))
    result = theme_mod.build_theme_map(tagged_config, client=fake)
    assert result.source == "clustered"
    assert result.n_themes >= 2
    path = tagged_config.theme_map_path
    assert path.exists()

    written = pd.read_csv(path, dtype=str)
    assert list(written.columns) == ["source", "raw_tag", "theme"]
    collected = theme_mod.collect_source_tags(tagged_config)
    assert set(collected["raw_tag"]) <= set(written["raw_tag"])


def test_existing_theme_map_is_reused_and_not_callable(tagged_config: Config) -> None:
    fake = FakeEmbedder(_groups_from_cfg(tagged_config))
    first = theme_mod.build_theme_map(tagged_config, client=fake)
    assert fake.calls == 1

    again = theme_mod.build_theme_map(tagged_config, client=fake)
    assert fake.calls == 1, "a second run must not re-embed"
    assert again.n_themes == first.n_themes
    assert again.source == "theme_map.csv"


def test_hand_edited_theme_map_overrides_clustering(tagged_config: Config) -> None:
    fake = FakeEmbedder(_groups_from_cfg(tagged_config))
    theme_mod.build_theme_map(tagged_config, client=fake)
    path = tagged_config.theme_map_path
    edited = pd.read_csv(path, dtype=str)
    edited.loc[0, "theme"] = "my-theme"
    edited.to_csv(path, index=False)

    reloaded = theme_mod.build_theme_map(tagged_config, client=fake)
    assert "my-theme" in set(reloaded.themes)


def test_theme_lookup_covers_normalised_tags(tagged_config: Config) -> None:
    fake = FakeEmbedder(_groups_from_cfg(tagged_config))
    result = theme_mod.build_theme_map(tagged_config, client=fake)
    lookup = theme_mod.theme_lookup(result.theme_map)
    for raw in result.theme_map["raw_tag"]:
        assert lookup[theme_mod.normalise_tag(raw)]


def test_build_theme_map_requires_data(make_config) -> None:
    cfg = make_config("tiny")
    with pytest.raises(RuntimeError):
        theme_mod.build_theme_map(cfg, client=FakeEmbedder())


def test_embed_tag_names_uses_cache(tagged_config: Config, tmp_path) -> None:
    fake = FakeEmbedder()
    vectors, hit = theme_mod.embed_tag_names(tagged_config, ["alpha", "beta"], client=fake)
    assert not hit and vectors.shape == (2, 8)
    vectors2, hit2 = theme_mod.embed_tag_names(tagged_config, ["alpha", "beta"], client=fake)
    assert hit2
    assert np.allclose(vectors, vectors2)


# --------------------------------------------------------------------------- #
# per item themes + genres
# --------------------------------------------------------------------------- #
def test_aggregate_theme_weights_top_n_and_normalisation() -> None:
    frame = pd.DataFrame(
        {
            "movie_id": [1, 1, 1, 1],
            "tag": ["gore", "bloody", "space", "puzzle"],
            "weight": [0.4, 0.2, 0.9, 0.1],
        }
    )
    lookup = {"gore": "gore", "bloody": "gore", "space": "space", "puzzle": "puzzle"}
    result = tax_pipeline._aggregate_theme_weights(frame, "movie_id", lookup, top_n=2)
    assert len(result) == 2
    assert result.iloc[0]["theme"] == "space"
    assert result.iloc[0]["weight"] == pytest.approx(1.0)


def test_aggregate_theme_weights_skips_unknown_tags() -> None:
    frame = pd.DataFrame({"movie_id": [1], "tag": ["nonsense"], "weight": [1.0]})
    assert tax_pipeline._aggregate_theme_weights(frame, "movie_id", {}, top_n=5).empty


def test_taxonomy_run_end_to_end(tagged_config: Config) -> None:
    fake = FakeEmbedder(_groups_from_cfg(tagged_config))
    assert tax_pipeline.run(tagged_config, None, client=fake) == 0
    processed = tagged_config.paths.processed
    movie_themes = pd.read_parquet(processed / theme_mod.MOVIE_THEMES_PARQUET)
    game_themes = pd.read_parquet(processed / theme_mod.GAME_THEMES_PARQUET)
    item_genres = pd.read_parquet(processed / theme_mod.ITEM_GENRES_PARQUET)

    assert not movie_themes.empty
    assert not game_themes.empty
    assert set(item_genres["domain"]) == {"movie", "game"}
    assert movie_themes["weight"].between(0, 1).all()
    assert tagged_config.report_path(tax_pipeline.TAXONOMY_MD).exists()

    movies = pd.read_parquet(processed / "movies.parquet")
    assert set(movie_themes["movie_id"]) <= set(movies["movie_id"])
    games = pd.read_parquet(processed / "games.parquet")
    assert set(game_themes["app_id"]) <= set(games["app_id"])
