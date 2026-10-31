"""Loader tests against the synthetic dataset tree from ``conftest``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from entdec.config import Config
from entdec.data import movielens, steam
from entdec.data.io import read_parquet


# --------------------------------------------------------------------------- #
# MovieLens
# --------------------------------------------------------------------------- #
def test_parse_title_with_year() -> None:
    assert movielens.parse_title("Toy Story (1995)") == ("Toy Story", 1995)


def test_parse_title_without_year() -> None:
    assert movielens.parse_title("Weird Title") == ("Weird Title", 0)


def test_parse_title_keeps_inner_parentheses() -> None:
    title, year = movielens.parse_title("Kill Bill: Vol. 1 (2003)")
    assert (title, year) == ("Kill Bill: Vol. 1", 2003)


def test_parse_genres_drops_placeholder() -> None:
    assert movielens.parse_genres("Comedy|Drama") == ["Comedy", "Drama"]
    assert movielens.parse_genres("(no genres listed)") == []
    assert movielens.parse_genres(None) == []


def test_load_movies(synthetic_config: Config) -> None:
    movies = movielens.load_movies(synthetic_config)
    assert len(movies) == 10
    assert movies["movie_id"].dtype == np.int32
    assert movies.iloc[0]["title"] == "Toy Story"
    assert movies.iloc[0]["year"] == 1995
    assert "Animation" in movies.iloc[0]["genres"]


def test_rating_stats_counts_and_average(synthetic_config: Config) -> None:
    stats = movielens.rating_stats(synthetic_config).set_index("movie_id")
    assert stats.loc[1, "n_ratings"] == 8
    assert 0 < stats.loc[1, "rating_avg"] <= 5


def test_movie_pipeline_end_to_end(synthetic_config: Config) -> None:
    stats = movielens.run(synthetic_config, force=True)
    assert stats.n_items > 0
    assert stats.n_users > 0
    assert stats.n_positives > 0
    processed = synthetic_config.paths.processed

    movies = read_parquet(processed / movielens.MOVIES_PARQUET)
    ratings = read_parquet(processed / movielens.MOVIE_RATINGS_PARQUET)
    genome = read_parquet(processed / movielens.MOVIE_GENOME_PARQUET)
    user_tags = read_parquet(processed / movielens.MOVIE_USER_TAGS_PARQUET)

    assert set(ratings["movie_id"]).issubset(set(movies["movie_id"]))
    assert ratings["is_positive"].dtype == bool
    assert set(ratings["split"].unique()) <= {"train", "valid", "test"}
    assert not genome.empty
    assert {"movie_id", "tag_id", "tag", "relevance"} <= set(genome.columns)
    assert not user_tags.empty
    assert not (processed / movielens.MOVIE_TEMP_PARQUET).exists()


def test_movie_split_has_no_leakage(synthetic_config: Config) -> None:
    movielens.run(synthetic_config, force=True)
    ratings = read_parquet(synthetic_config.paths.processed / movielens.MOVIE_RATINGS_PARQUET)
    from entdec.data.splits import validate_no_leakage

    assert validate_no_leakage(ratings, item_col="movie_id").empty


def test_movie_step_is_idempotent(synthetic_config: Config) -> None:
    first = movielens.run(synthetic_config, force=True)
    second = movielens.run(synthetic_config, force=False)
    # Second run short-circuits without touching anything.
    assert second.n_items == 0
    again = movielens.run(synthetic_config, force=True)
    assert again.n_items == first.n_items


def test_raw_data_is_never_modified(synthetic_config: Config) -> None:
    raw = synthetic_config.raw_path("movie", movielens.RATINGS_CSV)
    before = raw.read_bytes()
    movielens.run(synthetic_config, force=True)
    assert raw.read_bytes() == before


# --------------------------------------------------------------------------- #
# Steam
# --------------------------------------------------------------------------- #
def test_as_bool_handles_strings(synthetic_config: Config) -> None:
    assert steam.as_bool(pd.Series(["true", "false", "TRUE"])).tolist() == [True, False, True]


def test_parse_year_variants() -> None:
    assert steam.parse_year("2008-11-21") == 2008
    assert steam.parse_year("") == 0
    assert steam.parse_year(np.nan) == 0


def test_parse_date_seconds_handles_bad_dates() -> None:
    values = pd.Series(["2021-05-05", "", "not-a-date"])
    seconds = steam.parse_date_seconds(values)
    assert seconds[0] > 0
    assert seconds[1] == 0
    assert seconds[2] == 0


def test_parse_date_seconds_uses_seconds_not_nanoseconds() -> None:
    seconds = steam.parse_date_seconds(pd.Series(["2021-05-05"]))
    # 2021 epoch seconds are ~1.6e9; nanoseconds would be ~1.6e18.
    assert 1.5e9 < seconds[0] < 1.8e9


def test_load_games_and_metadata(synthetic_config: Config) -> None:
    games = steam.load_games(synthetic_config)
    metadata = steam.load_metadata(synthetic_config)
    assert len(games) == 10
    assert len(metadata) == 10
    assert "Windows" in games.iloc[0]["platforms"]
    assert metadata.iloc[0]["tags"]


def test_review_counts(synthetic_config: Config) -> None:
    counts = steam.review_counts(synthetic_config)
    assert counts.loc[1000] == 8


def test_game_pipeline_end_to_end(synthetic_config: Config) -> None:
    stats = steam.run(synthetic_config, force=True)
    assert stats.n_items > 0
    assert stats.n_users > 0
    processed = synthetic_config.paths.processed

    games = read_parquet(processed / steam.GAMES_PARQUET)
    reviews = read_parquet(processed / steam.GAME_REVIEWS_PARQUET)
    tags = read_parquet(processed / steam.GAME_TAGS_PARQUET)

    assert set(reviews["app_id"]).issubset(set(games["app_id"]))
    assert reviews["is_positive"].dtype == bool
    assert {"train", "valid", "test"} >= set(reviews["split"].unique())
    assert {"app_id", "tag", "rank", "weight"} <= set(tags.columns)
    assert tags["weight"].between(0, 1).all()


def test_game_positive_requires_recommendation(synthetic_config: Config) -> None:
    steam.run(synthetic_config, force=True)
    reviews = read_parquet(synthetic_config.paths.processed / steam.GAME_REVIEWS_PARQUET)
    assert (reviews["is_positive"] <= reviews["is_recommended"]).all()


def test_build_tag_table_weights_decay() -> None:
    games = pd.DataFrame({"app_id": [1]})
    metadata = pd.DataFrame({"app_id": [1], "tags": [["A", "B", "C"]]})
    table = steam.build_tag_table(games, metadata, top_n=3)
    assert table["tag"].tolist() == ["A", "B", "C"]
    assert table["rank"].tolist() == [1, 2, 3]
    assert table["weight"].is_monotonic_decreasing
    assert table.iloc[0]["weight"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_data_report_is_written(synthetic_config: Config) -> None:
    from entdec.data import pipeline

    assert pipeline.run(synthetic_config, None) == 0
    report = synthetic_config.report_path(pipeline.DATA_STATS_MD)
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "Dataset statistics" in text
    assert "| items |" in text
    assert synthetic_config.report_path(pipeline.DATA_STATS_JSON).exists()
