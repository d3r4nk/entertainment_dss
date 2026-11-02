"""MovieLens (ml-25m) loading, cleaning, sampling and extraction.

Raw files are read-only; every artefact produced here lands in
``data/processed``. Titles are split into ``title``/``year`` and genres are kept
as a list of the original MovieLens genre strings (the canonical taxonomy is
applied later, in :mod:`entdec.taxonomy`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from entdec.config import Config, ProfileConfig
from entdec.data.io import (
    ParquetBatchWriter,
    iter_csv_batches,
    iter_parquet_batches,
    read_parquet,
    write_parquet,
)
from entdec.data.sampling import accumulate_counts, select_items, select_users, user_stats_frame
from entdec.data.splits import assign_splits, split_sizes, validate_no_leakage
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

MOVIE_ID = "movieId"
USER_ID = "userId"

_TITLE_RE = re.compile(r"^(?P<title>.*?)\s*\((?P<year>\d{4})\)\s*$")
_NO_GENRES = {"(no genres listed)", "no genres listed", ""}

# File names inside data/movie_data
MOVIES_CSV = "movies.csv"
RATINGS_CSV = "ratings.csv"
TAGS_CSV = "tags.csv"
GENOME_SCORES_CSV = "genome-scores.csv"
GENOME_TAGS_CSV = "genome-tags.csv"

# Output file names inside data/processed
MOVIES_PARQUET = "movies.parquet"
MOVIE_RATINGS_PARQUET = "movie_ratings.parquet"
MOVIE_GENOME_PARQUET = "movie_genome.parquet"
MOVIE_USER_TAGS_PARQUET = "movie_user_tags.parquet"
MOVIE_TEMP_PARQUET = "_tmp_movie_ratings.parquet"


@dataclass
class MovieStats:
    """Summary of what the movie pipeline produced."""

    n_items: int = 0
    n_users: int = 0
    n_interactions: int = 0
    n_positives: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


def parse_title(raw: str) -> tuple[str, int]:
    """Split ``"Toy Story (1995)"`` into ``("Toy Story", 1995)``.

    Returns a year of ``0`` when the title carries no release year.
    """
    match = _TITLE_RE.match(str(raw).strip())
    if not match:
        return str(raw).strip(), 0
    return match.group("title").strip(), int(match.group("year"))


def parse_genres(raw: str) -> list[str]:
    """Split a pipe separated genre string, dropping the "no genres" marker."""
    if not isinstance(raw, str):
        return []
    return [genre.strip() for genre in raw.split("|") if genre.strip() not in _NO_GENRES]


def load_movies(cfg: Config) -> pd.DataFrame:
    """Read ``movies.csv`` into a cleaned frame (all movies, unfiltered)."""
    path = cfg.raw_path("movie", MOVIES_CSV)
    raw = pd.read_csv(path, dtype={"movieId": "int64", "title": "string", "genres": "string"})
    parsed = [parse_title(title) for title in raw["title"].tolist()]
    frame = pd.DataFrame(
        {
            "movie_id": raw["movieId"].astype("int32"),
            "title": [item[0] for item in parsed],
            "year": np.array([item[1] for item in parsed], dtype="int16"),
            "genres": [parse_genres(value) for value in raw["genres"].tolist()],
        }
    )
    logger.info("loaded %d movies from %s", len(frame), path.name)
    return frame


def rating_stats(cfg: Config) -> pd.DataFrame:
    """Count ratings and sum scores per movie (single streaming pass).

    Returns:
        Frame with ``movie_id``, ``n_ratings`` and ``rating_avg``.
    """
    path = cfg.raw_path("movie", RATINGS_CSV)
    counts: np.ndarray | None = None
    sums: np.ndarray | None = None
    rows = 0
    for batch in iter_csv_batches(path, ["movieId", "rating"]):
        ids = batch["movieId"].to_numpy()
        scores = batch["rating"].to_numpy(dtype="float64")
        counts = accumulate_counts(counts, ids)
        # Weighted bincount gives the per-movie rating sum in the same pass.
        batch_sums = np.bincount(ids, weights=scores, minlength=int(ids.max()) + 1)
        if sums is None or sums.size < batch_sums.size:
            grown = np.zeros(batch_sums.size, dtype="float64")
            if sums is not None:
                grown[: sums.size] = sums
            sums = grown
        sums[: batch_sums.size] += batch_sums
        rows += len(batch)

    observed = np.nonzero(counts)[0] if counts is not None else np.zeros(0, dtype="int64")
    frame = pd.DataFrame(
        {
            "movie_id": observed.astype("int32"),
            "n_ratings": counts[observed].astype("int32") if counts is not None else 0,
        }
    )
    rating_sums = sums[observed] if sums is not None else np.zeros(0)
    frame["rating_avg"] = (rating_sums / frame["n_ratings"].to_numpy()).astype("float32")
    logger.info("scanned %d ratings for %d movies", rows, len(frame))
    return frame


def extract_ratings(cfg: Config, movie_ids: np.ndarray, temp_path: Path) -> int:
    """Stream ``ratings.csv`` and keep only rows for the sampled movies.

    Returns:
        Number of kept rows.
    """
    wanted = set(np.asarray(movie_ids).tolist())
    path = cfg.raw_path("movie", RATINGS_CSV)
    kept = 0
    with ParquetBatchWriter(temp_path) as writer:
        for batch in iter_csv_batches(path, ["userId", "movieId", "rating", "timestamp"]):
            mask = batch["movieId"].isin(wanted)
            if not mask.any():
                continue
            chunk = pd.DataFrame(
                {
                    "uid": batch.loc[mask, "userId"].astype("int32"),
                    "movie_id": batch.loc[mask, "movieId"].astype("int32"),
                    "rating": batch.loc[mask, "rating"].astype("float32"),
                    "ts": batch.loc[mask, "timestamp"].astype("int64"),
                }
            )
            kept += len(chunk)
            writer.write(chunk)
    logger.info("kept %d ratings for %d sampled movies", kept, len(wanted))
    return kept


def build_interactions(
    cfg: Config,
    temp_path: Path,
    profile: ProfileConfig,
    min_rating: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select users from the sampled-movie ratings and finalise the interactions.

    Returns:
        ``(interactions, user_ids)`` where ``interactions`` carries
        ``is_positive`` and ``split``.
    """
    stats = user_stats_frame(_temp_with_flags(temp_path, min_rating))
    users = select_users(stats, profile)
    if users.size == 0:
        return pd.DataFrame(
            columns=["uid", "movie_id", "rating", "ts", "is_positive", "split"]
        ), users

    wanted = set(users.tolist())
    kept: list[pd.DataFrame] = []
    for batch in iter_parquet_batches(temp_path, ["uid", "movie_id", "rating", "ts"]):
        mask = batch["uid"].isin(wanted)
        if mask.any():
            kept.append(batch.loc[mask])
    interactions = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    interactions["is_positive"] = interactions["rating"] >= float(min_rating)
    interactions = assign_splits(interactions)
    interactions["split"] = interactions["split"].astype("category")

    leaked = validate_no_leakage(interactions, item_col="movie_id")
    if not leaked.empty:
        logger.warning("%d users leak train items into evaluation", len(leaked))
    logger.info("movie interactions: %s", split_sizes(interactions))
    return interactions, users


def _temp_with_flags(temp_path: Path, min_rating: float) -> pd.DataFrame:
    """Read the temp ratings file with the positive flag added (streamed concat)."""
    parts: list[pd.DataFrame] = []
    for batch in iter_parquet_batches(temp_path, ["uid", "rating"]):
        part = pd.DataFrame(
            {"uid": batch["uid"].astype("int32"), "is_positive": batch["rating"] >= float(min_rating)}
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["uid", "is_positive"], dtype=object)
    return pd.concat(parts, ignore_index=True)


def load_genome(cfg: Config, movie_ids: np.ndarray, out_path: Path) -> int:
    """Extract genome relevance rows for the sampled movies (streamed)."""
    wanted = set(np.asarray(movie_ids).tolist())
    tag_path = cfg.raw_path("movie", GENOME_TAGS_CSV)
    tags = pd.read_csv(tag_path, dtype={"tagId": "int32", "tag": "string"})
    tag_names = dict(zip(tags["tagId"].tolist(), tags["tag"].tolist(), strict=False))

    path = cfg.raw_path("movie", GENOME_SCORES_CSV)
    kept = 0
    with ParquetBatchWriter(out_path) as writer:
        for batch in iter_csv_batches(path, ["movieId", "tagId", "relevance"]):
            mask = batch["movieId"].isin(wanted) & (batch["relevance"] > 0)
            if not mask.any():
                continue
            chunk = pd.DataFrame(
                {
                    "movie_id": batch.loc[mask, "movieId"].astype("int32"),
                    "tag_id": batch.loc[mask, "tagId"].astype("int32"),
                    "tag": batch.loc[mask, "tagId"].map(tag_names).astype("string"),
                    "relevance": batch.loc[mask, "relevance"].astype("float32"),
                }
            )
            kept += len(chunk)
            writer.write(chunk)
    logger.info("kept %d genome rows for %d movies", kept, len(wanted))
    return kept


def load_user_tags(cfg: Config, movie_ids: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """Aggregate free-text tags per sampled movie, keeping the ``top_n`` frequent ones."""
    path = cfg.raw_path("movie", TAGS_CSV)
    tags = pd.read_csv(path, dtype={"movieId": "int32", "tag": "string"})
    wanted = set(np.asarray(movie_ids).tolist())
    tags = tags[tags["movieId"].isin(wanted)].copy()
    if tags.empty:
        return pd.DataFrame(columns=["movie_id", "tag", "count"])
    tags["tag"] = tags["tag"].str.strip().str.lower()
    tags = tags[tags["tag"].notna() & (tags["tag"] != "")]
    grouped = (
        tags.groupby(["movieId", "tag"], sort=False)
        .size()
        .reset_index(name="count")
        .rename(columns={"movieId": "movie_id"})
    )
    grouped["count"] = grouped["count"].astype("int32")
    grouped = grouped.sort_values(["movie_id", "count", "tag"], ascending=[True, False, True])
    trimmed = grouped.groupby("movie_id", sort=False).head(top_n).reset_index(drop=True)
    logger.info("kept %d user-tag rows for %d movies", len(trimmed), trimmed["movie_id"].nunique())
    return trimmed


def run(cfg: Config, force: bool = False) -> MovieStats:
    """Run the full MovieLens preparation step and write the Parquet artefacts."""
    profile = cfg.profile
    processed = cfg.paths.processed
    stats = MovieStats()

    if not force and (processed / MOVIES_PARQUET).exists() and (processed / MOVIE_RATINGS_PARQUET).exists():
        logger.info("movie artefacts already exist - use --force to rebuild")
        return stats

    min_rating = float(cfg.get("positive.movie_min_rating", 4.0))

    # 1. global rating statistics (one pass) -> item selection
    stats_frame = rating_stats(cfg)
    item_ids = select_items(
        stats_frame.set_index("movie_id")["n_ratings"], profile.max_items, profile.min_movie_ratings
    )
    if item_ids.size == 0:
        logger.error("no movies selected - aborting movie step")
        return stats

    # 2. extract ratings for the sampled movies (second pass)
    temp_path = processed / MOVIE_TEMP_PARQUET
    extract_ratings(cfg, item_ids, temp_path)

    # 3. pick users, finalise interactions, assign splits
    interactions, users = build_interactions(cfg, temp_path, profile, min_rating)
    temp_path.unlink(missing_ok=True)
    if interactions.empty:
        logger.error("no movie interactions after user sampling")
        return stats

    # 4. item table enriched with the global popularity signals
    movies = load_movies(cfg)
    movies = movies[movies["movie_id"].isin(set(item_ids.tolist()))].copy()
    movies = movies.merge(
        stats_frame[["movie_id", "n_ratings", "rating_avg"]],
        on="movie_id",
        how="left",
    ).sort_values("movie_id", ignore_index=True)
    movies["n_ratings"] = movies["n_ratings"].fillna(0).astype("int32")
    movies["rating_avg"] = movies["rating_avg"].fillna(0.0).astype("float32")
    movies = movies[["movie_id", "title", "year", "genres", "n_ratings", "rating_avg"]]

    # 5. genome relevance + free-text tags
    genome_path = processed / MOVIE_GENOME_PARQUET
    load_genome(cfg, item_ids, genome_path)
    user_tags = load_user_tags(cfg, item_ids)

    write_parquet(processed / MOVIES_PARQUET, movies)
    write_parquet(processed / MOVIE_RATINGS_PARQUET, interactions)
    write_parquet(processed / MOVIE_USER_TAGS_PARQUET, user_tags)

    stats = MovieStats(
        n_items=len(movies),
        n_users=int(users.size),
        n_interactions=len(interactions),
        n_positives=int(interactions["is_positive"].sum()),
        splits=split_sizes(interactions),
        files={
            "movies": str(processed / MOVIES_PARQUET),
            "ratings": str(processed / MOVIE_RATINGS_PARQUET),
            "genome": str(genome_path),
            "user_tags": str(processed / MOVIE_USER_TAGS_PARQUET),
        },
    )
    logger.info(
        "movies: %d items, %d users, %d interactions (%d positive)",
        stats.n_items,
        stats.n_users,
        stats.n_interactions,
        stats.n_positives,
    )
    return stats


def load_processed(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ``movies``, ``movie_ratings`` and ``movie_genome`` from ``data/processed``."""
    processed = cfg.paths.processed
    movies = read_parquet(processed / MOVIES_PARQUET)
    ratings = read_parquet(processed / MOVIE_RATINGS_PARQUET)
    genome_path = processed / MOVIE_GENOME_PARQUET
    genome = read_parquet(genome_path) if genome_path.exists() else pd.DataFrame()
    return movies, ratings, genome


__all__ = [
    "GENOME_SCORES_CSV",
    "GENOME_TAGS_CSV",
    "MOVIES_CSV",
    "MOVIES_PARQUET",
    "MOVIE_GENOME_PARQUET",
    "MOVIE_RATINGS_PARQUET",
    "MOVIE_TEMP_PARQUET",
    "MOVIE_USER_TAGS_PARQUET",
    "MOVIE_ID",
    "RATINGS_CSV",
    "TAGS_CSV",
    "USER_ID",
    "MovieStats",
    "build_interactions",
    "extract_ratings",
    "load_genome",
    "load_movies",
    "load_processed",
    "load_user_tags",
    "parse_genres",
    "parse_title",
    "rating_stats",
    "run",
]
