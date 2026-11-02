"""Kaggle "Game Recommendations on Steam" loading, cleaning and sampling.

``recommendations.csv`` is ~1.9 GB / 41M rows and is always streamed. A positive
interaction is a recommendation with ``is_recommended == True`` (and at least
``positive.game_min_hours`` hours played). Steam tags come from
``games_metadata.json`` and are kept with their original ordering, which is a
popularity ranking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from entdec.config import Config, ProfileConfig
from entdec.data.enrich import complete_game_tags
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

GAMES_CSV = "games.csv"
METADATA_JSON = "games_metadata.json"
USERS_CSV = "users.csv"
RECOMMENDATIONS_CSV = "recommendations.csv"

GAMES_PARQUET = "games.parquet"
GAME_REVIEWS_PARQUET = "game_reviews.parquet"
GAME_TAGS_PARQUET = "game_tags.parquet"
GAME_TEMP_PARQUET = "_tmp_game_reviews.parquet"

_YEAR_RE = re.compile(r"^(\d{4})")
_TRUE_VALUES = {"true", "1", "t", "yes", "y"}


@dataclass
class GameStats:
    """Summary of what the games pipeline produced."""

    n_items: int = 0
    n_users: int = 0
    n_interactions: int = 0
    n_positives: int = 0
    splits: dict[str, int] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)


def as_bool(values: pd.Series | np.ndarray) -> np.ndarray:
    """Convert Steam's "true"/"false" strings (or booleans) into a bool array."""
    series = pd.Series(values) if not isinstance(values, pd.Series) else values
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    return series.astype("string").str.strip().str.lower().isin(_TRUE_VALUES).to_numpy(dtype=bool)


def parse_year(raw: object) -> int:
    """Extract a 4 digit year from ``date_release``; ``0`` when unknown."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return 0
    match = _YEAR_RE.match(str(raw).strip())
    return int(match.group(1)) if match else 0


def parse_date_seconds(values: pd.Series) -> np.ndarray:
    """Convert ``YYYY-MM-DD`` dates to epoch **seconds** (``0`` when unparseable).

    Seconds keep the game and movie interaction timestamps in the same unit.
    """
    parsed = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce").to_numpy(
        dtype="datetime64[ns]"
    )
    seconds = parsed.astype("int64") // 1_000_000_000
    seconds[np.isnat(parsed)] = 0
    seconds[seconds < 0] = 0
    return seconds


def load_games(cfg: Config) -> pd.DataFrame:
    """Read ``games.csv`` (metadata only, no descriptions)."""
    path = cfg.raw_path("game", GAMES_CSV)
    raw = pd.read_csv(path)
    frame = pd.DataFrame(
        {
            "app_id": pd.to_numeric(raw["app_id"], errors="coerce").astype("int32"),
            "title": raw["title"].astype("string"),
            "year": np.array([parse_year(value) for value in raw["date_release"]], dtype="int16"),
            "rating_label": raw["rating"].astype("string"),
            "positive_ratio": pd.to_numeric(raw["positive_ratio"], errors="coerce")
            .fillna(0)
            .astype("int16"),
            "n_reviews": pd.to_numeric(raw["user_reviews"], errors="coerce").fillna(0).astype("int32"),
            "price": pd.to_numeric(raw["price_final"], errors="coerce").fillna(0.0).astype("float32"),
            "platforms": [
                [
                    name
                    for name, flag in (("Windows", win), ("Mac", mac), ("Linux", linux))
                    if flag
                ]
                for win, mac, linux in zip(
                    as_bool(raw["win"]), as_bool(raw["mac"]), as_bool(raw["linux"]), strict=False
                )
            ],
        }
    )
    logger.info("loaded %d games from %s", len(frame), path.name)
    return frame


def load_metadata(cfg: Config) -> pd.DataFrame:
    """Read the JSON-lines metadata file (description + ordered tag list)."""
    path = cfg.raw_path("game", METADATA_JSON)
    app_ids: list[int] = []
    descriptions: list[str] = []
    tags: list[list[str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            app_ids.append(int(record.get("app_id", 0)))
            descriptions.append(str(record.get("description") or ""))
            raw_tags = record.get("tags") or []
            tags.append([str(tag).strip() for tag in raw_tags if str(tag).strip()])
    frame = pd.DataFrame(
        {"app_id": np.array(app_ids, dtype="int32"), "description": descriptions, "tags": tags}
    )
    logger.info("loaded metadata for %d games", len(frame))
    return frame


def review_counts(cfg: Config) -> pd.Series:
    """Count reviews per app id (single streaming pass over recommendations.csv)."""
    path = cfg.raw_path("game", RECOMMENDATIONS_CSV)
    counts: np.ndarray | None = None
    rows = 0
    for batch in iter_csv_batches(path, ["app_id"]):
        counts = accumulate_counts(counts, batch["app_id"].to_numpy())
        rows += len(batch)
    if counts is None:
        return pd.Series(dtype="int64")
    observed = np.nonzero(counts)[0]
    series = pd.Series(counts[observed], index=observed, name="n_review_rows")
    logger.info("scanned %d recommendation rows for %d games", rows, len(series))
    return series


def extract_reviews(cfg: Config, app_ids: np.ndarray, temp_path: Path) -> int:
    """Stream recommendations.csv, keeping rows for the sampled games."""
    wanted = set(np.asarray(app_ids).tolist())
    path = cfg.raw_path("game", RECOMMENDATIONS_CSV)
    kept = 0
    with ParquetBatchWriter(temp_path) as writer:
        for batch in iter_csv_batches(
            path, ["app_id", "date", "is_recommended", "hours", "user_id", "review_id"]
        ):
            mask = batch["app_id"].isin(wanted)
            if not mask.any():
                continue
            chunk = batch.loc[mask]
            frame = pd.DataFrame(
                {
                    "uid": chunk["user_id"].astype("int32"),
                    "app_id": chunk["app_id"].astype("int32"),
                    "review_id": chunk["review_id"].astype("int32"),
                    "is_recommended": as_bool(chunk["is_recommended"]),
                    "hours": chunk["hours"].astype("float32"),
                    "ts": parse_date_seconds(chunk["date"]),
                }
            )
            kept += len(frame)
            writer.write(frame)
    logger.info("kept %d reviews for %d sampled games", kept, len(wanted))
    return kept


def _temp_with_flags(temp_path: Path, min_hours: float) -> pd.DataFrame:
    """Read the temp reviews file with the positive flag added."""
    parts: list[pd.DataFrame] = []
    for batch in iter_parquet_batches(temp_path, ["uid", "is_recommended", "hours"]):
        part = pd.DataFrame(
            {
                "uid": batch["uid"].astype("int32"),
                "is_positive": batch["is_recommended"].to_numpy(dtype=bool)
                & (batch["hours"].to_numpy() >= float(min_hours)),
            }
        )
        parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["uid", "is_positive"], dtype=object)
    return pd.concat(parts, ignore_index=True)


def build_interactions(
    cfg: Config, temp_path: Path, profile: ProfileConfig, min_hours: float
) -> tuple[pd.DataFrame, np.ndarray]:
    """Select users from the sampled-game reviews and finalise the interactions."""
    stats = user_stats_frame(_temp_with_flags(temp_path, min_hours))
    users = select_users(stats, profile)
    if users.size == 0:
        return pd.DataFrame(
            columns=["uid", "app_id", "review_id", "is_recommended", "hours", "ts", "is_positive", "split"]
        ), users

    wanted = set(users.tolist())
    kept: list[pd.DataFrame] = []
    for batch in iter_parquet_batches(
        temp_path, ["uid", "app_id", "review_id", "is_recommended", "hours", "ts"]
    ):
        mask = batch["uid"].isin(wanted)
        if mask.any():
            kept.append(batch.loc[mask])
    interactions = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame()
    interactions["is_positive"] = interactions["is_recommended"].to_numpy(dtype=bool) & (
        interactions["hours"].to_numpy() >= float(min_hours)
    )
    interactions = assign_splits(interactions)
    interactions["split"] = interactions["split"].astype("category")

    leaked = validate_no_leakage(interactions, item_col="app_id")
    if not leaked.empty:
        logger.warning("%d users leak train items into evaluation", len(leaked))
    logger.info("game interactions: %s", split_sizes(interactions))
    return interactions, users


def build_tag_table(games: pd.DataFrame, metadata: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Long table of the top ``top_n`` Steam tags per game with rank-based weights."""
    rows: list[pd.DataFrame] = []
    for app_id, tags in zip(metadata["app_id"], metadata["tags"], strict=False):
        selected = [tag for tag in tags if tag][:top_n]
        if not selected:
            continue
        count = len(selected)
        rows.append(
            pd.DataFrame(
                {
                    "app_id": app_id,
                    "tag": selected,
                    "rank": np.arange(1, count + 1, dtype="int16"),
                    # Linear decay: the first tag weighs 1.0, the last 1/count.
                    "weight": ((count - np.arange(count)) / count).astype("float32"),
                }
            )
        )
    if not rows:
        return pd.DataFrame(columns=["app_id", "tag", "rank", "weight"])
    tags = pd.concat(rows, ignore_index=True)
    tags["app_id"] = tags["app_id"].astype("int32")
    logger.info("built %d tag rows for %d games", len(tags), tags["app_id"].nunique())
    return tags


def run(cfg: Config, force: bool = False) -> GameStats:
    """Run the full Steam preparation step and write the Parquet artefacts."""
    profile = cfg.profile
    processed = cfg.paths.processed
    stats = GameStats()

    if not force and (processed / GAMES_PARQUET).exists() and (processed / GAME_REVIEWS_PARQUET).exists():
        logger.info("game artefacts already exist - use --force to rebuild")
        return stats

    min_hours = float(cfg.get("positive.game_min_hours", 0.0))
    require_positive = bool(cfg.get("positive.game_requires_recommendation", True))
    if not require_positive:
        logger.warning("positive.game_requires_recommendation is false - every review counts")

    # 1. review counts (one pass) -> item selection
    counts = review_counts(cfg)
    app_ids = select_items(counts, profile.max_items, profile.min_game_reviews)
    if app_ids.size == 0:
        logger.error("no games selected - aborting game step")
        return stats

    # 2. extract reviews for the sampled games (second pass)
    temp_path = processed / GAME_TEMP_PARQUET
    extract_reviews(cfg, app_ids, temp_path)

    # 3. pick users, finalise interactions, assign splits
    interactions, users = build_interactions(cfg, temp_path, profile, min_hours)
    if interactions.empty:
        logger.error("no game interactions after user sampling")
        temp_path.unlink(missing_ok=True)
        return stats

    # 4. item table joined with metadata (description + tags)
    games = load_games(cfg)
    sampled_ids = set(app_ids.tolist())
    metadata = load_metadata(cfg)
    metadata = metadata[metadata["app_id"].isin(sampled_ids)].reset_index(drop=True)
    games = games[games["app_id"].isin(sampled_ids)].copy()
    games = games.merge(metadata, on="app_id", how="left")
    games["description"] = games["description"].fillna("").astype("string")
    games["tags"] = games["tags"].apply(lambda value: value if isinstance(value, list) else [])
    games = games.sort_values("app_id", ignore_index=True)
    games["n_reviews"] = games["n_reviews"].fillna(0).astype("int32")
    games = games[
        [
            "app_id",
            "title",
            "year",
            "rating_label",
            "positive_ratio",
            "n_reviews",
            "price",
            "platforms",
            "description",
            "tags",
        ]
    ]

    top_tags = int(cfg.get("taxonomy.game_top_themes", 10))
    tag_table = build_tag_table(games, metadata, top_n=max(top_tags, 10))

    # 5. complete missing Steam metadata from co-review neighbours (see enrich.py)
    tag_table = complete_game_tags(cfg, temp_path, app_ids, tag_table)
    temp_path.unlink(missing_ok=True)

    write_parquet(processed / GAMES_PARQUET, games)
    write_parquet(processed / GAME_REVIEWS_PARQUET, interactions)
    write_parquet(processed / GAME_TAGS_PARQUET, tag_table)

    stats = GameStats(
        n_items=len(games),
        n_users=int(users.size),
        n_interactions=len(interactions),
        n_positives=int(interactions["is_positive"].sum()),
        splits=split_sizes(interactions),
        files={
            "games": str(processed / GAMES_PARQUET),
            "reviews": str(processed / GAME_REVIEWS_PARQUET),
            "tags": str(processed / GAME_TAGS_PARQUET),
        },
    )
    logger.info(
        "games: %d items, %d users, %d interactions (%d positive)",
        stats.n_items,
        stats.n_users,
        stats.n_interactions,
        stats.n_positives,
    )
    return stats


def load_processed(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load ``games``, ``game_reviews`` and ``game_tags`` from ``data/processed``."""
    processed = cfg.paths.processed
    games = read_parquet(processed / GAMES_PARQUET)
    reviews = read_parquet(processed / GAME_REVIEWS_PARQUET)
    tags_path = processed / GAME_TAGS_PARQUET
    tags = read_parquet(tags_path) if tags_path.exists() else pd.DataFrame()
    return games, reviews, tags


__all__ = [
    "GAMES_CSV",
    "GAMES_PARQUET",
    "GAME_REVIEWS_PARQUET",
    "GAME_TAGS_PARQUET",
    "GAME_TEMP_PARQUET",
    "METADATA_JSON",
    "RECOMMENDATIONS_CSV",
    "USERS_CSV",
    "GameStats",
    "as_bool",
    "build_interactions",
    "build_tag_table",
    "extract_reviews",
    "load_games",
    "load_metadata",
    "load_processed",
    "parse_date_seconds",
    "parse_year",
    "review_counts",
    "run",
]
