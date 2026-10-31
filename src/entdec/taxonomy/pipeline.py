"""The ``taxonomy`` step: canonical genres, clustered themes, theme_map.csv.

Outputs (profile-specific outputs in ``data/processed/<profile>``):

* ``data/processed/theme_map.csv`` - shared ``source, raw_tag, theme`` map (hand-editable)
* ``movie_themes.parquet``     - ``movie_id, theme, weight`` (top themes only)
* ``game_themes.parquet``      - ``app_id, theme, weight``
* ``item_genres.parquet``      - ``domain, item_id, genre``

A report with the theme catalogue and the unmapped raw genres is written to
``reports/<profile>/taxonomy.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger
from entdec.taxonomy import genres as genre_mod
from entdec.taxonomy import themes as theme_mod

logger = get_logger(__name__)

TAXONOMY_MD = "taxonomy.md"


@dataclass
class TaxonomyStats:
    """What the taxonomy step produced."""

    n_themes: int = 0
    n_raw_tags: int = 0
    movie_theme_rows: int = 0
    game_theme_rows: int = 0
    genre_rows: int = 0
    unmapped_genres: dict[str, int] = field(default_factory=dict)
    source: str = ""


# --------------------------------------------------------------------------- #
# theme weights per item
# --------------------------------------------------------------------------- #
def _aggregate_theme_weights(
    frame: pd.DataFrame,
    id_col: str,
    lookup: dict[str, str],
    *,
    top_n: int,
) -> pd.DataFrame:
    """Aggregate tag weights per item, keep the ``top_n`` themes, normalise to 1."""
    if frame.empty:
        return pd.DataFrame(columns=[id_col, "theme", "weight"])

    working = frame.copy()
    working["theme"] = working["tag"].map(lambda raw: lookup.get(theme_mod.normalise_tag(raw)))
    working = working[working["theme"].notna()]
    if working.empty:
        return pd.DataFrame(columns=[id_col, "theme", "weight"])

    aggregated = working.groupby([id_col, "theme"], sort=False)["weight"].sum().reset_index()
    aggregated = aggregated.sort_values(
        [id_col, "weight", "theme"], ascending=[True, False, True], kind="stable"
    )
    trimmed = aggregated.groupby(id_col, sort=False).head(top_n).reset_index(drop=True)
    maximum = trimmed.groupby(id_col, sort=False)["weight"].transform("max")
    trimmed["weight"] = (trimmed["weight"] / maximum.replace(0, 1.0)).astype("float32")
    return trimmed.sort_values(
        [id_col, "weight", "theme"], ascending=[True, False, True]
    ).reset_index(drop=True)


def build_movie_themes(cfg: Config, lookup: dict[str, str]) -> pd.DataFrame:
    """Movie themes from genome relevance aggregated per theme."""
    path = cfg.paths.processed / "movie_genome.parquet"
    if not path.exists():
        logger.warning("movie_genome.parquet missing - no movie themes")
        return pd.DataFrame(columns=["movie_id", "theme", "weight"])
    genome = pd.read_parquet(path, columns=["movie_id", "tag", "relevance"]).rename(
        columns={"relevance": "weight"}
    )
    top_n = int(cfg.get("taxonomy.movie_top_themes", 8))
    themes = _aggregate_theme_weights(genome, "movie_id", lookup, top_n=top_n)
    logger.info("movie themes: %d rows for %d movies", len(themes), themes["movie_id"].nunique())
    return themes


def build_game_themes(cfg: Config, lookup: dict[str, str]) -> pd.DataFrame:
    """Game themes from the (rank weighted) Steam tags aggregated per theme."""
    path = cfg.paths.processed / "game_tags.parquet"
    if not path.exists():
        logger.warning("game_tags.parquet missing - no game themes")
        return pd.DataFrame(columns=["app_id", "theme", "weight"])
    tags = pd.read_parquet(path, columns=["app_id", "tag", "weight"])
    top_n = int(cfg.get("taxonomy.game_top_themes", 10))
    themes = _aggregate_theme_weights(tags, "app_id", lookup, top_n=top_n)
    logger.info("game themes: %d rows for %d games", len(themes), themes["app_id"].nunique())
    return themes


# --------------------------------------------------------------------------- #
# genres
# --------------------------------------------------------------------------- #
def build_item_genres(cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Canonical genre rows for every item of both domains.

    Returns:
        ``(frame, unmapped)`` where ``unmapped`` counts raw genres that the
        hand-written map does not cover (audited in the report).
    """
    processed = cfg.paths.processed
    rows: list[pd.DataFrame] = []
    unmapped: dict[str, int] = {}

    movies_path = processed / "movies.parquet"
    if movies_path.exists():
        movies = pd.read_parquet(movies_path, columns=["movie_id", "genres"])
        records: list[dict[str, Any]] = []
        for movie_id, raw_genres in zip(movies["movie_id"], movies["genres"], strict=False):
            raw_list = list(raw_genres) if raw_genres is not None else []
            for name in genre_mod.unmapped_genres(raw_list):
                unmapped[name] = unmapped.get(name, 0) + 1
            for canonical in genre_mod.map_genres(raw_list):
                records.append({"domain": "movie", "item_id": int(movie_id), "genre": canonical})
        rows.append(pd.DataFrame(records))

    games_path = processed / "games.parquet"
    tags_path = processed / "game_tags.parquet"
    if games_path.exists() and tags_path.exists():
        games = pd.read_parquet(games_path, columns=["app_id"])
        tags = pd.read_parquet(tags_path, columns=["app_id", "tag", "rank"])
        tags = tags[tags["app_id"].isin(set(games["app_id"].tolist()))]
        records = []
        for app_id, group in tags.groupby("app_id", sort=False):
            raw_list = group.sort_values("rank")["tag"].astype(str).tolist()
            for name in genre_mod.unmapped_genres(raw_list):
                unmapped[name] = unmapped.get(name, 0) + 1
            for canonical in genre_mod.map_genres(raw_list):
                records.append({"domain": "game", "item_id": int(app_id), "genre": canonical})
        rows.append(pd.DataFrame(records))

    frame = pd.concat([item for item in rows if not item.empty], ignore_index=True)
    frame = (
        frame.drop_duplicates(["domain", "item_id", "genre"])
        .sort_values(["domain", "item_id", "genre"])
        .reset_index(drop=True)
    )
    logger.info(
        "genres: %d rows (%d movie items, %d game items)",
        len(frame),
        frame.loc[frame["domain"] == "movie", "item_id"].nunique(),
        frame.loc[frame["domain"] == "game", "item_id"].nunique(),
    )
    if unmapped:
        logger.info(
            "unmapped raw genres (top 10): %s",
            sorted(unmapped.items(), key=lambda item: -item[1])[:10],
        )
    return frame, unmapped


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def build_report(
    cfg: Config,
    build: theme_mod.ThemeBuildResult,
    movie_themes: pd.DataFrame,
    game_themes: pd.DataFrame,
    item_genres: pd.DataFrame,
    unmapped: dict[str, int],
) -> str:
    """Render the active profile's taxonomy report."""
    theme_map = build.theme_map
    sizes = theme_map.groupby("theme").size().sort_values(ascending=False)
    members = {
        theme: ", ".join(group["raw_tag"].head(8).tolist())
        for theme, group in theme_map.groupby("theme")
    }
    lines = [
        "# Taxonomy",
        "",
        f"Generated by `python -m entdec taxonomy --profile {cfg.profile.name}`.",
        "",
        "## Summary",
        "",
        f"- theme source: `{build.source}`",
        f"- raw tag rows in `theme_map.csv`: {build.n_raw_tags:,}",
        f"- themes: {build.n_themes}",
        f"- tags clustered: {build.n_clustered:,}; assigned to the nearest clustered tag: "
        f"{build.n_assigned_by_similarity:,}",
        f"- items with movie themes: {movie_themes['movie_id'].nunique() if not movie_themes.empty else 0:,}",
        f"- items with game themes: {game_themes['app_id'].nunique() if not game_themes.empty else 0:,}",
        f"- canonical genres: {len(genre_mod.CANONICAL_GENRES)}",
        f"- genre rows: {len(item_genres):,}",
        "",
        "Hand-editing `data/processed/theme_map.csv` is supported: the mapping is reused",
        "as-is unless `--force` is passed, which re-runs the clustering.",
        "",
        "## Lens: item coverage",
        "",
        "| domain | items with genres | items with themes |",
        "|--------|-------------------|-------------------|",
        f"| movies | {item_genres.loc[item_genres['domain'] == 'movie', 'item_id'].nunique() if not item_genres.empty else 0:,} "
        f"| {movie_themes['movie_id'].nunique() if not movie_themes.empty else 0:,} |",
        f"| games | {item_genres.loc[item_genres['domain'] == 'game', 'item_id'].nunique() if not item_genres.empty else 0:,} "
        f"| {game_themes['app_id'].nunique() if not game_themes.empty else 0:,} |",
        "",
        "## Themes",
        "",
        "| theme | tags | example member tags |",
        "|-------|------|---------------------|",
        *(f"| {theme} | {int(sizes[theme])} | {members.get(theme, '')} |" for theme in sizes.index),
        "",
    ]
    if unmapped:
        top_unmapped = sorted(unmapped.items(), key=lambda item: -item[1])[:25]
        lines += [
            "## Raw genres without a canonical mapping",
            "",
            "| raw genre | occurrences |",
            "|-----------|-------------|",
            *(f"| {name} | {count:,} |" for name, count in top_unmapped),
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def run(cfg: Config, args: Any = None, client: Any = None) -> int:
    """Run the taxonomy step.

    Args:
        cfg: Application config.
        args: Parsed CLI arguments (``force``, ``themes``, ``limit``).
        client: Optional Ollama client override (used by tests).
    """
    force = bool(getattr(args, "force", False))
    n_themes = getattr(args, "themes", None)
    limit_tags = getattr(args, "limit", None)

    build = theme_mod.build_theme_map(
        cfg, force=force, n_themes=n_themes, limit_tags=limit_tags, client=client
    )
    lookup = theme_mod.theme_lookup(build.theme_map)
    logger.info("theme lookup built for %d normalised tags", len(lookup))

    movie_themes = build_movie_themes(cfg, lookup)
    game_themes = build_game_themes(cfg, lookup)
    item_genres, unmapped = build_item_genres(cfg)

    processed = cfg.paths.processed
    movie_themes.to_parquet(processed / theme_mod.MOVIE_THEMES_PARQUET, index=False)
    game_themes.to_parquet(processed / theme_mod.GAME_THEMES_PARQUET, index=False)
    item_genres.to_parquet(processed / theme_mod.ITEM_GENRES_PARQUET, index=False)

    report = build_report(cfg, build, movie_themes, game_themes, item_genres, unmapped)
    cfg.report_path(TAXONOMY_MD).write_text(report, encoding="utf-8")
    logger.info(
        "taxonomy done: %d themes, %d movie theme rows, %d game theme rows, %d genre rows",
        build.n_themes,
        len(movie_themes),
        len(game_themes),
        len(item_genres),
    )

    if movie_themes.empty or game_themes.empty:
        logger.error("taxonomy produced no themes for at least one domain")
        return 1
    return 0


__all__ = [
    "TAXONOMY_MD",
    "TaxonomyStats",
    "build_game_themes",
    "build_item_genres",
    "build_movie_themes",
    "build_report",
    "run",
]
