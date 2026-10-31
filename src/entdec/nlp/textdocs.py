"""Item documents for embedding (PROJECT_SPEC.md section 6).

Formats:

* movie: ``"{title} ({year}). Genres: {genres}. Themes: {top 15 genome tags by
  relevance}. User tags: {top 10 frequent}."``
* game: ``"{title} ({year}). Tags: {top 15 tags}. {description}"``

Local Ollama models take no instruction prefix for *documents*; only queries are
prefixed (see :func:`query_text`). Outputs:

* ``data/processed/<profile>/item_texts.parquet``  - one row per item
* ``data/processed/<profile>/item_chunks.parquet`` - one row per chunk
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger
from entdec.nlp.chunker import (
    CHUNK_MAX_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    WORDS_TO_TOKENS,
    chunk_text,
    estimate_tokens,
)

logger = get_logger(__name__)

ITEM_TEXTS_PARQUET = "item_texts.parquet"
ITEM_CHUNKS_PARQUET = "item_chunks.parquet"

ITEM_TEXT_COLUMNS = ("domain", "item_id", "text", "n_tokens", "n_chunks")
ITEM_CHUNK_COLUMNS = ("domain", "item_id", "chunk_index", "text", "n_tokens")


@dataclass
class TextStats:
    """What the ``texts`` step produced."""

    n_items: int = 0
    n_chunks: int = 0
    movie_items: int = 0
    game_items: int = 0
    mean_chunks: float = 0.0
    max_chunks: int = 0
    mean_tokens: float = 0.0
    description_chars: int = 0


def _year_suffix(year: object) -> str:
    try:
        value = int(year)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    return f" ({value})" if value > 0 else ""


def build_movie_document(
    title: str,
    year: object,
    genres: list[str] | None,
    genome_tags: list[str] | None,
    user_tags: list[str] | None,
) -> str:
    """Build the movie document exactly as specified in section 6."""
    parts = [f"{title}{_year_suffix(year)}."]
    if genres:
        parts.append(f"Genres: {', '.join(genres)}.")
    if genome_tags:
        parts.append(f"Themes: {', '.join(genome_tags)}.")
    if user_tags:
        parts.append(f"User tags: {', '.join(user_tags)}.")
    return " ".join(parts).strip()


def build_game_document(
    title: str,
    year: object,
    tags: list[str] | None,
    description: str | None,
) -> str:
    """Build the game document exactly as specified in section 6."""
    parts = [f"{title}{_year_suffix(year)}."]
    if tags:
        parts.append(f"Tags: {', '.join(tags)}.")
    text = " ".join(parts)
    cleaned = (description or "").strip()
    if cleaned:
        text = f"{text} {cleaned}"
    return text.strip()


def query_text(query: str, instruction: str) -> str:
    """Prefix a user query with the retrieval instruction (queries only)."""
    return f"{instruction}{query}" if instruction else query


# --------------------------------------------------------------------------- #
# document builders over the processed tables
# --------------------------------------------------------------------------- #
def _top_n_by(frame: pd.DataFrame, value_col: str, n: int) -> dict[int, list[str]]:
    """Most valuable tags per item id, sorted by ``value_col`` descending."""
    if frame.empty:
        return {}
    ordered = frame.sort_values(
        ["movie_id" if "movie_id" in frame.columns else "app_id", value_col],
        ascending=[True, False],
        kind="stable",
    )
    key = "movie_id" if "movie_id" in frame.columns else "app_id"
    result: dict[int, list[str]] = {}
    for item_id, group in ordered.groupby(key, sort=False):
        result[int(item_id)] = group["tag"].astype(str).head(n).tolist()
    return result


def build_movie_documents(cfg: Config) -> list[dict[str, Any]]:
    """Build one document per sampled movie."""
    processed = cfg.paths.processed
    movies = pd.read_parquet(processed / "movies.parquet")

    genome_path = processed / "movie_genome.parquet"
    genome = (
        pd.read_parquet(genome_path, columns=["movie_id", "tag", "relevance"])
        if genome_path.exists()
        else pd.DataFrame(columns=["movie_id", "tag", "relevance"])
    )
    tags_path = processed / "movie_user_tags.parquet"
    user_tags = (
        pd.read_parquet(tags_path, columns=["movie_id", "tag", "count"])
        if tags_path.exists()
        else pd.DataFrame(columns=["movie_id", "tag", "count"])
    )

    n_genome = int(cfg.get("nlp.movie_top_text_tags", 15))
    n_user = int(cfg.get("nlp.movie_top_user_tags", 10))
    genome_by_movie = _top_n_by(genome, "relevance", n_genome)
    user_by_movie = _top_n_by(user_tags, "count", n_user)

    documents: list[dict[str, Any]] = []
    for row in movies.itertuples(index=False):
        genres = [str(value) for value in (row.genres if row.genres is not None else [])]
        text = build_movie_document(
            str(row.title),
            row.year,
            genres,
            genome_by_movie.get(int(row.movie_id), []),
            user_by_movie.get(int(row.movie_id), []),
        )
        documents.append({"domain": "movie", "item_id": int(row.movie_id), "text": text})
    logger.info("built %d movie documents", len(documents))
    return documents


def build_game_documents(cfg: Config) -> list[dict[str, Any]]:
    """Build one document per sampled game."""
    processed = cfg.paths.processed
    games = pd.read_parquet(processed / "games.parquet")

    tags_path = processed / "game_tags.parquet"
    tags = (
        pd.read_parquet(tags_path, columns=["app_id", "tag", "rank"])
        if tags_path.exists()
        else pd.DataFrame(columns=["app_id", "tag", "rank"])
    )
    n_tags = int(cfg.get("nlp.game_top_text_tags", 15))
    max_chars = int(cfg.get("nlp.max_description_chars", 1200))

    ordered = tags.sort_values(["app_id", "rank"], kind="stable") if not tags.empty else tags
    tags_by_game: dict[int, list[str]] = {}
    if not ordered.empty:
        for app_id, group in ordered.groupby("app_id", sort=False):
            tags_by_game[int(app_id)] = group["tag"].astype(str).head(n_tags).tolist()

    descriptions = (
        games["description"].fillna("").astype(str).tolist()
        if "description" in games
        else [""] * len(games)
    )
    if max_chars:
        descriptions = [text[:max_chars] for text in descriptions]

    documents: list[dict[str, Any]] = []
    for row, description in zip(games.itertuples(index=False), descriptions, strict=False):
        text = build_game_document(
            str(row.title), row.year, tags_by_game.get(int(row.app_id), []), description
        )
        documents.append({"domain": "game", "item_id": int(row.app_id), "text": text})
    logger.info("built %d game documents", len(documents))
    return documents


def documents_to_frames(
    documents: list[dict[str, Any]], cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, TextStats]:
    """Chunk the documents and turn them into the two output frames."""
    max_tokens = int(cfg.get("nlp.chunk_max_tokens", CHUNK_MAX_TOKENS))
    target = int(cfg.get("nlp.chunk_target_tokens", CHUNK_TARGET_TOKENS))
    overlap = int(cfg.get("nlp.chunk_overlap_tokens", CHUNK_OVERLAP_TOKENS))
    ratio = float(cfg.get("nlp.words_to_tokens", WORDS_TO_TOKENS))

    item_rows: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    for document in documents:
        chunks = chunk_text(
            document["text"],
            max_tokens=max_tokens,
            target_tokens=target,
            overlap_tokens=overlap,
            words_to_tokens=ratio,
        )
        item_rows.append(
            {
                "domain": document["domain"],
                "item_id": int(document["item_id"]),
                "text": document["text"],
                "n_tokens": estimate_tokens(document["text"], ratio),
                "n_chunks": len(chunks),
            }
        )
        for chunk in chunks:
            chunk_rows.append(
                {
                    "domain": document["domain"],
                    "item_id": int(document["item_id"]),
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "n_tokens": chunk.tokens,
                }
            )

    texts = pd.DataFrame(item_rows, columns=list(ITEM_TEXT_COLUMNS))
    chunk_frame = pd.DataFrame(chunk_rows, columns=list(ITEM_CHUNK_COLUMNS))

    stats = TextStats(
        n_items=len(texts),
        n_chunks=len(chunk_frame),
        movie_items=int((texts["domain"] == "movie").sum()),
        game_items=int((texts["domain"] == "game").sum()),
        mean_chunks=float(texts["n_chunks"].mean()) if len(texts) else 0.0,
        max_chunks=int(texts["n_chunks"].max()) if len(texts) else 0,
        mean_tokens=float(texts["n_tokens"].mean()) if len(texts) else 0.0,
    )
    return texts, chunk_frame, stats


def run(cfg: Config, args: Any = None) -> int:
    """The ``texts`` step: build documents and chunks and write them to Parquet."""
    force = bool(getattr(args, "force", False))
    processed = cfg.paths.processed
    texts_path = processed / ITEM_TEXTS_PARQUET

    if texts_path.exists() and not force:
        existing = pd.read_parquet(texts_path, columns=["domain"])
        logger.info("item texts already exist (%d items) - use --force to rebuild", len(existing))
        return 0

    documents = build_movie_documents(cfg) + build_game_documents(cfg)
    if not documents:
        logger.error("no documents built - run `entdec data` first")
        return 1

    texts, chunks, stats = documents_to_frames(documents, cfg)
    texts.to_parquet(texts_path, index=False)
    chunks.to_parquet(processed / ITEM_CHUNKS_PARQUET, index=False)

    stats.description_chars = int(
        pd.read_parquet(processed / "games.parquet", columns=["description"])["description"]
        .fillna("")
        .astype(str)
        .str.len()
        .sum()
        if (processed / "games.parquet").exists()
        else 0
    )
    logger.info(
        "texts: %d items (%d movies, %d games), %d chunks, mean %.2f chunks/item, "
        "mean %d tokens/item",
        stats.n_items,
        stats.movie_items,
        stats.game_items,
        stats.n_chunks,
        stats.mean_chunks,
        int(stats.mean_tokens),
    )
    cfg.report_path("texts.md").write_text(
        "\n".join(
            [
                "# Item documents",
                "",
                f"Generated by `python -m entdec texts --profile {cfg.profile.name}`.",
                "",
                "| metric | value |",
                "|--------|-------|",
                f"| items | {stats.n_items:,} |",
                f"| movies | {stats.movie_items:,} |",
                f"| games | {stats.game_items:,} |",
                f"| chunks | {stats.n_chunks:,} |",
                f"| mean chunks / item | {stats.mean_chunks:.2f} |",
                f"| max chunks / item | {stats.max_chunks} |",
                f"| mean tokens / item | {int(stats.mean_tokens):,} |",
                f"| chunk max / target / overlap (tokens) | {cfg.get('nlp.chunk_max_tokens')} / "
                f"{cfg.get('nlp.chunk_target_tokens')} / {cfg.get('nlp.chunk_overlap_tokens')} |",
                f"| game description truncation (characters) | {cfg.get('nlp.max_description_chars')} |",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


__all__ = [
    "ITEM_CHUNK_COLUMNS",
    "ITEM_CHUNKS_PARQUET",
    "ITEM_TEXT_COLUMNS",
    "ITEM_TEXTS_PARQUET",
    "TextStats",
    "build_game_document",
    "build_game_documents",
    "build_movie_document",
    "build_movie_documents",
    "documents_to_frames",
    "query_text",
    "run",
]
