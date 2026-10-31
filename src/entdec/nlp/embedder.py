"""Resumable embedding job (PROJECT_SPEC.md section 6).

* Documents are embedded chunk by chunk; an item vector is the L2-normalised
  mean of its chunk vectors.
* The job is **resumable**: every chunk is cached in shards under
  ``.cache/embeddings/<model>_<dim>/shard_XXXXX.parquet`` keyed by
  ``domain:item_id:chunk_index:text_hash``. Re-running continues where it
  stopped; ``--limit`` and ``--domain`` narrow the work, and changing a document
  changes its hash so only that chunk is re-embedded.
* Queries are embedded with the retrieval instruction prefix, documents without
  it (:func:`entdec.nlp.textdocs.query_text`).
* The output dimension is detected at runtime; ``embed_dim`` is requested through
  Ollama's ``dimensions`` parameter (verified to work for ``qwen3-embedding:4b``,
  which natively returns 2560) and falls back to truncation + re-normalisation.

Outputs: ``data/processed/<profile>/chunk_embeddings.parquet`` and
``data/processed/<profile>/item_embeddings.parquet``.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger
from entdec.nlp.ollama_client import OllamaClient, l2_normalize
from entdec.nlp.textdocs import ITEM_CHUNKS_PARQUET, ITEM_TEXTS_PARQUET

logger = get_logger(__name__)

CHUNK_EMBEDDINGS_PARQUET = "chunk_embeddings.parquet"
ITEM_EMBEDDINGS_PARQUET = "item_embeddings.parquet"
_CACHE_ROOT = "embeddings"
DEFAULT_SHARD_SIZE = 1024

CHUNK_EMBEDDING_COLUMNS = ("domain", "item_id", "chunk_index", "text_hash", "embedding")
ITEM_EMBEDDING_COLUMNS = ("domain", "item_id", "dim", "embedding")


@dataclass
class EmbedStats:
    """Outcome of one embedding run."""

    n_items: int = 0
    n_chunks: int = 0
    n_new_chunks: int = 0
    n_cached_chunks: int = 0
    dim: int = 0
    native_dim: int = 0
    duration_s: float = 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.n_cached_chunks / self.n_chunks if self.n_chunks else 0.0

    @property
    def docs_per_second(self) -> float:
        return self.n_new_chunks / self.duration_s if self.duration_s else 0.0


def text_hash(text: str) -> str:
    """Short stable hash of a chunk's text (cache key component)."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]  # noqa: S324 - cache key only


def cache_dir(cfg: Config, model: str, dim: int) -> Path:
    """Directory holding the embedding shards for a model/dimension pair."""
    slug = model.replace(":", "_").replace("/", "_")
    path = cfg.paths.cache / _CACHE_ROOT / f"{slug}_{dim}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def chunk_key(domain: str, item_id: int, chunk_index: int, digest: str) -> str:
    """Cache key of a single chunk."""
    return f"{domain}:{int(item_id)}:{int(chunk_index)}:{digest}"


class EmbeddingCache:
    """Shard based cache of chunk vectors."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._vectors: dict[str, np.ndarray] = {}
        self.n_loaded = 0

    @property
    def shards(self) -> list[Path]:
        return sorted(self.directory.glob("shard_*.parquet"))

    def load(self) -> EmbeddingCache:
        """Read every shard into memory."""
        for path in self.shards:
            try:
                frame = pd.read_parquet(path)
            except (OSError, ValueError) as exc:  # pragma: no cover - corrupt shard
                logger.warning("skipping unreadable shard %s (%s)", path.name, exc)
                continue
            for row in frame.itertuples(index=False):
                key = chunk_key(row.domain, row.item_id, row.chunk_index, row.text_hash)
                self._vectors.setdefault(key, np.asarray(row.embedding, dtype=np.float32))
        self.n_loaded = len(self._vectors)
        logger.info("embedding cache: %d chunks in %d shards", self.n_loaded, len(self.shards))
        return self

    def get(self, key: str) -> np.ndarray | None:
        return self._vectors.get(key)

    def put(self, key: str, vector: np.ndarray) -> None:
        self._vectors[key] = np.asarray(vector, dtype=np.float32)

    def save_shard(self, keys: list[str]) -> Path | None:
        """Write the vectors for ``keys`` into the next shard file."""
        rows = []
        for key in keys:
            vector = self._vectors.get(key)
            if vector is None:
                continue
            domain, item_id, chunk_index, digest = key.split(":")
            rows.append(
                {
                    "domain": domain,
                    "item_id": int(item_id),
                    "chunk_index": int(chunk_index),
                    "text_hash": digest,
                    "embedding": vector.astype(np.float32),
                }
            )
        if not rows:
            return None
        index = _next_shard_index(self.directory)
        path = self.directory / f"shard_{index:05d}.parquet"
        pd.DataFrame(rows, columns=list(CHUNK_EMBEDDING_COLUMNS)).to_parquet(path, index=False)
        logger.info("checkpoint: wrote %s (%d chunks)", path.name, len(rows))
        return path


def _next_shard_index(directory: Path) -> int:
    indices = []
    for path in directory.glob("shard_*.parquet"):
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):  # pragma: no cover - unexpected name
            continue
    return (max(indices) + 1) if indices else 0


def clear_cache(cfg: Config, model: str, dim: int) -> int:
    """Delete every shard for a model/dimension pair; returns the shard count."""
    directory = cache_dir(cfg, model, dim)
    shards = list(directory.glob("shard_*.parquet"))
    for path in shards:
        path.unlink(missing_ok=True)
    logger.info("cleared %d embedding shards in %s", len(shards), directory)
    return len(shards)


def select_chunks(
    chunks: pd.DataFrame,
    *,
    domain: str | None = None,
    limit_items: int | None = None,
) -> pd.DataFrame:
    """Apply the ``--domain`` / ``--limit`` filters to the chunk table."""
    selected = chunks
    if domain:
        selected = selected[selected["domain"] == domain]
    if limit_items:
        items = (
            selected[["domain", "item_id"]]
            .drop_duplicates()
            .sort_values(["domain", "item_id"])
            .head(int(limit_items))
        )
        selected = selected.merge(items, on=["domain", "item_id"], how="inner")
    return selected.reset_index(drop=True)


def aggregate_item_vectors(
    chunk_vectors: pd.DataFrame, chunk_index_col: str = "chunk_index"
) -> pd.DataFrame:
    """L2-normalised mean of the chunk vectors, per item."""
    if chunk_vectors.empty:
        return pd.DataFrame(columns=list(ITEM_EMBEDDING_COLUMNS))
    rows: list[dict[str, Any]] = []
    for (domain, item_id), group in chunk_vectors.groupby(["domain", "item_id"], sort=True):
        matrix = np.vstack(group.sort_values(chunk_index_col)["embedding"].to_numpy())
        mean = matrix.mean(axis=0)
        rows.append(
            {
                "domain": domain,
                "item_id": int(item_id),
                "dim": int(mean.shape[0]),
                "embedding": l2_normalize(mean.reshape(1, -1))[0],
            }
        )
    return pd.DataFrame(rows, columns=list(ITEM_EMBEDDING_COLUMNS))


def with_chunk_keys(chunks: pd.DataFrame) -> pd.DataFrame:
    """Add the ``text_hash`` and cache ``key`` columns to a chunk table."""
    frame = chunks.copy()
    frame["text_hash"] = frame["text"].map(text_hash)
    frame["key"] = [
        chunk_key(domain, item_id, index, digest)
        for domain, item_id, index, digest in zip(
            frame["domain"],
            frame["item_id"],
            frame["chunk_index"],
            frame["text_hash"],
            strict=False,
        )
    ]
    return frame


def cached_chunk_frame(cfg: Config, chunks: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Build the chunk-vector frame from the cache for **all** given chunks.

    Returns:
        ``(frame, n_missing)`` where ``n_missing`` counts chunks the cache does
        not know about yet.
    """
    if chunks.empty:
        return pd.DataFrame(columns=list(CHUNK_EMBEDDING_COLUMNS)), 0
    cache = EmbeddingCache(
        cache_dir(cfg, cfg.env.ollama_embed_model, int(cfg.get("nlp.embed_dim", 1024)))
    ).load()
    frame = with_chunk_keys(chunks)
    rows = []
    missing = 0
    for row in frame.itertuples(index=False):
        vector = cache.get(row.key)
        if vector is None:
            missing += 1
            continue
        rows.append(
            {
                "domain": row.domain,
                "item_id": int(row.item_id),
                "chunk_index": int(row.chunk_index),
                "text_hash": row.text_hash,
                "embedding": vector,
            }
        )
    return pd.DataFrame(rows, columns=list(CHUNK_EMBEDDING_COLUMNS)), missing


def embed_chunks(
    cfg: Config,
    chunks: pd.DataFrame,
    *,
    force: bool = False,
    client: OllamaClient | None = None,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> tuple[pd.DataFrame, EmbedStats]:
    """Embed every chunk, reusing the cache and checkpointing as it goes."""
    stats = EmbedStats()
    if chunks.empty:
        return pd.DataFrame(columns=list(CHUNK_EMBEDDING_COLUMNS)), stats

    embed_model = cfg.env.ollama_embed_model
    dim = int(cfg.get("nlp.embed_dim", 1024))
    if force:
        clear_cache(cfg, embed_model, dim)
    cache = EmbeddingCache(cache_dir(cfg, embed_model, dim)).load()

    chunks = with_chunk_keys(chunks)
    missing = [key for key in chunks["key"] if cache.get(key) is None]
    stats.n_chunks = len(chunks)
    stats.n_cached_chunks = stats.n_chunks - len(missing)
    logger.info(
        "%d chunks to embed, %d already cached (%.0f%%)",
        len(missing),
        stats.n_cached_chunks,
        100 * stats.cache_hit_rate,
    )

    if missing:
        ollama = client if client is not None else OllamaClient.from_config(cfg)
        by_key = dict(zip(chunks["key"], chunks["text"], strict=False))
        batch_size = int(cfg.get("nlp.embed_batch_size", 64))
        started = time.perf_counter()
        for start in range(0, len(missing), shard_size):
            shard_keys = missing[start : start + shard_size]
            texts = [by_key[key] for key in shard_keys]
            vectors = ollama.embed_batch(texts, batch_size=batch_size, dimensions=dim)
            for key, vector in zip(shard_keys, vectors, strict=False):
                cache.put(key, vector)
            cache.save_shard(shard_keys)
        stats.duration_s = time.perf_counter() - started
        stats.n_new_chunks = len(missing)
        if vectors.size:
            stats.dim = int(vectors.shape[1])
        stats.native_dim = stats.dim or dim

    rows = []
    for row in chunks.itertuples(index=False):
        vector = cache.get(row.key)
        if vector is None:
            continue
        rows.append(
            {
                "domain": row.domain,
                "item_id": int(row.item_id),
                "chunk_index": int(row.chunk_index),
                "text_hash": row.text_hash,
                "embedding": vector,
            }
        )
    chunk_frame = pd.DataFrame(rows, columns=list(CHUNK_EMBEDDING_COLUMNS))
    if not rows:
        return chunk_frame, stats
    stats.dim = stats.dim or int(len(chunk_frame.iloc[0]["embedding"]))
    return chunk_frame, stats


def run(cfg: Config, args: Any = None, client: OllamaClient | None = None) -> int:
    """The ``embed`` step: embed chunks, aggregate per item, write Parquet."""
    processed = cfg.paths.processed
    chunks_path = processed / ITEM_CHUNKS_PARQUET
    if not chunks_path.exists():
        logger.error("no chunks found - run `entdec texts` first")
        return 1

    chunks = pd.read_parquet(chunks_path)
    selected = select_chunks(
        chunks,
        domain=getattr(args, "domain", None),
        limit_items=getattr(args, "limit", None),
    )
    logger.info(
        "embedding %d chunks across %d items",
        len(selected),
        len(selected[["domain", "item_id"]].drop_duplicates()),
    )

    _, stats = embed_chunks(cfg, selected, force=bool(getattr(args, "force", False)), client=client)

    # The output always reflects every chunk the cache knows about, so a partial
    # run (--limit / --domain) never silently shrinks the artefact.
    chunk_vectors, n_missing = cached_chunk_frame(cfg, chunks)
    if n_missing:
        logger.warning(
            "%d of %d chunks are not embedded yet - run `entdec embed` without --limit",
            n_missing,
            len(chunks),
        )
    if chunk_vectors.empty:
        logger.error("no chunk vectors produced")
        return 1

    item_vectors = aggregate_item_vectors(chunk_vectors)
    stats.n_items = len(item_vectors)

    chunk_vectors.to_parquet(processed / CHUNK_EMBEDDINGS_PARQUET, index=False)
    item_vectors.to_parquet(processed / ITEM_EMBEDDINGS_PARQUET, index=False)

    texts_path = processed / ITEM_TEXTS_PARQUET
    texts = pd.read_parquet(texts_path) if texts_path.exists() else pd.DataFrame()
    stats.n_chunks = len(chunk_vectors)
    logger.info(
        "embeddings: %d items, %d chunks (dim %d), %d newly embedded, %.1f chunks/s",
        stats.n_items,
        stats.n_chunks,
        stats.dim,
        stats.n_new_chunks,
        stats.docs_per_second,
    )
    cfg.report_path("embeddings.md").write_text(
        "\n".join(
            [
                "# Embeddings",
                "",
                f"Generated by `python -m entdec embed --profile {cfg.profile.name}`.",
                "",
                "| metric | value |",
                "|--------|-------|",
                f"| embed model | `{cfg.env.ollama_embed_model}` |",
                f"| items embedded | {stats.n_items:,} |",
                f"| chunks embedded | {stats.n_chunks:,} |",
                f"| chunks newly embedded this run | {stats.n_new_chunks:,} |",
                f"| chunks re-used from cache | {stats.n_cached_chunks:,} "
                f"({100 * stats.cache_hit_rate:.1f}%) |",
                f"| output dimension | {stats.dim} |",
                f"| document tokens (approx) | "
                f"{int(texts['n_tokens'].sum()) if not texts.empty else 0:,} |",
                f"| throughput (new chunks) | {stats.docs_per_second:.1f} chunks/s |",
                "",
                "Item vectors are the L2-normalised mean of their chunk vectors.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


__all__ = [
    "CHUNK_EMBEDDINGS_PARQUET",
    "ITEM_EMBEDDINGS_PARQUET",
    "EmbedStats",
    "EmbeddingCache",
    "aggregate_item_vectors",
    "cache_dir",
    "chunk_key",
    "clear_cache",
    "embed_chunks",
    "run",
    "select_chunks",
    "text_hash",
]
