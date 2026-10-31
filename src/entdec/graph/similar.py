"""Offline SIMILAR_TO computation (PROJECT_SPEC.md section 5).

The top-10 most similar items per item by embedding cosine, **within and across
domains**, computed in blocks with numpy so the full item x item matrix is never
materialised. Embeddings are L2-normalised, so cosine is a plain dot product.

Results are written to ``data/processed/<profile>/similar_to.parquet`` and are
then loaded as ``:SIMILAR_TO {score}`` edges.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger
from entdec.nlp.ollama_client import l2_normalize

logger = get_logger(__name__)

SIMILAR_TO_PARQUET = "similar_to.parquet"
SIMILAR_COLUMNS = ("src_domain", "src_id", "dst_domain", "dst_id", "score", "rank")
DEFAULT_BLOCK_SIZE = 2048
DEFAULT_TOP_K = 10


@dataclass
class SimilarityStats:
    """What the similarity pass produced."""

    n_items: int = 0
    n_edges: int = 0
    cross_domain_edges: int = 0
    mean_top_score: float = 0.0
    duration_s: float = 0.0


def load_item_embeddings(cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the item embedding table.

    Returns:
        ``(items, matrix)`` where ``items`` has ``domain``/``item_id`` columns in
        matrix row order.
    """
    path = cfg.paths.processed / "item_embeddings.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `entdec embed` first")
    frame = pd.read_parquet(path, columns=["domain", "item_id", "embedding"])
    frame = frame.sort_values(["domain", "item_id"]).reset_index(drop=True)
    matrix = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
    if matrix.ndim != 2:  # pragma: no cover - defensive
        raise ValueError("item embeddings are not a 2D matrix")
    return frame, l2_normalize(matrix)


def top_k_similar(
    matrix: np.ndarray,
    k: int = DEFAULT_TOP_K,
    *,
    block_size: int = DEFAULT_BLOCK_SIZE,
    min_score: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-``k`` neighbours per row, in blocks.

    Args:
        matrix: ``(n, d)`` L2-normalised embeddings.
        k: Neighbours per item.
        block_size: Rows scored per block; the score block is
            ``block_size x n`` floats, which is what bounds memory.
        min_score: Drop neighbours below this cosine.

    Returns:
        ``(row_index, neighbour_index, score)`` arrays, flattened.
    """
    n_items = matrix.shape[0]
    if n_items == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), np.zeros(0, dtype=np.float32)
    k = max(1, min(int(k), n_items - 1)) if n_items > 1 else 1

    rows_out: list[np.ndarray] = []
    cols_out: list[np.ndarray] = []
    scores_out: list[np.ndarray] = []

    for start in range(0, n_items, block_size):
        stop = min(start + block_size, n_items)
        block = matrix[start:stop]
        scores = block @ matrix.T  # (block, n)
        # Never recommend the item itself.
        local = np.arange(stop - start)
        scores[local, np.arange(start, stop)] = -np.inf
        if min_score > 0:
            scores[scores < min_score] = -np.inf

        take = min(k, scores.shape[1])
        # argpartition gives the top-k cheaply, then sort those k.
        partitioned = np.argpartition(-scores, take - 1, axis=1)[:, :take]
        partitioned_scores = np.take_along_axis(scores, partitioned, axis=1)
        order = np.argsort(-partitioned_scores, axis=1)
        partitioned = np.take_along_axis(partitioned, order, axis=1)
        partitioned_scores = np.take_along_axis(partitioned_scores, order, axis=1)

        valid = np.isfinite(partitioned_scores)
        row_idx, col_in_block = np.nonzero(valid)
        rows_out.append(row_idx + start)
        cols_out.append(partitioned[row_idx, col_in_block])
        scores_out.append(partitioned_scores[row_idx, col_in_block])

    return (
        np.concatenate(rows_out) if rows_out else np.zeros(0, dtype=int),
        np.concatenate(cols_out) if cols_out else np.zeros(0, dtype=int),
        np.concatenate(scores_out) if scores_out else np.zeros(0, dtype=np.float32),
    )


def build_similar_to_frame(
    items: pd.DataFrame,
    matrix: np.ndarray,
    *,
    k: int = DEFAULT_TOP_K,
    block_size: int = DEFAULT_BLOCK_SIZE,
    min_score: float = 0.0,
) -> tuple[pd.DataFrame, SimilarityStats]:
    """Run :func:`top_k_similar` and turn the indices into a labelled frame."""
    started = time.perf_counter()
    rows, cols, scores = top_k_similar(
        matrix, k=k, block_size=block_size, min_score=min_score
    )
    stats = SimilarityStats(n_items=len(items), duration_s=time.perf_counter() - started)
    if rows.size == 0:
        return pd.DataFrame(columns=list(SIMILAR_COLUMNS)), stats

    domains = items["domain"].to_numpy()
    ids = items["item_id"].to_numpy()
    frame = pd.DataFrame(
        {
            "src_domain": domains[rows],
            "src_id": ids[rows],
            "dst_domain": domains[cols],
            "dst_id": ids[cols],
            "score": scores.astype(np.float32),
        }
    )
    frame["rank"] = frame.groupby(["src_domain", "src_id"], sort=False).cumcount() + 1
    frame = frame.sort_values(
        ["src_domain", "src_id", "rank"], ignore_index=True
    ).astype({"rank": "int16", "score": "float32"})

    stats.n_edges = len(frame)
    stats.cross_domain_edges = int((frame["src_domain"] != frame["dst_domain"]).sum())
    stats.mean_top_score = float(frame.loc[frame["rank"] == 1, "score"].mean())
    return frame[list(SIMILAR_COLUMNS)], stats


def compute_similar_to(
    cfg: Config,
    *,
    k: int | None = None,
    block_size: int | None = None,
    min_score: float | None = None,
    force: bool = False,
) -> tuple[Path, SimilarityStats]:
    """Compute (or reuse) ``similar_to.parquet``."""
    path = cfg.paths.processed / SIMILAR_TO_PARQUET
    if path.exists() and not force:
        frame = pd.read_parquet(path)
        stats = SimilarityStats(
            n_edges=len(frame),
            cross_domain_edges=int((frame["src_domain"] != frame["dst_domain"]).sum()),
            mean_top_score=float(frame.loc[frame["rank"] == 1, "score"].mean()),
        )
        logger.info("reusing %s (%d edges)", path.name, stats.n_edges)
        return path, stats

    items, matrix = load_item_embeddings(cfg)
    frame, stats = build_similar_to_frame(
        items,
        matrix,
        k=int(k or cfg.get("graph.similar_top_k", DEFAULT_TOP_K)),
        block_size=int(block_size or cfg.get("graph.similar_block_size", DEFAULT_BLOCK_SIZE)),
        min_score=float(
            min_score if min_score is not None else cfg.get("graph.similar_min_score", 0.0)
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    logger.info(
        "similar_to: %d items, %d edges (%d cross-domain), mean top score %.3f, %.1fs",
        stats.n_items,
        stats.n_edges,
        stats.cross_domain_edges,
        stats.mean_top_score,
        stats.duration_s,
    )
    return path, stats


__all__ = [
    "DEFAULT_BLOCK_SIZE",
    "DEFAULT_TOP_K",
    "SIMILAR_COLUMNS",
    "SIMILAR_TO_PARQUET",
    "SimilarityStats",
    "build_similar_to_frame",
    "compute_similar_to",
    "load_item_embeddings",
    "top_k_similar",
]
