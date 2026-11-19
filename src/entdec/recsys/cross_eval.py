"""Seed-based proxy evaluation for cross-domain recommendations.

The evaluation uses the deterministic scorer shared with ``entdec cross``;
MovieLens and Steam do not share user identities, so recommendation quality is
measured with theme/embedding proxies and a small manually curated gold set.
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger
from entdec.nlp.chunker import chunk_text
from entdec.nlp.ollama_client import OllamaClient, OllamaError, l2_normalize
from entdec.recsys.common import DomainData, load_domain
from entdec.recsys.cross_domain import CrossDomainScorer
from entdec.recsys.popularity import popularity_scores

logger = get_logger(__name__)

GOLD_COLUMNS = [
    "movie_id",
    "game_app_id",
    "movie_title",
    "game_title",
    "relation_type",
    "confidence",
    "note",
    "reviewed",
]
METHODS = (
    "random",
    "popularity",
    "genre_only",
    "embedding_only",
    "theme_only",
    "full_system",
)
SEED_METRICS = (
    "theme_jaccard",
    "theme_cosine",
    "shared_themes_per_result",
    "non_generic_share",
    "embedding_cosine_mean",
    "embedding_cosine_rank1",
)


def theme_idf(
    theme_vectors: Sequence[Mapping[str, float]], item_count: int | None = None
) -> dict[str, float]:
    """Compute ``log((N+1)/(df+1))+1`` across every item in both catalogues."""
    n_items = len(theme_vectors) if item_count is None else int(item_count)
    document_frequency: Counter[str] = Counter()
    for vector in theme_vectors:
        document_frequency.update(set(vector))
    return {
        theme: float(np.log((n_items + 1) / (count + 1)) + 1.0)
        for theme, count in document_frequency.items()
    }


def weighted_jaccard(
    left: Mapping[str, float], right: Mapping[str, float], idf: Mapping[str, float]
) -> float:
    """IDF-weighted Jaccard similarity; empty or disjoint vectors score zero."""
    if not left or not right:
        return 0.0
    themes = left.keys() | right.keys()
    numerator = sum(min(left.get(t, 0.0), right.get(t, 0.0)) * idf.get(t, 0.0) for t in themes)
    denominator = sum(max(left.get(t, 0.0), right.get(t, 0.0)) * idf.get(t, 0.0) for t in themes)
    return float(numerator / denominator) if denominator > 0 else 0.0


def weighted_theme_cosine(
    left: Mapping[str, float], right: Mapping[str, float], idf: Mapping[str, float]
) -> float:
    """Cosine similarity after multiplying each theme weight by its IDF."""
    if not left or not right:
        return 0.0
    shared = left.keys() & right.keys()
    dot = sum(left[t] * idf.get(t, 0.0) * right[t] * idf.get(t, 0.0) for t in shared)
    left_norm = sum((weight * idf.get(theme, 0.0)) ** 2 for theme, weight in left.items())
    right_norm = sum((weight * idf.get(theme, 0.0)) ** 2 for theme, weight in right.items())
    denominator = float(np.sqrt(left_norm * right_norm))
    return float(dot / denominator) if denominator > 0 else 0.0


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    *,
    seed: int = 42,
    resamples: int = 1000,
) -> tuple[float, float, float]:
    """Mean and percentile 95% bootstrap CI, resampling independent seed rows."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(resamples, array.size))
    means = array[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(array.mean()), float(low), float(high)


def rank_of_target(scores: Sequence[float] | np.ndarray, catalog: Sequence[int], target: int) -> int:
    """One-based stable rank for a target ID among every catalogue item."""
    score_array = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-score_array, kind="stable")
    positions = {int(catalog[pos]): rank for rank, pos in enumerate(order, start=1)}
    try:
        return positions[int(target)]
    except KeyError as exc:
        raise ValueError(f"target item {target} is absent from the catalogue") from exc


def catalogue_coverage(recommendations: Sequence[Sequence[int]], catalog_size: int) -> float:
    """Share of the target catalogue appearing in at least one list."""
    if catalog_size <= 0:
        return 0.0
    return len({int(item) for row in recommendations for item in row}) / float(catalog_size)


def popularity_bias(
    recommendations: Sequence[Sequence[int]], popularity_percentiles: Mapping[int, float]
) -> float:
    """Mean catalogue popularity percentile of all returned items."""
    values = [
        popularity_percentiles[int(item)]
        for row in recommendations
        for item in row
        if int(item) in popularity_percentiles
    ]
    return float(np.mean(values)) if values else 0.0


def hubness(recommendations: Sequence[Sequence[int]], top_n: int = 10) -> float:
    """Fraction of recommendation slots occupied by the most frequent items."""
    counts = Counter(int(item) for row in recommendations for item in row)
    total = sum(counts.values())
    return sum(count for _, count in counts.most_common(top_n)) / total if total else 0.0


def validate_gold_pairs(
    pairs: pd.DataFrame, movies: pd.DataFrame, games: pd.DataFrame
) -> pd.DataFrame:
    """Validate gold-pair schema, catalogue IDs/titles, and pair uniqueness."""
    missing = set(GOLD_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"gold-pair file is missing columns: {sorted(missing)}")
    pairs = pairs[GOLD_COLUMNS].copy()
    if pairs[["movie_id", "game_app_id"]].duplicated().any():
        raise ValueError("gold-pair file contains duplicate movie/game IDs")

    movie_titles = movies.set_index("movie_id")["title"].astype(str)
    game_titles = games.set_index("app_id")["title"].astype(str)
    if pairs["movie_id"].isna().any() or pairs["game_app_id"].isna().any():
        raise ValueError("gold-pair IDs must not be empty")
    for row in pairs.itertuples(index=False):
        movie_id, game_id = int(row.movie_id), int(row.game_app_id)
        if movie_id not in movie_titles.index:
            raise ValueError(f"gold movie ID {movie_id} is absent from the catalogue")
        if game_id not in game_titles.index:
            raise ValueError(f"gold game ID {game_id} is absent from the catalogue")
        if str(row.movie_title) != movie_titles.loc[movie_id]:
            raise ValueError(f"gold movie title does not match ID {movie_id}")
        if str(row.game_title) != game_titles.loc[game_id]:
            raise ValueError(f"gold game title does not match ID {game_id}")
        if row.relation_type not in {"franchise", "adaptation", "thematic"}:
            raise ValueError(f"invalid relation type: {row.relation_type}")
    confidence = pd.to_numeric(pairs["confidence"], errors="coerce")
    if confidence.isna().any() or not confidence.between(0.0, 1.0).all():
        raise ValueError("gold-pair confidence must be a number in [0, 1]")
    return pairs


def _vectors(data: DomainData) -> dict[int, dict[str, float]]:
    vectors: dict[int, dict[str, float]] = {}
    for row in data.themes.itertuples(index=False):
        vectors.setdefault(int(row.item_id), {})[str(row.theme)] = float(row.weight)
    return vectors


def _seed_sample(
    data: DomainData, n_seeds: int, seed: int
) -> tuple[list[int], list[str]]:
    """Draw equal uniform and source-popularity strata, preserving draw counts."""
    rng = np.random.default_rng(seed)
    n_items = len(data.catalog)
    if n_items == 0:
        raise ValueError(f"{data.domain} catalogue is empty")
    n_random = n_seeds // 2
    n_popular = n_seeds - n_random
    popularity = popularity_scores(data)
    top_count = max(1, int(np.ceil(0.2 * n_items)))
    popular_positions = np.argsort(-popularity, kind="stable")[:top_count]
    random_positions = rng.choice(
        n_items, size=n_random, replace=n_random > n_items
    ).astype(int)
    remaining_popular = popular_positions[
        ~np.isin(popular_positions, random_positions)
    ]
    if n_popular <= len(remaining_popular):
        popular_draws = rng.choice(remaining_popular, size=n_popular, replace=False)
    else:
        popular_draws = rng.choice(
            popular_positions, size=n_popular, replace=n_popular > len(popular_positions)
        )
    popular_draws = popular_draws.astype(int)
    positions = np.concatenate((random_positions, popular_draws))
    strata = ["uniform"] * n_random + ["top20_popularity"] * n_popular
    order = rng.permutation(len(positions))
    return [int(data.catalog[pos]) for pos in positions[order]], [strata[i] for i in order]


def _genre_matrix(data: DomainData, genre_positions: dict[str, int]) -> np.ndarray:
    matrix = np.zeros((len(data.catalog), len(genre_positions)), dtype=np.float32)
    item_positions = data.catalog_index
    for row in data.genres.itertuples(index=False):
        item_pos = item_positions.get(int(row.item_id))
        genre_pos = genre_positions.get(str(row.genre))
        if item_pos is not None and genre_pos is not None:
            matrix[item_pos, genre_pos] = 1.0
    return matrix


def _ranked_ids(
    scores: np.ndarray, catalog: Sequence[int], k: int, *, positive_only: bool = False
) -> list[int]:
    positions = np.argsort(-scores, kind="stable")
    if positive_only:
        positions = positions[scores[positions] > 0]
    return [int(catalog[pos]) for pos in positions[:k]]


def _embedding_lookup(data: DomainData) -> dict[int, np.ndarray]:
    if data.embeddings is None:
        raise RuntimeError(f"{data.domain} catalogue has no cached item embeddings")
    return {
        int(item_id): data.embeddings[pos]
        for pos, item_id in enumerate(data.catalog)
    }


def _one_seed_metrics(
    source_id: int,
    ranked: Sequence[int],
    *,
    source_vectors: Mapping[int, Mapping[str, float]],
    target_vectors: Mapping[int, Mapping[str, float]],
    idf: Mapping[str, float],
    generic_threshold: float,
    embedding_scores: np.ndarray,
    target: DomainData,
) -> dict[str, float]:
    source_theme = source_vectors.get(int(source_id), {})
    target_positions = target.catalog_index
    jaccard: list[float] = []
    cosine: list[float] = []
    shared_count: list[float] = []
    non_generic: list[float] = []
    embedding: list[float] = []
    source_embedding_score_row = embedding_scores
    for raw_id in ranked:
        candidate = target_vectors.get(int(raw_id), {})
        shared = source_theme.keys() & candidate.keys()
        jaccard.append(weighted_jaccard(source_theme, candidate, idf))
        cosine.append(weighted_theme_cosine(source_theme, candidate, idf))
        shared_count.append(float(len(shared)))
        non_generic.append(
            float(any(idf.get(theme, 0.0) > generic_threshold for theme in shared))
        )
        position = target_positions[int(raw_id)]
        embedding.append(float(source_embedding_score_row[position]))

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    return {
        "theme_jaccard": mean(jaccard),
        "theme_cosine": mean(cosine),
        "shared_themes_per_result": mean(shared_count),
        "non_generic_share": mean(non_generic),
        "embedding_cosine_mean": mean(embedding),
        "embedding_cosine_rank1": embedding[0] if embedding else 0.0,
        "n_recommendations": float(len(ranked)),
    }


def _format_ci(value: tuple[float, float, float], digits: int = 4) -> str:
    return f"{value[0]:.{digits}f} [{value[1]:.{digits}f}, {value[2]:.{digits}f}]"


def _centered_bootstrap_ci(
    estimate: float,
    bootstrap_values: Sequence[float] | np.ndarray,
    *,
    lower_bound: float = 0.0,
    upper_bound: float = 1.0,
) -> tuple[float, float, float]:
    """Center percentile-bootstrap deviations on the observed estimate."""
    values = np.asarray(bootstrap_values, dtype=np.float64)
    if values.size == 0:
        return estimate, estimate, estimate
    deviations = values - values.mean()
    low, high = estimate + np.percentile(deviations, [2.5, 97.5])
    low = min(estimate, max(lower_bound, float(low)))
    high = max(estimate, min(upper_bound, float(high)))
    return estimate, low, high


def _evaluate_direction(
    cfg: Config,
    source_domain: str,
    target_domain: str,
    source: DomainData,
    target: DomainData,
    *,
    n_seeds: int,
    k_values: list[int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    started = time.perf_counter()
    source_vectors = _vectors(source)
    target_vectors = _vectors(target)
    all_theme_vectors = list(source_vectors.values()) + list(target_vectors.values())
    idf = theme_idf(
        all_theme_vectors, item_count=len(source.catalog) + len(target.catalog)
    )
    generic_threshold = float(np.median(list(idf.values()))) if idf else 0.0
    scorer = CrossDomainScorer(source, target)
    source_ids, strata = _seed_sample(source, n_seeds, seed)
    max_k = max(k_values)
    system_scores = scorer.score_sources(source_ids)

    genres = sorted(set(source.genres["genre"].astype(str)) | set(target.genres["genre"].astype(str)))
    genre_positions = {genre: pos for pos, genre in enumerate(genres)}
    source_genres = _genre_matrix(source, genre_positions)
    target_genres = _genre_matrix(target, genre_positions)
    source_positions = np.asarray([source.item_pos(item_id) for item_id in source_ids], dtype=int)
    genre_scores = source_genres[source_positions] @ target_genres.T

    pop_scores = popularity_scores(target)
    pop_order = np.argsort(-pop_scores, kind="stable")
    pop_ranked = [int(target.catalog[pos]) for pos in pop_order[:max_k]]
    pop_percentile_values = pd.Series(pop_scores).rank(method="average", pct=True).to_numpy()
    pop_percentiles = {
        int(target.catalog[pos]): float(pop_percentile_values[pos])
        for pos in range(len(target.catalog))
    }
    random_rng = np.random.default_rng(seed + 7919)
    rankings: dict[str, list[list[int]]] = {name: [] for name in METHODS}
    for row in range(n_seeds):
        scores_by_method = {
            "full_system": system_scores.combined_scores[row],
            "theme_only": system_scores.theme_scores[row],
            "embedding_only": system_scores.embedding_scores[row],
            "genre_only": genre_scores[row],
        }
        for method in METHODS:
            if method == "random":
                positions = random_rng.permutation(len(target.catalog))[:max_k]
                ranked = [int(target.catalog[pos]) for pos in positions]
            elif method == "popularity":
                ranked = pop_ranked.copy()
            else:
                ranked = _ranked_ids(
                    scores_by_method[method],
                    target.catalog,
                    max_k,
                    positive_only=method in {"full_system", "theme_only", "genre_only"},
                )
            rankings[method].append(ranked)

    seed_rows: list[dict[str, Any]] = []
    metric_values: dict[tuple[str, int, str], list[float]] = {}
    source_embedding_lookup = _embedding_lookup(source)
    target_embedding_lookup = _embedding_lookup(target)
    for row, (source_id, stratum) in enumerate(zip(source_ids, strata, strict=True)):
        source_embedding = source_embedding_lookup.get(int(source_id))
        for method in METHODS:
            ranked_max = rankings[method][row]
            embedding_row = system_scores.embedding_scores[row]
            for cutoff in k_values:
                ranked = ranked_max[:cutoff]
                values = _one_seed_metrics(
                    source_id,
                    ranked,
                    source_vectors=source_vectors,
                    target_vectors=target_vectors,
                    idf=idf,
                    generic_threshold=generic_threshold,
                    embedding_scores=embedding_row,
                    target=target,
                )
                record: dict[str, Any] = {
                    "profile": cfg.profile.name,
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "source_item_id": int(source_id),
                    "seed_stratum": stratum,
                    "method": method,
                    "k": cutoff,
                    **values,
                }
                if source_embedding is not None:
                    emb_scores = np.asarray(
                        [np.dot(source_embedding, target_embedding_lookup[item]) for item in ranked],
                        dtype=np.float64,
                    )
                    record["embedding_cosine_mean"] = (
                        float(emb_scores.mean()) if len(emb_scores) else 0.0
                    )
                    record["embedding_cosine_rank1"] = (
                        float(emb_scores[0]) if len(emb_scores) else 0.0
                    )
                    values["embedding_cosine_mean"] = record["embedding_cosine_mean"]
                    values["embedding_cosine_rank1"] = record["embedding_cosine_rank1"]
                seed_rows.append(record)
                for metric in SEED_METRICS:
                    metric_values.setdefault((method, cutoff, metric), []).append(
                        float(values[metric])
                    )

    rng = np.random.default_rng(seed + 104729)
    bootstrap_indices = rng.integers(0, n_seeds, size=(1000, n_seeds))
    summary: dict[str, Any] = {
        "rankings": rankings,
        "metric_values": metric_values,
        "bootstrap_indices": bootstrap_indices,
        "pop_percentiles": pop_percentiles,
        "duration_s": time.perf_counter() - started,
        "source_count": len(source.catalog),
        "target_count": len(target.catalog),
        "strata_counts": {name: strata.count(name) for name in sorted(set(strata))},
        "list_metrics": {},
        "list_boot": {},
        "strata_ci": {},
        "ci": {},
        "deltas": {},
    }
    for method in METHODS:
        summary["list_metrics"][method] = {}
        for cutoff in k_values:
            recommendations = [row[:cutoff] for row in rankings[method]]
            point = {
                "catalogue_coverage": catalogue_coverage(recommendations, len(target.catalog)),
                "popularity_bias": popularity_bias(recommendations, pop_percentiles),
                "hubness": hubness(recommendations),
            }
            boot_values = _bootstrap_list_metric_values(
                rankings[method],
                target.catalog,
                pop_percentiles,
                bootstrap_indices,
                cutoff,
            )
            for metric, value in point.items():
                summary["metric_values"][(method, cutoff, metric)] = [value]
            summary["list_metrics"][method][cutoff] = {
                metric: _centered_bootstrap_ci(point[metric], boot_values[metric])
                for metric in point
            }
            summary["list_boot"][(method, cutoff)] = boot_values
    for stratum_name in ("uniform", "top20_popularity"):
        row_indices = np.asarray(
            [pos for pos, value in enumerate(strata) if value == stratum_name],
            dtype=np.int32,
        )
        if not len(row_indices):
            continue
        stratum_rng = np.random.default_rng(seed + (1 if stratum_name == "uniform" else 2))
        stratum_bootstrap = stratum_rng.integers(
            0, len(row_indices), size=(1000, len(row_indices))
        )
        for method in METHODS:
            for cutoff in k_values:
                for metric in SEED_METRICS:
                    values = np.asarray(summary["metric_values"][(method, cutoff, metric)])
                    summary["strata_ci"][(stratum_name, method, cutoff, metric)] = bootstrap_ci(
                        values[row_indices], seed=seed + cutoff + len(method)
                    )
                stratum_rankings = [rankings[method][pos] for pos in row_indices]
                list_values = _bootstrap_list_metric_values(
                    stratum_rankings,
                    target.catalog,
                    pop_percentiles,
                    stratum_bootstrap,
                    cutoff,
                )
                stratum_recommendations = [row[:cutoff] for row in stratum_rankings]
                list_point = {
                    "catalogue_coverage": catalogue_coverage(
                        stratum_recommendations, len(target.catalog)
                    ),
                    "popularity_bias": popularity_bias(
                        stratum_recommendations, pop_percentiles
                    ),
                    "hubness": hubness(stratum_recommendations),
                }
                for metric, point in list_point.items():
                    summary["strata_ci"][(stratum_name, method, cutoff, metric)] = (
                        _centered_bootstrap_ci(point, list_values[metric])
                    )
    logger.info(
        "%s -> %s: evaluated %d seeds (%s) in %.1fs",
        source_domain,
        target_domain,
        n_seeds,
        summary["strata_counts"],
        summary["duration_s"],
    )
    return seed_rows, summary, system_scores.embedding_scores


def _gold_pairs(cfg: Config, movies: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    path = cfg.path("config", "cross_domain_gold_pairs.csv")
    if not path.exists():
        raise FileNotFoundError(f"cross-domain gold-pair file not found: {path}")
    pairs = pd.read_csv(path)
    if cfg.profile.name == "tiny":
        pairs = pairs[
            pairs["movie_id"].isin(movies["movie_id"])
            & pairs["game_app_id"].isin(games["app_id"])
        ]
    return validate_gold_pairs(pairs, movies, games)


def remove_document_title(text: str, title: str) -> str:
    """Remove only the leading title/year clause from an item document."""
    prefix = re.compile(
        rf"^\s*{re.escape(title)}(?:\s+\(\d{{4}}\))?\.\s*",
        flags=re.IGNORECASE,
    )
    masked = prefix.sub("", text, count=1)
    if masked == text:
        raise ValueError(f"document does not begin with the catalogue title {title!r}")
    return masked


def _gold_rank_rows(
    pairs: pd.DataFrame,
    movie: DomainData,
    game: DomainData,
    *,
    title_mask_status: str,
    masked_embeddings: dict[tuple[str, int], np.ndarray],
    seed: int,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    direction_data = {
        ("movie", "game"): (movie, game),
        ("game", "movie"): (game, movie),
    }
    scorers = {
        direction: CrossDomainScorer(source, target)
        for direction, (source, target) in direction_data.items()
    }
    popularity_by_domain = {
        domain: popularity_scores(data)
        for domain, data in (("movie", movie), ("game", game))
    }
    for pair_no, pair in enumerate(pairs.itertuples(index=False), start=1):
        for source_domain, target_domain, source_id, target_id in (
            ("movie", "game", int(pair.movie_id), int(pair.game_app_id)),
            ("game", "movie", int(pair.game_app_id), int(pair.movie_id)),
        ):
            source = movie if source_domain == "movie" else game
            target = game if target_domain == "game" else movie
            scorer = scorers[(source_domain, target_domain)]
            scores = scorer.score_sources([source_id])

            source_genres = set(
                source.genres.loc[source.genres["item_id"] == source_id, "genre"].astype(str)
            )
            genre_scores = np.zeros(len(target.catalog), dtype=np.float64)
            if source_genres:
                target_genres = target.genres[target.genres["item_id"].isin(target.catalog)]
                counts = target_genres[target_genres["genre"].astype(str).isin(source_genres)][
                    "item_id"
                ].value_counts()
                for pos, raw in enumerate(target.catalog):
                    genre_scores[pos] = float(counts.get(int(raw), 0))
            popularity = popularity_by_domain[target_domain]
            random_order = np.random.default_rng(
                seed + pair_no * 37 + (0 if source_domain == "movie" else 1)
            )
            random_scores = np.empty(len(target.catalog), dtype=np.float64)
            random_scores[random_order.permutation(len(target.catalog))] = np.arange(
                len(target.catalog), 0, -1
            )
            method_scores = {
                "random": random_scores,
                "popularity": popularity,
                "genre_only": genre_scores,
                "embedding_only": scores.embedding_scores[0],
                "theme_only": scores.theme_scores[0],
                "full_system": scores.combined_scores[0],
            }
            ranks = {
                method: rank_of_target(values, target.catalog, target_id)
                for method, values in method_scores.items()
            }

            masked_source = masked_embeddings.get((source_domain, source_id))
            masked_target = masked_embeddings.get((target_domain, target_id))
            if masked_source is not None or masked_target is not None:
                masked = scorer.score_sources(
                    [source_id],
                    source_embedding_overrides=(
                        {source_id: masked_source} if masked_source is not None else None
                    ),
                    target_embedding_overrides=(
                        {target_id: masked_target} if masked_target is not None else None
                    ),
                )
                ranks["embedding_only_title_masked"] = rank_of_target(
                    masked.embedding_scores[0], target.catalog, target_id
                )
                ranks["full_system_title_masked"] = rank_of_target(
                    masked.combined_scores[0], target.catalog, target_id
                )
            else:
                ranks["embedding_only_title_masked"] = np.nan
                ranks["full_system_title_masked"] = np.nan
            outputs.append(
                {
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "source_item_id": source_id,
                    "target_item_id": target_id,
                    "movie_id": int(pair.movie_id),
                    "game_app_id": int(pair.game_app_id),
                    "movie_title": str(pair.movie_title),
                    "game_title": str(pair.game_title),
                    "relation_type": str(pair.relation_type),
                    "gold_bucket": (
                        "thematic"
                        if pair.relation_type == "thematic"
                        else "franchise_adaptation"
                    ),
                    "confidence": float(pair.confidence),
                    **{f"rank_{method}": rank for method, rank in ranks.items()},
                    "title_mask_status": title_mask_status,
                }
            )
    return outputs


def _masked_gold_embeddings(
    cfg: Config, pairs: pd.DataFrame
) -> tuple[dict[tuple[str, int], np.ndarray], str]:
    """Re-embed only gold documents after removing their leading title."""
    client = OllamaClient.from_config(cfg)
    configured_timeout = client.timeout_s
    configured_retries = client.max_retries
    client.max_retries = 1
    client.timeout_s = min(configured_timeout, 3.0)
    models = client.list_models()
    client.timeout_s = configured_timeout
    client.max_retries = configured_retries
    if cfg.env.ollama_embed_model not in models:
        return {}, "skipped (Ollama or configured embedding model unavailable)"

    texts_path = cfg.processed_path("item_texts.parquet")
    texts = pd.read_parquet(texts_path).set_index(["domain", "item_id"])["text"]
    gold_items: dict[tuple[str, int], str] = {}
    for row in pairs.itertuples(index=False):
        gold_items[("movie", int(row.movie_id))] = str(row.movie_title)
        gold_items[("game", int(row.game_app_id))] = str(row.game_title)
    docs: list[tuple[str, int, str]] = []
    for (domain, item_id), title in gold_items.items():
        text = str(texts.loc[(domain, item_id)])
        masked = remove_document_title(text, title)
        chunks = chunk_text(
            masked,
            max_tokens=int(cfg.get("nlp.chunk_max_tokens", 400)),
            target_tokens=int(cfg.get("nlp.chunk_target_tokens", 300)),
            overlap_tokens=int(cfg.get("nlp.chunk_overlap_tokens", 50)),
            words_to_tokens=float(cfg.get("nlp.words_to_tokens", 1.3)),
        )
        docs.extend((domain, item_id, chunk.text) for chunk in chunks)

    try:
        vectors = client.embed_batch(
            [doc[2] for doc in docs],
            batch_size=int(cfg.get("nlp.embed_batch_size", 64)),
            dimensions=int(cfg.get("nlp.embed_dim", 1024)),
        )
    except OllamaError as exc:
        logger.warning("title-masked gold re-embedding skipped: %s", exc)
        return {}, f"skipped (Ollama embedding failed: {exc})"
    grouped: dict[tuple[str, int], list[np.ndarray]] = {}
    for (domain, item_id, _), vector in zip(docs, vectors, strict=True):
        grouped.setdefault((domain, item_id), []).append(vector)
    output = {
        key: l2_normalize(np.mean(np.vstack(chunks), axis=0, keepdims=True))[0]
        for key, chunks in grouped.items()
    }
    logger.info("title-masked re-embedded %d gold item documents", len(output))
    return output, f"completed for {len(output)} gold items"


def _gold_summary(
    rows: list[dict[str, Any]], seed: int
) -> dict[tuple[str, str, str], dict[str, tuple[float, float, float]]]:
    frame = pd.DataFrame(rows)
    result: dict[tuple[str, str, str], dict[str, tuple[float, float, float]]] = {}
    if frame.empty:
        return result
    rank_metrics = {
        "hit_at_5": lambda ranks: (ranks <= 5).astype(float),
        "hit_at_10": lambda ranks: (ranks <= 10).astype(float),
        "hit_at_20": lambda ranks: (ranks <= 20).astype(float),
        "mrr": lambda ranks: 1.0 / ranks,
        "median_rank": lambda ranks: ranks,
    }
    for (direction, bucket, method), group in (
        frame.melt(
            id_vars=["source_domain", "target_domain", "gold_bucket"],
            value_vars=[
                f"rank_{name}"
                for name in METHODS
                + ("embedding_only_title_masked", "full_system_title_masked")
            ],
            var_name="rank_column",
            value_name="rank",
        )
        .assign(
            direction=lambda d: d["source_domain"] + " -> " + d["target_domain"],
            method=lambda d: d["rank_column"].str.replace(r"^rank_", "", regex=True),
        )
        .groupby(["direction", "gold_bucket", "method"], sort=False)
    ):
        ranks = group["rank"].dropna().to_numpy(dtype=np.float64)
        if not len(ranks):
            continue
        metrics: dict[str, tuple[float, float, float]] = {}
        for metric, transform in rank_metrics.items():
            values = transform(ranks)
            if metric == "median_rank":
                rng = np.random.default_rng(seed + len(ranks))
                indices = rng.integers(0, len(ranks), size=(1000, len(ranks)))
                boot = np.median(ranks[indices], axis=1)
                metrics[metric] = (
                    float(np.median(ranks)),
                    float(np.percentile(boot, 2.5)),
                    float(np.percentile(boot, 97.5)),
                )
            else:
                metrics[metric] = bootstrap_ci(values, seed=seed + len(ranks))
        result[(direction, bucket, method)] = metrics
    return result


def _build_report(
    cfg: Config,
    *,
    n_seeds: int,
    k_values: list[int],
    seed: int,
    summaries: dict[str, dict[str, Any]],
    gold_rows: list[dict[str, Any]],
    gold_pairs_count: int,
    title_mask_status: str,
    duration_s: float,
    gold_only: bool,
) -> str:
    development = (
        "> **DEVELOPMENT SAMPLE:** this report uses the tiny profile and is only a sanity check.\n\n"
        if cfg.profile.name == "tiny"
        else ""
    )
    lines = [
        "# Cross-domain evaluation",
        "",
        development.rstrip(),
        f"Profile: `{cfg.profile.name}`. Random seed: `{seed}`. Bootstrap resamples: 1,000.",
        f"Gold pairs: {gold_pairs_count}; title-masked check: {title_mask_status}.",
        f"Runtime: {duration_s:.1f} s.",
        "",
        "## Protocol and interpretation",
        "",
        "MovieLens and Steam have no common user IDs, so cross-domain Recall/NDCG against "
        "user-held-out ground truth is not defined. The seed proxy draws half uniformly "
        "from the source catalogue and half from its top-20% positive-interaction items; "
        "the same seed rows are used for every method. Popularity-stratum draws avoid "
        "uniform-stratum items when enough top-20% candidates remain; draws use replacement "
        "only when a stratum is too small (as can happen on tiny). Each result list is scored against "
        "the seed's item themes and cached item embedding.",
        "",
        "The `full_system` ranking is the deterministic `cross` command scorer: source "
        "theme profile, target-catalogue theme IDF (`ln(n_themed/df)`), and embedding "
        "cosine, min-maxed per seed and blended 0.6/0.4. The chatbot's cross-reference "
        "path transfers the same weighted theme vector through `extra_theme_weights`; "
        "the report omits its query-driven candidate retrieval, LLM intent, and answer "
        "generation. No Neo4j or LLM generation is used.",
        "",
        "Seed-level proxy metrics use `idf(t)=ln((N+1)/(df(t)+1))+1`, with `N` all items "
        "in both domains and document frequency counted once per item. Weighted Jaccard "
        "is `sum(min(w_s,w_r)*idf)/sum(max(w_s,w_r)*idf)`; theme cosine is cosine after "
        "multiplying each vector weight by IDF. Generic themes are those with IDF at or "
        "below the median. Embedding cosine is computed from the cached item vectors.",
        "",
        f"The optional title mask removes each gold item's leading title/year clause and "
        f"re-embeds only those gold documents with `{cfg.env.ollama_embed_model}` at "
        f"{int(cfg.get('nlp.embed_dim', 1024))} dimensions; document embeddings use no "
        "query-instruction prefix, matching the regular embedder.",
        "",
        "Beyond-accuracy metrics aggregate all lists at each cutoff: catalogue coverage "
        "is distinct recommended target items / target catalogue size; popularity bias "
        "is the mean target-catalogue positive-interaction percentile; hubness is the "
        "share of recommendation slots occupied by the ten most-recommended items. "
        "Intervals are percentile 95% bootstrap CIs resampled over seed rows; baseline "
        "deltas use paired seed resamples.",
        "",
        "The embedding-only baseline is favoured by embedding-cosine and theme-only by "
        "theme-overlap by construction (metric circularity). Read across all measures, "
        "not as independent evidence of user relevance.",
        "Seed-level theme/embedding measures use percentile bootstrap means. The "
        "set-valued coverage and hubness measures use centered percentile-bootstrap "
        "deviations to reduce occupancy bias from duplicate seed rows in resamples.",
        "",
    ]
    if not gold_only:
        for direction in ("movie -> game", "game -> movie"):
            summary = summaries[direction]
            lines.extend([f"## {direction}: seed proxy results", ""])
            lines.append(
                f"Seeds: {n_seeds} draws ({summary['strata_counts']}); "
                f"catalogues: {summary['source_count']} source / {summary['target_count']} target; "
                f"runtime: {summary['duration_s']:.1f} s."
            )
            lines.append("")
            for cutoff in k_values:
                lines.extend(
                    [
                        f"### Metrics @ {cutoff}",
                        "",
                        "| method | IDF Jaccard | IDF theme cosine | shared themes/result | non-generic share | "
                        "embedding cosine mean | embedding rank-1 | catalogue coverage | popularity bias | hubness |",
                        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                    ]
                )
                for method in METHODS:
                    cells = []
                    for metric in SEED_METRICS:
                        cells.append(_format_ci(summary["ci"][(method, cutoff, metric)]))
                    for metric in ("catalogue_coverage", "popularity_bias", "hubness"):
                        cells.append(
                            _format_ci(summary["list_metrics"][method][cutoff][metric], 3)
                        )
                    lines.append(f"| {method} | " + " | ".join(cells) + " |")
                lines.append("")

            strata_cutoff = min(10, max(k_values))
            lines.extend(
                    [
                        f"### Seed strata @ {strata_cutoff}",
                        "",
                        "| stratum | method | IDF Jaccard | IDF theme cosine | shared themes/result | "
                        "non-generic share | embedding cosine mean | embedding rank-1 | "
                        "catalogue coverage | popularity bias | hubness |",
                        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                    ]
            )
            for stratum_name in ("uniform", "top20_popularity"):
                    for method in METHODS:
                        cells = [
                            _format_ci(
                                summary["strata_ci"][
                                    (stratum_name, method, strata_cutoff, metric)
                                ]
                            )
                            for metric in SEED_METRICS
                        ]
                        cells.extend(
                            _format_ci(
                                summary["strata_ci"][
                                    (stratum_name, method, strata_cutoff, metric)
                                ],
                                3,
                            )
                            for metric in (
                                "catalogue_coverage",
                                "popularity_bias",
                                "hubness",
                            )
                        )
                        lines.append(
                            f"| {stratum_name} | {method} | " + " | ".join(cells) + " |"
                        )
            lines.append("")

            for cutoff in k_values:
                lines.extend(
                    [
                        f"### Paired full-system differences @ {cutoff}",
                        "",
                        "| baseline | metric | full minus baseline (95% CI) |",
                        "|---|---|---:|",
                    ]
                )
                for baseline in METHODS:
                    if baseline == "full_system":
                        continue
                    for metric in SEED_METRICS + (
                        "catalogue_coverage",
                        "popularity_bias",
                        "hubness",
                    ):
                        lines.append(
                            f"| {baseline} | {metric} | "
                            f"{_format_ci(summary['deltas'][(baseline, cutoff, metric)])} |"
                        )
                lines.append("")

    lines.extend(["## Gold-pair all-catalogue ranks", ""])
    lines.append(
        f"{gold_pairs_count} curated pairs are unreviewed (`reviewed=false`) and both "
        "directions are ranked against every target item. Bootstrap CIs resample pairs. "
        f"Title-mask status: {title_mask_status}."
    )
    lines.append("")
    gold_summary = _gold_summary(gold_rows, seed)
    rank_methods = METHODS + ("embedding_only_title_masked", "full_system_title_masked")
    for direction in ("movie -> game", "game -> movie"):
        lines.extend([f"### {direction}", ""])
        lines.append("| pair group | method | Hit@5 | Hit@10 | Hit@20 | MRR | median rank |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for bucket in ("franchise_adaptation", "thematic"):
            for method in rank_methods:
                metrics = gold_summary.get((direction, bucket, method))
                if not metrics:
                    continue
                lines.append(
                    f"| {bucket} | {method} | "
                    + " | ".join(
                        _format_ci(metrics[key], 3)
                        for key in ("hit_at_5", "hit_at_10", "hit_at_20", "mrr", "median_rank")
                    )
                    + " |"
                )
        lines.append("")
    lines.extend(
        [
            "Per-pair ranks are saved in `cross_domain_gold_ranks.csv`; the unreviewed "
            "candidate-selection notes are in `cross_domain_gold_unresolved.md`.",
            "",
            "## Limitations and rerun",
            "",
            "These proxy scores establish similarity under the project's own learned "
            "representations, not shared-user preference, satisfaction, or causal lift. "
            "Genre-only may return short lists when no canonical genre overlaps. The gold "
            "set is small and not yet human-reviewed; franchise/adaptation labels and "
            "thematic judgments need independent review. Title masking diagnoses title "
            "leakage only for these gold items; it does not remove all franchise or "
            "metadata cues.",
            "",
            f"Rerun with `python -m entdec crosseval --profile {cfg.profile.name} "
            f"--seeds {n_seeds} --k 10` (or `--gold-only`). Runtime: {duration_s:.1f} s.",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def run_evaluation(
    cfg: Config,
    *,
    n_seeds: int = 1000,
    k: int = 10,
    gold_only: bool = False,
    no_title_mask_check: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run both recommendation directions and write profile-specific artefacts."""
    if n_seeds <= 0:
        raise ValueError("--seeds must be positive")
    if k <= 0:
        raise ValueError("--k must be positive")
    started = time.perf_counter()
    k_values = sorted({5, int(k), 20})
    seed = cfg.seed
    movie = load_domain(cfg, "movie", with_embeddings=True)
    game = load_domain(cfg, "game", with_embeddings=True)
    pairs = _gold_pairs(cfg, movie.items, game.items)

    seed_rows: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    if not gold_only:
        for direction, source_domain, target_domain, source, target, offset in (
            ("movie -> game", "movie", "game", movie, game, 0),
            ("game -> movie", "game", "movie", game, movie, 1),
        ):
            rows, summary, _ = _evaluate_direction(
                cfg,
                source_domain,
                target_domain,
                source,
                target,
                n_seeds=n_seeds,
                k_values=k_values,
                seed=seed + offset * 1009,
            )
            seed_rows.extend(rows)
            summaries[direction] = summary
            bootstrap_indices = summary["bootstrap_indices"]
            for method in METHODS:
                for cutoff in k_values:
                    for metric in SEED_METRICS:
                        values = np.asarray(summary["metric_values"][(method, cutoff, metric)])
                        sampled = values[bootstrap_indices]
                        means = sampled.mean(axis=1)
                        low, high = np.percentile(means, [2.5, 97.5])
                        summary["ci"][(method, cutoff, metric)] = (
                            float(values.mean()),
                            float(low),
                            float(high),
                        )
            for cutoff in k_values:
                for baseline in METHODS:
                    if baseline == "full_system":
                        continue
                    for metric in SEED_METRICS:
                        full = np.asarray(
                            summary["metric_values"][("full_system", cutoff, metric)]
                        )
                        base = np.asarray(summary["metric_values"][(baseline, cutoff, metric)])
                        diffs = (full - base)[bootstrap_indices].mean(axis=1)
                        low, high = np.percentile(diffs, [2.5, 97.5])
                        summary["deltas"][(baseline, cutoff, metric)] = (
                            float((full - base).mean()),
                            float(low),
                            float(high),
                        )
                    for metric in ("catalogue_coverage", "popularity_bias", "hubness"):
                        full_values = summary["list_boot"][("full_system", cutoff)][metric]
                        base_values = summary["list_boot"][(baseline, cutoff)][metric]
                        delta_values = full_values - base_values
                        point_delta = float(
                            summary["list_metrics"]["full_system"][cutoff][metric][0]
                            - summary["list_metrics"][baseline][cutoff][metric][0]
                        )
                        summary["deltas"][(baseline, cutoff, metric)] = (
                            _centered_bootstrap_ci(
                                point_delta,
                                delta_values,
                                lower_bound=-1.0,
                                upper_bound=1.0,
                            )
                        )

    masked_embeddings: dict[tuple[str, int], np.ndarray] = {}
    title_mask_status = "not requested"
    if not no_title_mask_check:
        masked_embeddings, title_mask_status = _masked_gold_embeddings(cfg, pairs)
    gold_rows = _gold_rank_rows(
        pairs,
        movie,
        game,
        title_mask_status=title_mask_status,
        masked_embeddings=masked_embeddings,
        seed=seed,
    )

    reports = cfg.paths.reports
    reports.mkdir(parents=True, exist_ok=True)
    seed_frame = pd.DataFrame(seed_rows)
    seed_frame.to_csv(reports / "cross_domain_eval.csv", index=False)
    gold_frame = pd.DataFrame(gold_rows)
    gold_frame.to_csv(reports / "cross_domain_gold_ranks.csv", index=False)
    duration_s = time.perf_counter() - started
    report = _build_report(
        cfg,
        n_seeds=n_seeds,
        k_values=k_values,
        seed=seed,
        summaries=summaries,
        gold_rows=gold_rows,
        gold_pairs_count=len(pairs),
        title_mask_status=title_mask_status,
        duration_s=duration_s,
        gold_only=gold_only,
    )
    (reports / "cross_domain_eval.md").write_text(report, encoding="utf-8")
    logger.info(
        "cross-domain evaluation wrote reports/%s/cross_domain_eval.md, "
        "cross_domain_eval.csv, and cross_domain_gold_ranks.csv in %.1fs",
        cfg.profile.name,
        duration_s,
    )
    return seed_frame, gold_frame, title_mask_status


def _bootstrap_list_metric_values(
    recommendations: list[list[int]],
    catalog: Sequence[int],
    popularity_percentiles: Mapping[int, float],
    bootstrap_indices: np.ndarray,
    cutoff: int,
) -> dict[str, np.ndarray]:
    """Values per shared bootstrap draw for paired list-metric differences."""
    target_to_pos = {int(raw): pos for pos, raw in enumerate(catalog)}
    selected = np.full((len(recommendations), cutoff), -1, dtype=np.int32)
    for row, ranked in enumerate(recommendations):
        for col, raw in enumerate(ranked[:cutoff]):
            selected[row, col] = target_to_pos[int(raw)]
    output = {
        name: np.zeros(len(bootstrap_indices), dtype=np.float64)
        for name in ("catalogue_coverage", "popularity_bias", "hubness")
    }
    popularity_by_pos = np.asarray(
        [popularity_percentiles[int(raw)] for raw in catalog], dtype=np.float64
    )
    top_n = min(10, len(catalog))
    for sample_no, sample_indices in enumerate(bootstrap_indices):
        items = selected[sample_indices].ravel()
        items = items[items >= 0]
        if not len(items):
            continue
        counts = np.bincount(items, minlength=len(catalog))
        output["catalogue_coverage"][sample_no] = np.count_nonzero(counts) / max(
            len(catalog), 1
        )
        output["hubness"][sample_no] = np.partition(counts, -top_n)[-top_n:].sum() / len(items)
        output["popularity_bias"][sample_no] = popularity_by_pos[items].mean()
    return output


def run(cfg: Config, args: argparse.Namespace | None = None) -> int:
    """CLI entry point: ``python -m entdec crosseval --profile default``."""
    run_evaluation(
        cfg,
        n_seeds=int(getattr(args, "seeds", 1000)),
        k=int(getattr(args, "k", 10)),
        gold_only=bool(getattr(args, "gold_only", False)),
        no_title_mask_check=bool(getattr(args, "no_title_mask_check", False)),
    )
    return 0


__all__ = [
    "bootstrap_ci",
    "catalogue_coverage",
    "hubness",
    "popularity_bias",
    "rank_of_target",
    "remove_document_title",
    "run",
    "run_evaluation",
    "theme_idf",
    "validate_gold_pairs",
    "weighted_jaccard",
    "weighted_theme_cosine",
]
