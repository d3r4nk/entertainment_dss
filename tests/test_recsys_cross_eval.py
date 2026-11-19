"""Hand-checked metrics and catalogue validation for cross-domain evaluation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from entdec.recsys.cross_domain import CrossDomainScorer
from entdec.recsys.cross_eval import (
    _centered_bootstrap_ci,
    _seed_sample,
    bootstrap_ci,
    catalogue_coverage,
    hubness,
    popularity_bias,
    rank_of_target,
    remove_document_title,
    theme_idf,
    validate_gold_pairs,
    weighted_jaccard,
    weighted_theme_cosine,
)


def _domain(catalog: list[int], themes: pd.DataFrame, embeddings: np.ndarray):
    positions = {item: pos for pos, item in enumerate(catalog)}
    return SimpleNamespace(
        domain="movie",
        catalog=catalog,
        themes=themes,
        embeddings=embeddings,
        item_pos=lambda item: positions.get(int(item)),
    )


def test_idf_weighted_theme_overlap_and_empty_edges() -> None:
    vectors = [{"a": 1.0}, {"a": 0.5, "b": 1.0}, {}]
    idf = theme_idf(vectors, item_count=3)
    expected_a = np.log(4 / 3) + 1
    expected_b = np.log(4 / 2) + 1
    assert idf == pytest.approx({"a": expected_a, "b": expected_b})

    left = {"a": 2.0, "b": 1.0}
    right = {"a": 1.0, "c": 3.0}
    assert weighted_jaccard(left, right, {"a": 2.0, "b": 1.0, "c": 1.0}) == pytest.approx(
        0.25
    )
    assert weighted_jaccard({}, right, idf) == 0.0
    assert weighted_jaccard({"a": 1.0}, {"b": 1.0}, idf) == 0.0


def test_weighted_theme_cosine() -> None:
    assert weighted_theme_cosine(
        {"a": 1.0, "b": 1.0}, {"a": 1.0}, {"a": 1.0, "b": 1.0}
    ) == pytest.approx(1 / np.sqrt(2))
    assert weighted_theme_cosine({}, {"a": 1.0}, {"a": 1.0}) == 0.0


def test_bootstrap_ci_is_deterministic() -> None:
    values = [0.0, 0.5, 1.0]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    mean, low, high = bootstrap_ci(values, seed=7)
    assert mean == pytest.approx(0.5)
    assert low <= mean <= high


def test_centered_bootstrap_ci_tracks_observed_non_additive_metric() -> None:
    estimate, low, high = _centered_bootstrap_ci(0.84, [0.68, 0.69, 0.70])
    assert estimate == pytest.approx(0.84)
    assert low == pytest.approx(0.8305)
    assert high == pytest.approx(0.8495)


def test_seed_sampling_keeps_popularity_strata_disjoint_when_possible() -> None:
    data = SimpleNamespace(
        domain="movie",
        catalog=np.arange(1000),
        matrix=sparse.diags(np.arange(1000, dtype=np.float64)),
    )
    ids, strata = _seed_sample(data, 100, seed=7)
    assert len(ids) == len(set(ids)) == 100
    assert strata.count("uniform") == strata.count("top20_popularity") == 50
    assert all(item >= 800 for item, group in zip(ids, strata, strict=True) if group == "top20_popularity")


def test_title_mask_removes_only_the_document_prefix() -> None:
    assert remove_document_title("The Film (2001). Themes: mystery.", "The Film") == (
        "Themes: mystery."
    )
    with pytest.raises(ValueError, match="does not begin"):
        remove_document_title("Other title. Themes: mystery.", "The Film")


def test_rank_coverage_hubness_and_popularity_bias() -> None:
    assert rank_of_target([0.3, 0.9, 0.2], [10, 20, 30], 10) == 2
    assert catalogue_coverage([[1, 2], [2, 3]], 4) == pytest.approx(0.75)
    assert catalogue_coverage([], 0) == 0.0
    recommendations = [[1, 2], [1, 3], [1, 4]]
    assert hubness(recommendations, top_n=1) == pytest.approx(0.5)
    assert popularity_bias(recommendations, {1: 1.0, 2: 0.2, 3: 0.4, 4: 0.6}) == pytest.approx(
        0.7
    )
    assert hubness([]) == 0.0
    assert popularity_bias([], {}) == 0.0


def test_cross_scorer_uses_shared_theme_and_embedding_components() -> None:
    source = _domain(
        [1],
        pd.DataFrame({"item_id": [1], "theme": ["story"], "weight": [1.0]}),
        np.array([[1.0, 0.0]], dtype=np.float32),
    )
    target = _domain(
        [2, 3],
        pd.DataFrame(
            {
                "item_id": [2, 3],
                "theme": ["story", "horror"],
                "weight": [1.0, 1.0],
            }
        ),
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    scores = CrossDomainScorer(source, target).score_sources([1])
    assert scores.theme_scores.shape == (1, 2)
    assert scores.embedding_scores[0, 0] == pytest.approx(1.0)
    assert scores.embedding_scores[0, 1] == pytest.approx(0.0)
    assert scores.combined_scores[0, 0] > scores.combined_scores[0, 1]


def test_gold_pair_schema_ids_and_duplicate_validation() -> None:
    movies = pd.DataFrame({"movie_id": [10], "title": ["Movie"]})
    games = pd.DataFrame({"app_id": [20], "title": ["Game"]})
    valid = pd.DataFrame(
        [
            {
                "movie_id": 10,
                "game_app_id": 20,
                "movie_title": "Movie",
                "game_title": "Game",
                "relation_type": "thematic",
                "confidence": 0.8,
                "note": "Thematic similarity.",
                "reviewed": False,
            }
        ]
    )
    assert len(validate_gold_pairs(valid, movies, games)) == 1
    with pytest.raises(ValueError, match="missing columns"):
        validate_gold_pairs(valid.drop(columns="note"), movies, games)
    with pytest.raises(ValueError, match="movie ID"):
        validate_gold_pairs(valid.assign(movie_id=99), movies, games)
    with pytest.raises(ValueError, match="duplicate"):
        validate_gold_pairs(pd.concat([valid, valid]), movies, games)
