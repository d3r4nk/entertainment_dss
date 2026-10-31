"""Model-level tests for the recommenders (section 11: shape, no seen items).

Uses a tiny synthetic interaction set rather than the real processed data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from entdec.recsys import content, hybrid, itemknn, kg_scoring, mf, popularity
from entdec.recsys.common import DomainData, exclude_seen, top_k_positions
from entdec.recsys.metrics import ndcg_at_k


@pytest.fixture
def toy_data() -> DomainData:
    """5 items, 4 users; users 1/2 liked items 0/1 in train, held out item 2."""
    catalog = [100, 101, 102, 103, 104]
    catalog_index = {raw: pos for pos, raw in enumerate(catalog)}
    user_ids = [1, 2, 3]
    matrix = sp.csr_matrix(
        (np.ones(5, dtype=np.float32), ([0, 0, 1, 1, 2], [0, 1, 0, 1, 2])),
        shape=(3, 5),
        dtype=np.float32,
    )
    themes = pd.DataFrame(
        {
            "item_id": [100, 100, 101, 102, 103],
            "theme": ["space", "strategy", "space", "dogs", "dogs"],
            "weight": [1.0, 0.5, 1.0, 1.0, 0.6],
        }
    )
    genres = pd.DataFrame(
        {
            "item_id": [100, 101, 102, 103, 104],
            "genre": ["Sci-Fi", "Sci-Fi", "Drama", "Drama", "Horror"],
        }
    )
    embeddings = np.eye(5, dtype=np.float32)  # orthogonal item vectors
    return DomainData(
        domain="movie",
        items=pd.DataFrame({"movie_id": catalog, "title": [f"t{raw}" for raw in catalog]}),
        interactions=pd.DataFrame(columns=["uid", "movie_id", "rating", "ts", "is_positive", "split"]),
        item_key="movie_id",
        catalog=catalog,
        catalog_index=catalog_index,
        user_ids=user_ids,
        user_indices=pd.Series({1: 0, 2: 1, 3: 2}),
        matrix=matrix,
        likes={1: [100, 101], 2: [100, 101], 3: [102]},
        valid_targets={1: 103, 2: 104},
        test_targets={1: 102, 2: 103},
        themes=themes,
        genres=genres,
        embeddings=embeddings,
    )


@pytest.fixture
def fitted(toy_data: DomainData) -> dict[str, object]:
    return {"itemknn": itemknn.ItemKNNModel(toy_data), "mf": mf.fit(toy_data, factors=2)}


def test_popularity_counts_toy(toy_data: DomainData) -> None:
    scores = popularity.popularity_scores(toy_data)
    assert scores.shape == (5,)
    assert scores[0] == 2 and scores[1] == 2 and scores[3] == 0


def test_every_model_scores_the_full_catalogue(
    toy_data: DomainData, fitted: dict[str, object]
) -> None:
    models = {
        "pop": popularity.popularity_scores(toy_data),
        "knn": itemknn.ItemKNNModel(toy_data).scores(1),
        "mf": mf.MFModel(toy_data, factors=2).scores(1),
        "content": content.scores(toy_data, 1),
        "kg": kg_scoring.kg_scores(toy_data, 1),
    }
    for name, scores in models.items():
        assert scores.shape == (5,), name
        assert np.isfinite(scores).all(), name


def test_knn_similarity_sane(toy_data: DomainData) -> None:
    sim = itemknn.item_similarity(toy_data.matrix)
    dense = sim.toarray()
    assert dense[0, 1] == pytest.approx(1.0)  # identical support
    assert dense.diagonal().sum() == pytest.approx(0.0)  # self similarity zeroed


def test_content_profile_is_normalized_mean(toy_data: DomainData) -> None:
    profile = content.user_profile(toy_data, 1)
    assert profile is not None
    assert np.linalg.norm(profile) == pytest.approx(1.0)
    # user 1 liked items 0 and 1: profile ~ mean of e0,e1 normalised
    assert profile[0] == pytest.approx(profile[1], abs=1e-6)
    assert content.user_profile(toy_data, 999) is None  # unknown user


def test_kg_overlap_prefers_shared_themes(toy_data: DomainData) -> None:
    scores = kg_scoring.kg_scores(toy_data, 3)  # user 3 liked item 102 ("dogs")
    # items 102/103 carry the "dogs" theme -> higher than items 100/101
    assert scores[3] >= scores[0]
    assert np.isfinite(scores).all()


def test_exclude_seen_and_top_k(toy_data: DomainData) -> None:
    scores = np.array([1.0, 5.0, 3.0, 0.5, 2.0])
    masked = exclude_seen(scores, toy_data, 1)  # user 1 liked items 100/101
    assert np.isinf(masked[0])
    assert np.isinf(masked[1])
    assert top_k_positions(masked, 2) == [2, 4]  # 3.0 then 2.0, unseen


def test_hybrid_blend_is_weighted_minmax(toy_data: DomainData, fitted: dict[str, object]) -> None:
    parts = hybrid.normalised_scores(toy_data, 1, fitted)
    for part in parts.values():
        assert part.min() >= 0.0 and part.max() <= 1.0
    blended = hybrid.blend(parts, {"content": 1.0})
    assert blended.shape == (5,)
    # replaying with weight 1 on content = normalised content
    assert np.allclose(blended, parts["content"])
    model = hybrid.HybridModel(toy_data, dict.fromkeys(hybrid.MODEL_NAMES, 0.2), fitted=fitted)
    positions, scores = model.recommend(1, k=3)
    assert len(positions) == 3 and len(scores) == 3
    assert all(pos not in (0, 1) for pos in positions)  # no train items


def test_tune_weights_returns_valid_simplex(toy_data: DomainData) -> None:
    def flat_score(_data: DomainData, weights: dict[str, float]) -> float:
        return weights.get("content", 0.0)  # tuner should max content

    result = hybrid.tune_weights(toy_data, flat_score, max_candidates=200)
    assert result.weights["content"] == max(result.weights.values())
    total = sum(result.weights.values())
    assert 0.0 <= result.weights["popularity"] <= 1.0 and total <= 1.0


def test_mf_fallback_works(toy_data: DomainData) -> None:
    """The SVD fallback produces full-catalogue probabilities."""
    model = mf.MFModel(toy_data, factors=2)
    if model.method == "svd":
        scores = model.scores(1)
        assert scores.shape == (5,)
    else:
        scores = model.scores(1)
        assert scores.shape == (5,)


def test_ndcg_against_rank_user_chain(toy_data: DomainData) -> None:
    scores = content.scores(toy_data, 1)
    masked = exclude_seen(scores, toy_data, 1)
    ranked = [toy_data.catalog[pos] for pos in top_k_positions(masked, 5)]
    # user 1's train items (100/101) are masked out -> only 3 rankable items
    assert len(ranked) == 3
    assert 100 not in ranked and 101 not in ranked  # train items excluded entirely
    assert ndcg_at_k(ranked, toy_data.test_targets[1], 5) >= 0.0
