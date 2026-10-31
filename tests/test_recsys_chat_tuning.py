"""Tests for validation tuning of anonymous-chat weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

import entdec.config as config_module
from entdec.config import Config, load_config
from entdec.recsys.common import DomainData
from entdec.recsys.evaluate import (
    Evaluator,
    save_chat_weights,
    tune_chat_weights,
)


def _toy_data() -> DomainData:
    items = pd.DataFrame(
        {"movie_id": [100, 101, 102], "title": ["Liked", "Validation", "Test"]}
    )
    interactions = pd.DataFrame(
        {
            "uid": [1, 1, 1],
            "movie_id": [100, 101, 102],
            "is_positive": [True, True, True],
            "split": ["train", "valid", "test"],
        }
    )
    return DomainData(
        domain="movie",
        items=items,
        interactions=interactions,
        item_key="movie_id",
        catalog=[100, 101, 102],
        catalog_index={100: 0, 101: 1, 102: 2},
        user_ids=[1],
        user_indices=pd.Series([0], index=[1]),
        matrix=csr_matrix(([1.0], ([0], [0])), shape=(1, 3)),
        likes={1: [100]},
        valid_targets={1: 101},
        test_targets={1: 102},
        themes=pd.DataFrame(
            {
                "item_id": [100, 101, 102],
                "theme": ["shared", "shared", "other"],
                "weight": [1.0, 1.0, 1.0],
            }
        ),
        genres=pd.DataFrame(columns=["item_id", "genre"]),
        embeddings=np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32),
    )


def test_chat_validation_uses_liked_item_query_and_valid_target() -> None:
    evaluator = Evaluator(load_config("tiny"), _toy_data())
    cache = evaluator.prepare_chat_tuning()

    assert len(cache) == 1
    user, parts, unseen_mask = cache[0]
    assert user == 1
    assert set(parts) == {"popularity", "content", "kg"}
    assert unseen_mask.tolist() == [False, True, True]
    assert evaluator.chat_validation_score(
        {"popularity": 0.0, "content": 1.0, "kg": 0.0}
    ) == pytest.approx(1.0)


def test_chat_tuner_only_uses_allowed_models_and_small_popularity_prior() -> None:
    observed: list[dict[str, float]] = []

    def score(weights: dict[str, float]) -> float:
        observed.append(dict(weights))
        return weights["content"]

    weights, score_value = tune_chat_weights(score)

    assert all(set(candidate) == {"popularity", "content", "kg"} for candidate in observed)
    assert all(sum(candidate.values()) == pytest.approx(1.0) for candidate in observed)
    assert max(candidate["popularity"] for candidate in observed) <= 0.1
    assert weights == {"popularity": 0.0, "content": 1.0, "kg": 0.0}
    assert score_value == pytest.approx(1.0)


def test_saving_chat_weights_preserves_existing_profile_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config_module, "PROJECT_ROOT", tmp_path)
    cfg: Config = load_config("tiny")
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    tuned_path = config_dir / "tuned_weights.yaml"
    tuned_path.write_text(
        "tiny:\n"
        "  movie:\n"
        "    mf: 0.5\n"
        "    content: 0.5\n"
        "default:\n"
        "  game:\n"
        "    mf: 1.0\n",
        encoding="utf-8",
    )

    save_chat_weights(
        cfg,
        {"popularity": 0.05, "content": 0.475, "kg": 0.475},
        "movie",
    )

    saved = config_module.load_yaml(tuned_path)
    assert saved["tiny"]["movie"] == {"mf": 0.5, "content": 0.5}
    assert saved["default"]["game"] == {"mf": 1.0}
    assert cfg.load_tuned_weights() == {"movie": {"mf": 0.5, "content": 0.5}}
    assert cfg.load_tuned_chat_weights("movie") == {
        "popularity": 0.05,
        "content": 0.475,
        "kg": 0.475,
    }
