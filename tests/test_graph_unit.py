"""Unit tests for the graph helpers (PROJECT_SPEC.md section 5) - no server needed.

Neo4j bolt and docker are exercised in the integration tests instead, so this
module stubs :class:`entdec.graph.neo4j_client.Neo4jClient` with a fake client
that records the statements it was given.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import pytest

from entdec.graph import ids
from entdec.graph.queries import templates, vector_query_params
from entdec.graph.schema import UNIQUENESS_CONSTRAINTS, VECTOR_INDEX_NAME, apply_schema
from entdec.graph.similar import build_similar_to_frame, top_k_similar


# --------------------------------------------------------------------------- #
# fake graph client
# --------------------------------------------------------------------------- #
class FakeClient:
    """Collects queries and fakes the SHOW INDEXES round trip."""

    def __init__(self) -> None:
        self.queries: list[tuple[str, dict[str, Any]]] = []
        self.indexes: list[dict[str, Any]] = []

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append((query, params))
        if "SHOW INDEXES" in query:
            return [dict(index) for index in self.indexes]
        if query.lstrip().startswith(f"DROP INDEX {VECTOR_INDEX_NAME}"):
            self.indexes = []
            return []
        if VECTOR_INDEX_NAME in query and "CREATE VECTOR INDEX" in query:
            import re

            match = re.search(r"vector\.dimensions`: (\d+)", query)
            dim = int(match.group(1)) if match else 1024
            self.indexes.append(
                {
                    "name": VECTOR_INDEX_NAME,
                    "options": {"indexConfig": {"vector.dimensions": dim}},
                }
            )
        return []

    def run_batches(
        self, query: str, rows: Sequence[dict[str, Any]], *, batch_size: int = 5000, label: str = ""
    ) -> int:
        for start in range(0, len(rows), batch_size):
            self.run(query, rows=list(rows[start : start + batch_size]))
        return len(rows)

    def queries_matching(self, needle: str) -> list[tuple[str, dict[str, Any]]]:
        return [entry for entry in self.queries if needle in entry[0]]


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


# --------------------------------------------------------------------------- #
# ids
# --------------------------------------------------------------------------- #
def test_node_ids_are_namespaced() -> None:
    assert ids.movie_node_id(1) == "ml_1"
    assert ids.game_node_id(730) == "st_730"
    assert ids.user_node_id("movie", 7) == "ml_7"
    assert ids.user_node_id("game", 7) == "st_7"
    with pytest.raises(ValueError):
        ids.user_node_id("nope", 1)


def test_item_node_id_round_trip() -> None:
    node_id = ids.item_node_id("game", 440)
    assert node_id == "st_440"
    assert ids.parse_item_id(node_id) == ("game", 440)
    assert ids.parse_item_id(ids.movie_node_id(12)) == ("movie", 12)
    with pytest.raises(ValueError):
        ids.parse_item_id("xx_3")


def test_slugified_taxonomy_ids_are_stable_and_unique() -> None:
    sci_fi = ids.genre_node_id("Sci-Fi")
    assert sci_fi == "gn_sci_fi"
    assert sci_fi == ids.genre_node_id("Sci-fi")
    all_ids = [
        ids.genre_node_id("Sci-Fi"),
        ids.theme_node_id("great soundtrack"),
        ids.platform_node_id("Windows"),
    ]
    assert len(set(all_ids)) == 3


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_apply_schema_is_idempotent(fake_client: FakeClient) -> None:
    apply_schema(fake_client, dim=1024)
    apply_schema(fake_client, dim=1024)
    # `IF NOT EXISTS` alone makes it harmless to repeat.
    assert len(fake_client.queries_matching("CREATE CONSTRAINT")) == 2 * len(UNIQUENESS_CONSTRAINTS)
    assert all("IF NOT EXISTS" in q for q, _ in fake_client.queries_matching("CREATE CONSTRAINT"))


def test_apply_schema_recreates_index_when_dimension_changes(fake_client: FakeClient) -> None:
    apply_schema(fake_client, dim=1024)
    apply_schema(fake_client, dim=768)
    assert len(fake_client.queries_matching(f"DROP INDEX {VECTOR_INDEX_NAME}")) == 1
    assert len(fake_client.queries_matching("CREATE VECTOR INDEX")) == 2  # two CREATEs
    assert fake_client.indexes[0]["options"]["indexConfig"]["vector.dimensions"] == 768


# --------------------------------------------------------------------------- #
# SIMILAR_TO computation
# --------------------------------------------------------------------------- #
def test_top_k_similar_excludes_self_and_ranks_correctly() -> None:
    # three documents: A, B identical, C orthogonal
    matrix = np.array(

        [[0.6, 0.8, 0.0], [0.6, 0.8, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32
    )
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norm
    rows, cols, scores = top_k_similar(matrix, k=3)
    pairs = {(int(r), int(c)) for r, c in zip(rows, cols, strict=True)}
    assert (0, 0) not in pairs and (1, 1) not in pairs and (2, 2) not in pairs
    by_src: dict[int, list[tuple[int, float]]] = {}
    for r, c, s in zip(rows, cols, scores, strict=True):
        by_src.setdefault(int(r), []).append((int(c), float(s)))
    for _, neighbours in by_src.items():
        ordered = [c for c, _ in neighbours]
        assert ordered == sorted(ordered, key=lambda c: -dict(neighbours)[c])
        assert len(neighbours) == 2  # only the other two items exist


def test_top_k_similar_empty_input() -> None:
    rows, cols, scores = top_k_similar(np.zeros((0, 4), dtype=np.float32), k=5)
    assert rows.size == cols.size == scores.size == 0


def test_build_similar_to_frame_cross_domain_counts() -> None:
    items = pd.DataFrame(
        {
            "domain": ["movie", "movie", "game", "game"],
            "item_id": [1, 2, 10, 11],
        }
    )
    # rows sort by (domain, item_id): movie 1, movie 2, game 10, game 11
    matrix = np.eye(4, dtype=np.float32)
    # make movie1 <-> movie2 and movie1 <-> game10 similar; the rest orthogonal
    matrix[0, 1] = 0.8
    matrix[1, 0] = 0.8
    matrix[0, 2] = 0.7
    matrix[2, 0] = 0.7
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / norm
    frame, stats = build_similar_to_frame(items, matrix, k=2)
    assert stats.n_edges == len(frame) > 0
    # every source has k neighbours
    assert (frame.groupby("src_id").size() == 2).all()
    # the best movie1 neighbour is movie2 (0.8) then game10 (0.7)
    src_movie1 = frame[(frame.src_domain == "movie") & (frame.src_id == 1)]
    assert list(src_movie1.src_id) == list(src_movie1.src_id)  # sanity
    top = src_movie1.sort_values("rank").dst_id.tolist()
    assert top[0] == 2 and 10 in top
    cross = (frame.src_domain != frame.dst_domain).sum()
    assert stats.cross_domain_edges == int(cross) and cross >= 2


def test_compute_similar_to_writes_parquet(
    make_config: Callable[..., Any],
) -> None:
    import entdec.graph.similar as similar_mod

    cfg = make_config("tiny")
    processed = cfg.paths.processed
    rng = np.random.default_rng(0)
    items = pd.DataFrame(
        {
            "domain": ["movie"] * 5 + ["game"] * 5,
            "item_id": list(range(1, 6)) + list(range(100, 105)),
        }
    )
    processed.mkdir(parents=True, exist_ok=True)
    frame = items.copy()
    vectors = rng.normal(size=(10, 8)).astype(np.float32)
    vector_frame = pd.DataFrame(
        {
            "domain": items["domain"],
            "item_id": frame["item_id"],
            "embedding": list(vectors),
        }
    )
    vector_frame.to_parquet(processed / "item_embeddings.parquet", index=False)
    path_without = processed / "item_embeddings.parquet"

    original_load = similar_mod.load_item_embeddings
    invoked = {"called": False}

    def fake_load(cfg: Any) -> tuple[pd.DataFrame, np.ndarray]:
        invoked["called"] = True
        loaded = pd.read_parquet(path_without, columns=["domain", "item_id", "embedding"])
        return loaded, np.vstack(loaded["embedding"].to_numpy())

    similar_mod.load_item_embeddings = fake_load  # type: ignore[method-assign]
    try:
        path, stats = similar_mod.compute_similar_to(cfg, force=True, k=3)
    finally:
        similar_mod.load_item_embeddings = original_load  # type: ignore[method-assign]
    assert invoked["called"] and path.name == "similar_to.parquet"
    assert stats.n_edges == 30  # 10 items x k=3
    # second run reuses the parquet
    path2, stats2 = similar_mod.compute_similar_to(cfg)
    assert path2 == path and stats2.n_edges == 30


# --------------------------------------------------------------------------- #
# query templates
# --------------------------------------------------------------------------- #
def test_templates_are_parameterized_and_named() -> None:
    templates_map = templates()
    assert {"item_ids_by_theme", "item_ids_similar_to", "vector_query", "shared_themes"} <= set(
        templates_map
    )
    for template in templates_map.values():
        assert template.name and template.cypher


def test_vector_query_params_shape() -> None:
    params = vector_query_params([0.1, 0.2], k=10, domain="movie")
    assert params == {
        "embedding": [0.1, 0.2],
        "k": 10,
        "domain": "movie",
        "index": "item_embedding",
    }
