"""Integration tests: real Docker Neo4j (bolt://localhost:7688, profile `tiny`).

Run explicitly with `pytest -m integration`; `pytest -m "not integration"` skips
them. Ollama is not needed here because the item embeddings are already in
`data/processed/<profile>/item_embeddings.parquet`.
"""

from __future__ import annotations

from typing import Any

import pytest

from entdec.config import load_config
from entdec.graph.neo4j_client import Neo4jClient
from entdec.graph.schema import VECTOR_INDEX_NAME

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def client() -> Any:
    cfg = load_config()
    neo = Neo4jClient.from_config(cfg)
    if not neo.is_ready():
        pytest.skip("Neo4j is not running - start it with `python -m entdec graph up`")
    neo.close()
    from entdec.graph.store import use_profile

    use_profile(cfg)
    neo = Neo4jClient.from_config(cfg)
    yield neo
    neo.close()


def test_counts_match_tiny_profile(client: Neo4jClient) -> None:
    items = client.run(
        "MATCH (m:Movie) WITH count(m) AS movies "
        "MATCH (g:Game) RETURN movies, count(g) AS games"
    )[0]
    assert items["movies"] == 500 and items["games"] == 500
    users = client.run(
        "MATCH (u:User) WHERE u.domain = 'movie' WITH count(u) AS movie_users "
        "MATCH (u:User) WHERE u.domain = 'game' RETURN movie_users, count(u) AS game_users"
    )[0]
    assert users["movie_users"] == 1000 and users["game_users"] == 1000


def test_every_item_has_embedding_and_vector_index_answers(client: Neo4jClient) -> None:
    row = client.run(
        "MATCH (i:Item) RETURN count(i) AS items, count(i.embedding) AS embedded"
    )[0]
    assert row["items"] == 1000 and row["embedded"] == 1000

    index = client.run("SHOW INDEXES YIELD name, state WHERE name = $n RETURN name, state", n=VECTOR_INDEX_NAME)
    assert index and index[0]["state"] == "ONLINE"

    probe = client.run("MATCH (i:Item) RETURN i.embedding AS v LIMIT 1")[0]["v"]
    hits = client.run(
        "CALL db.index.vector.queryNodes($index, 5, $v) YIELD node, score RETURN node.id AS id, score",
        index=VECTOR_INDEX_NAME, v=probe,
    )
    assert len(hits) == 5 and all(h["score"] > 0 for h in hits)


def test_edges_exist_and_shared_themes_span_domains(client: Neo4jClient) -> None:
    for rel in ("HAS_GENRE", "HAS_THEME", "ON_PLATFORM", "RATED", "REVIEWED", "SIMILAR_TO"):
        n = client.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS n")[0]["n"]
        assert n > 0, f"{rel} is empty"

    shared = client.run(
        "MATCH (t:Theme)<-[:HAS_THEME]-(:Movie) MATCH (t)<-[:HAS_THEME]-(:Game) "
        "RETURN count(DISTINCT t) AS n"
    )[0]["n"]
    assert shared > 0, "no theme is shared by both domains"


def test_similar_to_has_10_per_item_and_cross_domain_edges(client: Neo4jClient) -> None:
    worst = client.run(
        "MATCH (i:Item) OPTIONAL MATCH (i)-[:SIMILAR_TO]->() "
        "WITH i, count(*) AS k RETURN min(k) AS min_k"
    )[0]["min_k"]
    assert worst >= 10 or worst is None
    cross = client.run(
        "MATCH (a)-[r:SIMILAR_TO]->(b) WHERE (a:Movie AND b:Game) OR (a:Game AND b:Movie) "
        "RETURN count(r) AS n"
    )[0]["n"]
    assert cross > 0, "no cross-domain SIMILAR_TO edge"


def test_loader_is_idempotent(client: Neo4jClient) -> None:
    from entdec.graph.loader import load_all, verify

    before = client.run(
        "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS rels"
    )[0]
    cfg = load_config()
    load_all(cfg, client)
    after = client.run(
        "MATCH (n) WITH count(n) AS nodes MATCH ()-[r]->() RETURN nodes, count(r) AS rels"
    )[0]
    assert before == after, f"loading twice changed counts: {before} -> {after}"
    result = verify(client)
    assert result["checks"]["ok"] is True
