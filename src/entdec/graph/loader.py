"""Idempotent Neo4j loader and the ``graph`` CLI step.

Every write is a ``MERGE`` inside an ``UNWIND $rows`` batch, so running the loader
twice changes nothing. Steps:

1. ``init``   - constraints + vector index
2. ``load``   - nodes (genre/theme/platform/item/user) and edges
   (``HAS_GENRE``, ``HAS_THEME``, ``ON_PLATFORM``, ``RATED``, ``REVIEWED``)
3. embeddings - the item vectors, in smaller batches
4. ``similar`` - offline cosine top-10 -> ``SIMILAR_TO`` edges
5. ``verify``  - node/edge counts and sanity queries
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from entdec.config import Config
from entdec.graph import ids
from entdec.graph.neo4j_client import (
    EMBEDDING_BATCH_SIZE,
    Neo4jClient,
    compose_down,
    compose_up,
)
from entdec.graph.schema import apply_schema, reset_database
from entdec.graph.similar import compute_similar_to
from entdec.graph.store import (
    GraphStoreError,
    current_marker,
    dump_profile,
    graph_status,
    preserve_current_graph,
    require_loaded_profile,
    restore_profile,
    use_profile,
    write_graph_marker,
)
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_THEME_WEIGHT = 1.0
CROSS_DOMAIN_SIMILARITY_FLOOR = 0.0

# --------------------------------------------------------------------------- #
# Cypher: node upserts
# --------------------------------------------------------------------------- #
MERGE_MOVIES = """
UNWIND $rows AS row
MERGE (m:Movie {id: row.id})
SET m:Item,
    m.movie_id = row.movie_id,
    m.title = row.title,
    m.year = row.year,
    m.rating_avg = row.rating_avg,
    m.n_ratings = row.n_ratings
"""

MERGE_GAMES = """
UNWIND $rows AS row
MERGE (g:Game {id: row.id})
SET g:Item,
    g.app_id = row.app_id,
    g.title = row.title,
    g.year = row.year,
    g.price = row.price,
    g.positive_ratio = row.positive_ratio,
    g.n_reviews = row.n_reviews
"""

MERGE_GENRES = """
UNWIND $rows AS row
MERGE (n:Genre {id: row.id})
SET n.name = row.name
"""

MERGE_THEMES = """
UNWIND $rows AS row
MERGE (n:Theme {id: row.id})
SET n.name = row.name
"""

MERGE_PLATFORMS = """
UNWIND $rows AS row
MERGE (n:Platform {id: row.id})
SET n.name = row.name
"""

MERGE_USERS = """
UNWIND $rows AS row
MERGE (u:User {id: row.id})
SET u.uid = row.uid,
    u.domain = row.domain
"""

MERGE_EMBEDDINGS = """
UNWIND $rows AS row
MATCH (i:Item {id: row.item_id})
SET i.embedding = row.embedding
"""

# --------------------------------------------------------------------------- #
# Cypher: edges
# --------------------------------------------------------------------------- #
MERGE_HAS_GENRE = """
UNWIND $rows AS row
MATCH (i:Item {id: row.item_id})
MATCH (g:Genre {id: row.genre_id})
MERGE (i)-[:HAS_GENRE]->(g)
"""

MERGE_HAS_THEME = """
UNWIND $rows AS row
MATCH (i:Item {id: row.item_id})
MATCH (t:Theme {id: row.theme_id})
MERGE (i)-[r:HAS_THEME]->(t)
SET r.weight = row.weight
"""

MERGE_ON_PLATFORM = """
UNWIND $rows AS row
MATCH (i:Item {id: row.item_id})
MATCH (p:Platform {id: row.platform_id})
MERGE (i)-[:ON_PLATFORM]->(p)
"""

MERGE_RATED = """
UNWIND $rows AS row
MATCH (u:User {id: row.user_id})
MATCH (m:Movie {id: row.movie_id})
MERGE (u)-[r:RATED]->(m)
SET r.score = row.score,
    r.ts = row.ts
"""

MERGE_REVIEWED = """
UNWIND $rows AS row
MATCH (u:User {id: row.user_id})
MATCH (g:Game {id: row.game_id})
MERGE (u)-[r:REVIEWED]->(g)
SET r.recommended = row.recommended,
    r.hours = row.hours,
    r.ts = row.ts
"""

MERGE_SIMILAR_TO = """
UNWIND $rows AS row
MATCH (a:Item {id: row.src_id})
MATCH (b:Item {id: row.dst_id})
MERGE (a)-[r:SIMILAR_TO]->(b)
SET r.score = row.score,
    r.rank = row.rank
"""

# Node/edge counts reported by `verify`.
COUNT_NODES = """
MATCH (n)
RETURN head(labels(n)) AS label, count(*) AS count
ORDER BY count DESC
"""

COUNT_EDGES = """
MATCH ()-[r]->()
RETURN type(r) AS type, count(*) AS count
ORDER BY count DESC
"""


@dataclass
class LoadStats:
    """Row counts produced by one load."""

    nodes: dict[str, int] = field(default_factory=dict)
    edges: dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0

    def total_nodes(self) -> int:
        return sum(self.nodes.values())

    def total_edges(self) -> int:
        return sum(self.edges.values())


# --------------------------------------------------------------------------- #
# typed row builders
# --------------------------------------------------------------------------- #
def _clean(value: Any) -> Any:
    """Convert numpy/pandas scalars into driver-friendly Python values."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def movie_rows(cfg: Config) -> list[dict[str, Any]]:
    """Movie node rows."""
    frame = pd.read_parquet(cfg.paths.processed / "movies.parquet")
    return [
        {
            "id": ids.movie_node_id(row.movie_id),
            "movie_id": int(row.movie_id),
            "title": str(row.title),
            "year": int(row.year),
            "rating_avg": float(row.rating_avg),
            "n_ratings": int(row.n_ratings),
        }
        for row in frame.itertuples(index=False)
    ]


def game_rows(cfg: Config) -> list[dict[str, Any]]:
    """Game node rows."""
    frame = pd.read_parquet(cfg.paths.processed / "games.parquet")
    return [
        {
            "id": ids.game_node_id(row.app_id),
            "app_id": int(row.app_id),
            "title": str(row.title),
            "year": int(row.year),
            "price": float(row.price),
            "positive_ratio": int(row.positive_ratio),
            "n_reviews": int(row.n_reviews),
        }
        for row in frame.itertuples(index=False)
    ]


def taxonomy_rows(cfg: Config) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Genre/theme/platform node rows plus the matching ``item_id`` link rows."""
    processed = cfg.paths.processed
    genres = pd.read_parquet(processed / "item_genres.parquet")
    genre_nodes = [
        {"id": ids.genre_node_id(name), "name": str(name)}
        for name in sorted(genres["genre"].unique())
    ]
    genre_links: list[dict[str, Any]] = []
    for row in genres.itertuples(index=False):
        genre_links.append(
            {
                "item_id": ids.item_node_id(row.domain, int(row.item_id)),
                "genre_id": ids.genre_node_id(row.genre),
            }
        )

    theme_frames = [
        pd.read_parquet(processed / "movie_themes.parquet")
        .rename(columns={"movie_id": "item_id"})
        .assign(domain="movie"),
        pd.read_parquet(processed / "game_themes.parquet")
        .rename(columns={"app_id": "item_id"})
        .assign(domain="game"),
    ]
    themes = pd.concat(theme_frames, ignore_index=True)
    theme_nodes = [
        {"id": ids.theme_node_id(name), "name": str(name)}
        for name in sorted(themes["theme"].unique())
    ]
    theme_links = [
        {
            "item_id": ids.item_node_id(row.domain, row.item_id),
            "theme_id": ids.theme_node_id(row.theme),
            "weight": float(row.weight),
        }
        for row in themes.itertuples(index=False)
    ]

    games = pd.read_parquet(processed / "games.parquet", columns=["app_id", "platforms"])
    platform_nodes: dict[str, dict[str, Any]] = {}
    platform_links: list[dict[str, Any]] = []
    for row in games.itertuples(index=False):
        for platform in row.platforms if row.platforms is not None else []:
            name = str(platform)
            platform_nodes[name] = {"id": ids.platform_node_id(name), "name": name}
            platform_links.append(
                {
                    "item_id": ids.game_node_id(row.app_id),
                    "platform_id": ids.platform_node_id(name),
                }
            )

    link_rows = {
        "HAS_GENRE": genre_links,
        "HAS_THEME": theme_links,
        "ON_PLATFORM": platform_links,
    }
    nodes = genre_nodes + theme_nodes + list(platform_nodes.values())
    return nodes, link_rows


def user_rows(cfg: Config) -> list[dict[str, Any]]:
    """User node rows for both domains."""
    rows: list[dict[str, Any]] = []
    processed = cfg.paths.processed
    movies_path = processed / "movie_ratings.parquet"
    games_path = processed / "game_reviews.parquet"
    if movies_path.exists():
        frame = pd.read_parquet(movies_path, columns=["uid"])
        for uid in sorted(frame["uid"].unique().tolist()):
            rows.append({"id": ids.user_node_id("movie", uid), "uid": int(uid), "domain": "movie"})
    if games_path.exists():
        frame = pd.read_parquet(games_path, columns=["uid"])
        for uid in sorted(frame["uid"].unique().tolist()):
            rows.append({"id": ids.user_node_id("game", uid), "uid": int(uid), "domain": "game"})
    return rows


def interaction_rows(cfg: Config) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``RATED`` and ``REVIEWED`` edge rows."""
    processed = cfg.paths.processed
    rated: list[dict[str, Any]] = []
    reviewed: list[dict[str, Any]] = []

    movies_path = processed / "movie_ratings.parquet"
    if movies_path.exists():
        frame = pd.read_parquet(movies_path, columns=["uid", "movie_id", "rating", "ts"])
        rated = [
            {
                "user_id": ids.user_node_id("movie", row.uid),
                "movie_id": ids.movie_node_id(row.movie_id),
                "score": float(row.rating),
                "ts": int(row.ts),
            }
            for row in frame.itertuples(index=False)
        ]

    games_path = processed / "game_reviews.parquet"
    if games_path.exists():
        frame = pd.read_parquet(
            games_path, columns=["uid", "app_id", "is_recommended", "hours", "ts"]
        )
        reviewed = [
            {
                "user_id": ids.user_node_id("game", row.uid),
                "game_id": ids.game_node_id(row.app_id),
                "recommended": bool(row.is_recommended),
                "hours": float(row.hours),
                "ts": int(row.ts),
            }
            for row in frame.itertuples(index=False)
        ]
    return rated, reviewed


def embedding_rows(cfg: Config) -> list[dict[str, Any]]:
    """Item embedding rows (one python list per item)."""
    frame = pd.read_parquet(
        cfg.paths.processed / "item_embeddings.parquet",
        columns=["domain", "item_id", "embedding"],
    )
    return [
        {
            "item_id": ids.item_node_id(row.domain, row.item_id),
            "embedding": [float(value) for value in np.asarray(row.embedding).tolist()],
        }
        for row in frame.itertuples(index=False)
    ]


def similar_rows(cfg: Config, force: bool = False) -> list[dict[str, Any]]:
    """``SIMILAR_TO`` edge rows (computing the parquet first when needed)."""
    path, _ = compute_similar_to(cfg, force=force)
    frame = pd.read_parquet(path)
    return [
        {
            "src_id": ids.item_node_id(row.src_domain, row.src_id),
            "dst_id": ids.item_node_id(row.dst_domain, row.dst_id),
            "score": float(row.score),
            "rank": int(row.rank),
        }
        for row in frame.itertuples(index=False)
    ]


# --------------------------------------------------------------------------- #
# load
# --------------------------------------------------------------------------- #
def load_all(
    cfg: Config,
    client: Neo4jClient,
    *,
    batch_size: int | None = None,
    with_similar: bool = True,
    force_similar: bool = False,
) -> LoadStats:
    """Load the whole graph (idempotent)."""
    started = time.perf_counter()
    batch = int(batch_size or cfg.get("graph.batch_size", 5000))
    stats = LoadStats()

    dim = int(cfg.get("nlp.embed_dim", 1024))
    embeddings = embedding_rows(cfg)
    if embeddings:
        dim = len(embeddings[0]["embedding"])
    apply_schema(client, dim=dim)

    taxonomy_nodes, link_rows = taxonomy_rows(cfg)
    stats.nodes["Genre"] = client.run_batches(
        MERGE_GENRES, [row for row in taxonomy_nodes if row["id"].startswith(ids.GENRE_PREFIX)],
        batch_size=batch, label="Genre",
    )
    stats.nodes["Theme"] = client.run_batches(
        MERGE_THEMES, [row for row in taxonomy_nodes if row["id"].startswith(ids.THEME_PREFIX)],
        batch_size=batch, label="Theme",
    )
    stats.nodes["Platform"] = client.run_batches(
        MERGE_PLATFORMS, [row for row in taxonomy_nodes if row["id"].startswith(ids.PLATFORM_PREFIX)],
        batch_size=batch, label="Platform",
    )

    stats.nodes["Movie"] = client.run_batches(MERGE_MOVIES, movie_rows(cfg), batch_size=batch, label="Movie")
    stats.nodes["Game"] = client.run_batches(MERGE_GAMES, game_rows(cfg), batch_size=batch, label="Game")
    stats.nodes["User"] = client.run_batches(MERGE_USERS, user_rows(cfg), batch_size=batch, label="User")

    stats.edges["HAS_GENRE"] = client.run_batches(
        MERGE_HAS_GENRE, link_rows["HAS_GENRE"], batch_size=batch * 2, label="HAS_GENRE"
    )
    stats.edges["HAS_THEME"] = client.run_batches(
        MERGE_HAS_THEME, link_rows["HAS_THEME"], batch_size=batch * 2, label="HAS_THEME"
    )
    stats.edges["ON_PLATFORM"] = client.run_batches(
        MERGE_ON_PLATFORM, link_rows["ON_PLATFORM"], batch_size=batch * 2, label="ON_PLATFORM"
    )

    rated, reviewed = interaction_rows(cfg)
    stats.edges["RATED"] = client.run_batches(MERGE_RATED, rated, batch_size=batch, label="RATED")
    stats.edges["REVIEWED"] = client.run_batches(
        MERGE_REVIEWED, reviewed, batch_size=batch, label="REVIEWED"
    )

    # Embeddings are heavy rows: many floats each, so smaller batches.
    stats.nodes["embeddings"] = client.run_batches(
        MERGE_EMBEDDINGS,
        embeddings,
        batch_size=min(batch, EMBEDDING_BATCH_SIZE),
        label="item embeddings",
    )

    if with_similar:
        rows = similar_rows(cfg, force=force_similar)
        stats.edges["SIMILAR_TO"] = client.run_batches(
            MERGE_SIMILAR_TO, rows, batch_size=batch, label="SIMILAR_TO"
        )

    stats.duration_s = time.perf_counter() - started
    marker = write_graph_marker(client, cfg)
    logger.info(
        "graph loaded: %d node rows, %d edge rows in %.1fs",
        stats.total_nodes(),
        stats.total_edges(),
        stats.duration_s,
    )
    logger.info(
        "graph marker written: profile=%s nodes=%d relationships=%d",
        marker["profile"],
        marker["counts"]["node_count"],
        marker["counts"]["relationship_count"],
    )
    return stats


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #
def verify(client: Neo4jClient, *, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    """Print node/edge counts and run three sanity queries.

    The sanity checks are:

    1. every item carries an embedding, and the vector index answers a query;
    2. themes are genuinely shared (some ``Theme`` nodes link both movies and games);
    3. users are connected (every user has at least one interaction edge).

    Returns:
        A dict with the counts and the check results.
    """
    nodes = {row["label"]: row["count"] for row in client.run(COUNT_NODES)}
    edges = {row["type"]: row["count"] for row in client.run(COUNT_EDGES)}
    logger.info("nodes: %s", nodes)
    logger.info("edges: %s", edges)

    indexes = client.run(
        "SHOW INDEXES YIELD name, type, state, labelsOrTypes, properties RETURN *"
    )
    vector_index = next(
        (row for row in indexes if row.get("name") == "item_embedding"), None
    )
    logger.info("vector index: %s", vector_index)

    checks: dict[str, Any] = {}

    # --- check 1: embedding coverage + vector index actually answers --------
    coverage = client.run(
        "MATCH (i:Item) RETURN count(i) AS items, "
        "count(i.embedding) AS with_embedding"
    )[0]
    probe = client.run("MATCH (i:Item) WHERE i.embedding IS NOT NULL RETURN i.embedding AS v LIMIT 1")
    neighbours: list[dict[str, Any]] = []
    if probe:
        neighbours = client.run(
            "CALL db.index.vector.queryNodes('item_embedding', 5, $embedding) "
            "YIELD node, score RETURN node.id AS id, score",
            embedding=probe[0]["v"],
        )
    checks["embeddings"] = {
        "items": coverage["items"],
        "with_embedding": coverage["with_embedding"],
        "vector_index_state": (vector_index or {}).get("state"),
        "vector_search_results": len(neighbours),
        "top_hit": neighbours[0] if neighbours else None,
    }
    logger.info(
        "check 1 (embeddings): %d/%d items embedded, index state %s, vector search returned %d rows",
        coverage["with_embedding"],
        coverage["items"],
        (vector_index or {}).get("state"),
        len(neighbours),
    )

    # --- check 2: shared themes across domains ------------------------------
    shared = client.run(
        "MATCH (t:Theme)<-[:HAS_THEME]-(m:Movie) "
        "MATCH (t)<-[:HAS_THEME]-(g:Game) "
        "RETURN count(DISTINCT t) AS shared_themes"
    )[0]["shared_themes"]
    total_themes = nodes.get("Theme", 0)
    checks["shared_themes"] = {"total": total_themes, "shared": int(shared)}
    logger.info("check 2 (shared themes): %d of %d themes link both domains", shared, total_themes)

    # --- check 3: every user has an interaction -----------------------------
    orphans = client.run(
        "MATCH (u:User) WHERE NOT (u)-[:RATED|REVIEWED]->() "
        "RETURN count(u) AS orphans"
    )[0]["orphans"]
    users = nodes.get("User", 0)
    checks["users"] = {"total": users, "without_interactions": int(orphans)}
    logger.info("check 3 (users): %d users, %d without any interaction", users, orphans)

    cross_domain_similar = edges.get("SIMILAR_TO", 0)
    checks["similar_to_edges"] = cross_domain_similar
    if expected:
        mismatches = {
            key: (value, nodes.get(key))
            for key, value in expected.items()
            if key in nodes and nodes.get(key) != value
        }
        checks["expected_mismatches"] = mismatches
        if mismatches:
            logger.warning("count mismatches vs expectation: %s", mismatches)

    ok = (
        coverage["items"] > 0
        and coverage["with_embedding"] == coverage["items"]
        and (vector_index or {}).get("state") == "ONLINE"
        and len(neighbours) > 0
        and int(orphans) == 0
        and int(shared) > 0
    )
    checks["ok"] = ok
    logger.info("graph verification: %s", "PASSED" if ok else "FAILED")
    return {"nodes": nodes, "edges": edges, "checks": checks}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run(cfg: Config, args: Any = None, client: Neo4jClient | None = None) -> int:
    """Dispatch the ``graph`` sub-actions."""
    action = getattr(args, "action", "load")
    force = bool(getattr(args, "force", False))

    if action == "up":
        return 0 if compose_up() else 1
    if action == "down":
        return 0 if compose_down(volumes=False) else 1
    try:
        if action == "status":
            graph_status(cfg, client=client)
            return 0
        if action == "use":
            use_profile(cfg, client=client, force=force)
            return 0
        if action == "dump":
            dump_profile(cfg, cfg.profile.name, force=force, client=client)
            return 0
        if action == "restore":
            restore_profile(cfg, cfg.profile.name, client=client, force=force)
            return 0
    except (GraphStoreError, OSError, subprocess.SubprocessError) as exc:
        logger.error("%s", exc)
        return 1

    own_client = client is None
    active = client or Neo4jClient.from_config(cfg)
    try:
        if not active.is_ready() and not active.wait_until_ready(timeout_s=20):
            logger.error(
                "neo4j is not reachable at %s - start it with `docker compose up -d` "
                "or `python -m entdec graph up`",
                cfg.env.neo4j_uri,
            )
            return 1

        if action == "reset":
            marker = current_marker(cfg, active)
            if marker and active.is_ready():
                preserve_current_graph(cfg, active, marker, force=force)
            reset_database(active)
            return 0
        if action == "init":
            apply_schema(active, dim=int(cfg.get("nlp.embed_dim", 1024)))
            return 0
        if action == "similar":
            path, stats = compute_similar_to(cfg, force=force)
            logger.info("wrote %s (%d edges, %d cross-domain)", path.name, stats.n_edges, stats.cross_domain_edges)
            return 0
        if action == "verify":
            result = verify(active)
            return 0 if result["checks"]["ok"] else 1
        if action == "load":
            marker = current_marker(cfg, active)
            if marker:
                require_loaded_profile(marker, cfg.profile.name, "load")
            load_all(cfg, active, force_similar=force)
            result = verify(active)
            return 0 if result["checks"]["ok"] else 1
        logger.error("unknown graph action %r", action)
        return 2
    except (GraphStoreError, OSError, subprocess.SubprocessError) as exc:
        logger.error("%s", exc)
        return 1
    finally:
        if own_client:
            active.close()


__all__: Sequence[str] = (
    "LoadStats",
    "game_rows",
    "interaction_rows",
    "load_all",
    "movie_rows",
    "run",
    "similar_rows",
    "taxonomy_rows",
    "user_rows",
    "verify",
)
