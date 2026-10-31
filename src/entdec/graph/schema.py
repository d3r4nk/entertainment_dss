"""Graph schema: uniqueness constraints and the item embedding vector index.

The embedding dimension is taken from the artefacts, so a change of
``nlp.embed_dim`` rebuilds the index with the right size.
"""

from __future__ import annotations

from dataclasses import dataclass

from entdec.graph.neo4j_client import Neo4jClient
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

ITEM_LABELS = ("Movie", "Game")
VECTOR_INDEX_NAME = "item_embedding"
VECTOR_SIMILARITY = "cosine"
DEFAULT_EMBED_DIM = 1024

# label -> property that must be unique
# `Item` carries the constraint that matters for relationship loading (every
# item lookup goes through `MATCH (i:Item {id: ...})`, which needs an index on
# that label); Movie/Game keep their own so a `MERGE (m:Movie {id: ...})` is
# indexed too. The ids are namespaced (`ml_`/`st_`), so the three cannot clash.
UNIQUENESS_CONSTRAINTS: dict[str, str] = {
    "Item": "id",
    "Movie": "id",
    "Game": "id",
    "User": "id",
    "Genre": "id",
    "Theme": "id",
    "Platform": "id",
}

CONSTRAINT_TEMPLATE = "CREATE CONSTRAINT {name} IF NOT EXISTS FOR (n:`{label}`) REQUIRE n.{prop} IS UNIQUE"

VECTOR_INDEX_TEMPLATE = """
CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR (n:Item) ON (n.embedding)
OPTIONS {{indexConfig: {{`vector.dimensions`: {dim}, `vector.similarity_function`: '{similarity}'}}}}
"""


@dataclass
class SchemaReport:
    """What the schema step created."""

    constraints: list[str]
    vector_index: str
    dim: int
    rebuilt: bool = False


def apply_schema(client: Neo4jClient, dim: int = DEFAULT_EMBED_DIM, *, rebuild_index: bool = False) -> SchemaReport:
    """Create the constraints and the vector index (idempotent).

    Args:
        client: Connected Neo4j client.
        dim: Embedding dimension for the vector index.
        rebuild_index: Drop and recreate the vector index (needed after a
            dimension change).

    Returns:
        A :class:`SchemaReport`.
    """
    created: list[str] = []
    for label, prop in UNIQUENESS_CONSTRAINTS.items():
        name = f"{label.lower()}_{prop}_unique"
        client.run(CONSTRAINT_TEMPLATE.format(name=name, label=label, prop=prop))
        created.append(name)

    current = list(
        client.run(
            "SHOW INDEXES YIELD name, type, options "
            "WHERE name = $name RETURN name, options",
            name=VECTOR_INDEX_NAME,
        )
    )
    existing_dim = None
    if current:
        options = current[0].get("options") or {}
        config = options.get("indexConfig") or options.get("index_config") or {}
        existing_dim = config.get("vector.dimensions") or config.get("vector_dimensions")

    rebuilt = False
    if current and existing_dim and int(existing_dim) != int(dim):
        logger.warning(
            "vector index dimension %s != requested %d - dropping and recreating",
            existing_dim,
            dim,
        )
        client.run(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")
        current = []
        rebuilt = True
    elif current and rebuild_index:
        client.run(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")
        current = []
        rebuilt = True

    if not current:
        client.run(
            VECTOR_INDEX_TEMPLATE.format(
                name=VECTOR_INDEX_NAME, dim=int(dim), similarity=VECTOR_SIMILARITY
            )
        )
        logger.info("created vector index %s (dim=%d, %s)", VECTOR_INDEX_NAME, dim, VECTOR_SIMILARITY)

    logger.info("schema applied: %d constraints, vector index %s", len(created), VECTOR_INDEX_NAME)
    return SchemaReport(constraints=created, vector_index=VECTOR_INDEX_NAME, dim=int(dim), rebuilt=rebuilt)


def describe_schema(client: Neo4jClient) -> dict[str, object]:
    """Return the constraints and indexes currently in the database."""
    constraints = client.run("SHOW CONSTRAINTS YIELD name, type, labelsOrTypes, properties RETURN *")
    indexes = client.run("SHOW INDEXES YIELD name, type, state, labelsOrTypes RETURN *")
    return {"constraints": constraints, "indexes": indexes}


def reset_database(client: Neo4jClient, batch_size: int = 10_000) -> int:
    """Delete every node and relationship; returns the number of nodes removed."""
    total = int(client.run("MATCH (n) RETURN count(n) AS c")[0]["c"])
    if total == 0:
        return 0
    logger.info("deleting %d nodes in batches of %d", total, batch_size)
    while True:
        deleted = client.run(
            "MATCH (n) WITH n LIMIT $limit DETACH DELETE n RETURN count(*) AS deleted",
            limit=batch_size,
        )[0]["deleted"]
        if not deleted:
            break
    logger.info("database reset (%d nodes removed)", total)
    return total


__all__ = [
    "DEFAULT_EMBED_DIM",
    "ITEM_LABELS",
    "UNIQUENESS_CONSTRAINTS",
    "VECTOR_INDEX_NAME",
    "VECTOR_INDEX_TEMPLATE",
    "VECTOR_SIMILARITY",
    "SchemaReport",
    "apply_schema",
    "describe_schema",
    "reset_database",
]
