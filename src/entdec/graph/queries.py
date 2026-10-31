"""Parameterized Cypher templates.

Everything the API, the chatbot and the recommenders read from the graph goes
through these constants. Values are always passed as ``$parameters`` - the LLM
never writes Cypher in v1 (PROJECT_SPEC.md section 8, step 2), it only produces a
validated intent object that is translated into these templates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Template:
    """A named, documented Cypher statement."""

    name: str
    cypher: str
    description: str


# --------------------------------------------------------------------------- #
# catalog
# --------------------------------------------------------------------------- #
ALL_ITEM_TITLES = Template(
    name="all_item_titles",
    description="Every catalogue title, for rapidfuzz reference resolution.",
    cypher="""
        MATCH (i:Item)
        RETURN i.id AS id, i.title AS title, i.year AS year,
               CASE WHEN i:Movie THEN 'movie' ELSE 'game' END AS domain
        ORDER BY i.id
    """,
)

ITEM_DETAILS = Template(
    name="item_details",
    description="Full item payload (both domains) for a list of node ids.",
    cypher="""
        MATCH (i:Item)
        WHERE i.id IN $ids
        OPTIONAL MATCH (i)-[:HAS_GENRE]->(g:Genre)
        OPTIONAL MATCH (i)-[ht:HAS_THEME]->(t:Theme)
        RETURN i.id AS id, i.title AS title, i.year AS year,
               CASE WHEN i:Movie THEN 'movie' ELSE 'game' END AS domain,
               coalesce(i.movie_id, i.app_id) AS item_id,
               coalesce(i.rating_avg, 0.0) AS rating_avg,
               coalesce(i.n_ratings, i.n_reviews, 0) AS n_ratings,
               collect(DISTINCT g.name) AS genres,
               collect(DISTINCT t.name) AS themes,
               coalesce(sum(DISTINCT ht.weight), 0.0) AS theme_weight
    """,
)

CATALOG_STATS = Template(
    name="catalog_stats",
    description="Node/edge counts per label and type.",
    cypher="""
        MATCH (n) WITH head(labels(n)) AS label, count(*) AS count
        RETURN collect({label: label, count: count}) AS nodes
    """,
)

# --------------------------------------------------------------------------- #
# retrieval
# --------------------------------------------------------------------------- #
ITEM_IDS_BY_THEME = Template(
    name="item_ids_by_theme",
    description="Items linked to any of the given themes, scored by theme weight.",
    cypher="""
        MATCH (i:Item)-[ht:HAS_THEME]->(t:Theme)
        WHERE t.name IN $themes
        WITH i, sum(ht.weight) AS theme_score, collect(DISTINCT t.name) AS matched
        WHERE ($domain = 'both' OR ($domain = 'movie' AND i:Movie) OR ($domain = 'game' AND i:Game))
        RETURN i.id AS id, theme_score, matched
        ORDER BY theme_score DESC, id
        LIMIT $limit
    """,
)

ITEM_IDS_BY_GENRE = Template(
    name="item_ids_by_genre",
    description="Items linked to any of the given canonical genres.",
    cypher="""
        MATCH (i:Item)-[:HAS_GENRE]->(g:Genre)
        WHERE g.name IN $genres
        WITH i, count(DISTINCT g) AS genre_hits, collect(DISTINCT g.name) AS matched
        WHERE ($domain = 'both' OR ($domain = 'movie' AND i:Movie) OR ($domain = 'game' AND i:Game))
        RETURN i.id AS id, genre_hits, matched
        ORDER BY genre_hits DESC, id
        LIMIT $limit
    """,
)

ITEM_IDS_SIMILAR_TO = Template(
    name="item_ids_similar_to",
    description="Neighbours of the reference items (embedding similarity edges).",
    cypher="""
        MATCH (src:Item)-[r:SIMILAR_TO]->(dst:Item)
        WHERE src.id IN $ids
        WITH dst, max(r.score) AS score, collect(DISTINCT src.id) AS from_items
        WHERE ($domain = 'both' OR ($domain = 'movie' AND dst:Movie) OR ($domain = 'game' AND dst:Game))
        RETURN dst.id AS id, score, from_items
        ORDER BY score DESC
        LIMIT $limit
    """,
)

VECTOR_QUERY = Template(
    name="vector_query",
    description="Vector index search over :Item.embedding (cosine).",
    cypher="""
        CALL db.index.vector.queryNodes($index, $k, $embedding)
        YIELD node, score
        WHERE ($domain = 'both' OR ($domain = 'movie' AND node:Movie)
               OR ($domain = 'game' AND node:Game))
        RETURN node.id AS id, score
    """,
)

USER_LIKED_ITEMS = Template(
    name="user_liked_items",
    description="Items a user liked: rating >= threshold, or a positive review.",
    cypher="""
        MATCH (u:User {id: $user_id})-[r:RATED]->(m:Movie)
        WHERE r.score >= $min_rating
        RETURN m.id AS id, r.score AS score, r.ts AS ts, 'movie' AS domain
        UNION
        MATCH (u:User {id: $user_id})-[r:REVIEWED]->(g:Game)
        WHERE r.recommended = true
        RETURN g.id AS id, r.hours AS score, r.ts AS ts, 'game' AS domain
        ORDER BY ts DESC
        LIMIT $limit
    """,
)

# --------------------------------------------------------------------------- #
# evidence (used by the chatbot explanations)
# --------------------------------------------------------------------------- #
SHARED_THEMES = Template(
    name="shared_themes",
    description="Themes a candidate shares with the reference items, with weights.",
    cypher="""
        MATCH (ref:Item)-[rt:HAS_THEME]->(t:Theme)<-[ct:HAS_THEME]-(cand:Item)
        WHERE ref.id IN $reference_ids AND cand.id IN $candidate_ids
        WITH cand.id AS candidate, t.name AS theme,
             min(rt.weight) AS ref_weight, max(ct.weight) AS cand_weight
        WHERE ref_weight > $min_weight AND cand_weight > $min_weight
        RETURN candidate, collect(DISTINCT theme) AS shared_themes
    """,
)

SHARED_GENRES = Template(
    name="shared_genres",
    description="Genres a candidate shares with the reference items.",
    cypher="""
        MATCH (ref:Item)-[:HAS_GENRE]->(g:Genre)<-[:HAS_GENRE]-(cand:Item)
        WHERE ref.id IN $reference_ids AND cand.id IN $candidate_ids
        RETURN cand.id AS candidate, collect(DISTINCT g.name) AS shared_genres
    """,
)

SIMILAR_TO_REFERENCE = Template(
    name="similar_to_reference",
    description="The SIMILAR_TO links between candidates and the reference items.",
    cypher="""
        MATCH (ref:Item)-[r:SIMILAR_TO]->(cand:Item)
        WHERE ref.id IN $reference_ids AND cand.id IN $candidate_ids
        RETURN ref.id AS reference, cand.id AS candidate, r.score AS score,
               ref.title AS reference_title
        ORDER BY score DESC
    """,
)

EXPLAIN_ITEM = Template(
    name="explain_item",
    description="Full context for one item: themes, genres and its closest items.",
    cypher="""
        MATCH (i:Item {id: $id})
        OPTIONAL MATCH (i)-[ht:HAS_THEME]->(t:Theme)
        OPTIONAL MATCH (i)-[:HAS_GENRE]->(g:Genre)
        OPTIONAL MATCH (i)-[r:SIMILAR_TO]->(other:Item)
        RETURN i.id AS id, i.title AS title,
               collect(DISTINCT t.name) AS themes,
               collect(DISTINCT g.name) AS genres,
               collect(DISTINCT {title: other.title, score: r.score}) AS similar
    """,
)


def templates() -> dict[str, Template]:
    """Every template, keyed by name (used by the API/docs and tests)."""
    return {
        value.name: value
        for value in globals().values()
        if isinstance(value, Template)
    }


def vector_query_params(
    embedding: list[float], k: int = 30, domain: str = "both", index: str = "item_embedding"
) -> dict[str, Any]:
    """Parameters for :data:`VECTOR_QUERY`."""
    return {"embedding": list(embedding), "k": int(k), "domain": domain, "index": index}


def theme_params(themes: list[str], domain: str = "both", limit: int = 50) -> dict[str, Any]:
    """Parameters for :data:`ITEM_IDS_BY_THEME`."""
    return {"themes": list(themes), "domain": domain, "limit": int(limit)}


def genre_params(genres: list[str], domain: str = "both", limit: int = 50) -> dict[str, Any]:
    """Parameters for :data:`ITEM_IDS_BY_GENRE`."""
    return {"genres": list(genres), "domain": domain, "limit": int(limit)}


def similar_params(ids: list[str], domain: str = "both", limit: int = 50) -> dict[str, Any]:
    """Parameters for :data:`ITEM_IDS_SIMILAR_TO`."""
    return {"ids": list(ids), "domain": domain, "limit": int(limit)}


__all__ = [
    "ALL_ITEM_TITLES",
    "CATALOG_STATS",
    "EXPLAIN_ITEM",
    "ITEM_DETAILS",
    "ITEM_IDS_BY_GENRE",
    "ITEM_IDS_BY_THEME",
    "ITEM_IDS_SIMILAR_TO",
    "SHARED_GENRES",
    "SHARED_THEMES",
    "SIMILAR_TO_REFERENCE",
    "USER_LIKED_ITEMS",
    "VECTOR_QUERY",
    "Template",
    "genre_params",
    "similar_params",
    "templates",
    "theme_params",
    "vector_query_params",
]
