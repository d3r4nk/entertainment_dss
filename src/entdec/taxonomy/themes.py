"""Theme taxonomy: one shared theme space for movies and games.

Pipeline (PROJECT_SPEC.md section 6):

1. collect every raw tag from both domains (MovieLens genome tags + Steam tags,
   including the propagated ones) with its frequency;
2. normalise the strings and merge exact/alias duplicates;
3. embed the unique tag names with ``qwen3-embedding:4b`` (documents get no
   instruction prefix);
4. agglomerative clustering (cosine, average linkage) into ~80 themes;
5. name each theme after its most frequent / most central member;
6. write the shared ``data/processed/theme_map.csv`` with
   ``source, raw_tag, theme`` so the mapping can be hand-edited - an existing
   CSV is always reused, never overwritten unless ``--force`` is given.

Tags that are dropped before clustering (too rare, or beyond
``taxonomy.max_tags_for_clustering``) are assigned to the nearest clustered tag
by embedding cosine, so every raw tag ends up with a theme.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

THEME_MAP_CSV = "theme_map.csv"
MOVIE_THEMES_PARQUET = "movie_themes.parquet"
GAME_THEMES_PARQUET = "game_themes.parquet"
ITEM_GENRES_PARQUET = "item_genres.parquet"
_TAG_CACHE_DIR = "tag_embeddings"

# Explicit alias merges applied *before* embedding: keys are normalised strings.
TAG_ALIASES: dict[str, str] = {
    "coop": "co-op",
    "co op": "co-op",
    "online co op": "online co-op",
    "first person": "first-person",
    "third person": "third-person",
    "point and click": "point & click",
    "hack & slash": "hack and slash",
    "beat em up": "beat 'em up",
    "f2p": "free to play",
    "mmo": "massively multiplayer",
    "mmorpg": "massively multiplayer",
    "role playing game": "rpg",
    "action rpg": "action rpg",
    "sci fi": "sci-fi",
    "open-world": "open world",
    "role-playing": "rpg",
    "soundtrack": "great soundtrack",
    "ost": "great soundtrack",
    "single player": "singleplayer",
    "multi player": "multiplayer",
    "turn based": "turn-based",
    "story-rich": "story rich",
    "visual-novel": "visual novel",
    "tower-defense": "tower defense",
    "soulslike": "souls-like",
    "souls like": "souls-like",
}

_PUNCT = re.compile(r"[^a-z0-9\-'&+ ]+")
_SPACES = re.compile(r"\s+")


def normalise_tag(raw: str) -> str:
    """Lower-case, strip punctuation and collapse whitespace for tag merging."""
    if raw is None:
        return ""
    text = str(raw).strip().lower().replace("’", "'").replace("–", "-")
    text = _PUNCT.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return TAG_ALIASES.get(text, text)


@dataclass
class ThemeBuildResult:
    """Artifacts of one theme-map build."""

    theme_map: pd.DataFrame
    themes: list[str] = field(default_factory=list)
    n_raw_tags: int = 0
    n_clustered: int = 0
    n_assigned_by_similarity: int = 0
    cache_hit: bool = False
    source: str = "clustered"

    @property
    def n_themes(self) -> int:
        return len(self.themes)


# --------------------------------------------------------------------------- #
# source tag collection
# --------------------------------------------------------------------------- #
def collect_source_tags(cfg: Config) -> pd.DataFrame:
    """Collect every raw tag per domain with its frequency.

    Returns:
        Frame with ``source`` (``movie`` / ``game``), ``raw_tag``,
        ``normalised`` and ``frequency`` (number of items using the tag).
    """
    processed = cfg.paths.processed
    rows: list[pd.DataFrame] = []

    genome_path = processed / "movie_genome.parquet"
    if genome_path.exists():
        genome = pd.read_parquet(genome_path, columns=["movie_id", "tag"])
        counts = genome.drop_duplicates(["movie_id", "tag"])["tag"].value_counts()
        rows.append(
            pd.DataFrame(
                {
                    "source": "movie",
                    "raw_tag": counts.index.astype(str),
                    "frequency": counts.to_numpy(dtype="int64"),
                }
            )
        )
    else:
        logger.warning("movie_genome.parquet missing - no movie tags collected")

    tags_path = processed / "game_tags.parquet"
    if tags_path.exists():
        game_tags = pd.read_parquet(tags_path, columns=["app_id", "tag"])
        counts = game_tags.drop_duplicates(["app_id", "tag"])["tag"].value_counts()
        rows.append(
            pd.DataFrame(
                {
                    "source": "game",
                    "raw_tag": counts.index.astype(str),
                    "frequency": counts.to_numpy(dtype="int64"),
                }
            )
        )
    else:
        logger.warning("game_tags.parquet missing - no game tags collected")

    if not rows:
        return pd.DataFrame(columns=["source", "raw_tag", "normalised", "frequency"])

    frame = pd.concat(rows, ignore_index=True)
    frame["raw_tag"] = frame["raw_tag"].astype(str)
    frame["normalised"] = frame["raw_tag"].map(normalise_tag)
    frame = frame[frame["normalised"] != ""].reset_index(drop=True)
    logger.info(
        "collected %d (source, raw_tag) pairs -> %d unique normalised tags",
        len(frame),
        frame["normalised"].nunique(),
    )
    return frame


# --------------------------------------------------------------------------- #
# embeddings (cached)
# --------------------------------------------------------------------------- #
def _cache_path(cfg: Config, key: str) -> Path:
    directory = cfg.paths.cache / _TAG_CACHE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.parquet"


def _tags_key(cfg: Config, tags: list[str], model: str, dim: int) -> str:
    payload = "\n".join([model, str(dim), *tags])
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]  # noqa: S324 - cache key only
    return f"tags_{digest}"


def embed_tag_names(
    cfg: Config,
    tags: list[str],
    *,
    force: bool = False,
    client: object | None = None,
) -> tuple[np.ndarray, bool]:
    """Embed unique tag names, caching the result in ``.cache/tag_embeddings``.

    Returns:
        ``(vectors, cache_hit)``.
    """
    from entdec.nlp.ollama_client import OllamaClient

    model = cfg.env.ollama_embed_model
    dim = int(cfg.get("nlp.embed_dim", 1024))
    key = _tags_key(cfg, tags, model, dim)
    path = _cache_path(cfg, key)

    if path.exists() and not force:
        cached = pd.read_parquet(path)
        vectors = np.vstack(cached["embedding"].to_numpy()).astype(np.float32)
        if vectors.shape[0] == len(tags):
            logger.info("tag embedding cache hit (%s)", path.name)
            return vectors, True
        logger.warning(
            "tag embedding cache is stale (%d vs %d) - recomputing", vectors.shape[0], len(tags)
        )

    ollama = client if client is not None else OllamaClient.from_config(cfg)
    batch_size = int(cfg.get("nlp.tag_embed_batch_size", 128))
    vectors = ollama.embed_batch(tags, batch_size=batch_size, dimensions=dim)
    pd.DataFrame({"tag": tags, "embedding": list(vectors)}).to_parquet(path, index=False)
    logger.info("embedded %d tag names (%d dims) -> %s", len(tags), vectors.shape[1], path.name)
    return vectors, False


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def cluster_tags(
    vectors: np.ndarray,
    tags: list[str],
    frequencies: np.ndarray,
    n_themes: int,
) -> np.ndarray:
    """Agglomerative clustering of the tag embeddings.

    Ward linkage on L2-normalised vectors is used. Measured on the real tag set
    (1,301 tags, 80 clusters) it is far more balanced than average linkage with
    the cosine metric, which collapses everything into one ~240 tag cluster plus
    singletons: ward gives min=3, max=58, median=14, no singletons. Euclidean
    distance between normalised vectors is monotone in cosine distance, so the
    geometry is unchanged.

    Returns:
        Integer cluster label per tag.
    """
    n_items = len(tags)
    n_clusters = max(1, min(int(n_themes), n_items))
    if n_clusters == 1:
        return np.zeros(n_items, dtype=np.int32)

    from sklearn.cluster import AgglomerativeClustering

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalised = vectors / np.where(norms == 0, 1.0, norms)
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = model.fit_predict(normalised.astype(np.float64))
    sizes = np.bincount(labels)
    logger.info(
        "clustered %d tags into %d clusters (sizes: min=%d, median=%d, max=%d)",
        n_items,
        n_clusters,
        int(sizes.min()),
        int(np.median(sizes)),
        int(sizes.max()),
    )
    return labels.astype(np.int32)


def name_clusters(
    labels: np.ndarray,
    tags: list[str],
    frequencies: np.ndarray,
    vectors: np.ndarray,
) -> dict[int, str]:
    """Name each cluster after its most frequent, most central member."""
    names: dict[int, str] = {}
    used: set[str] = set()
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        member_vectors = vectors[members]
        centroid = member_vectors.mean(axis=0)
        centroid /= np.linalg.norm(centroid) or 1.0
        centrality = member_vectors @ centroid
        order = np.lexsort(
            (-centrality, -frequencies[members])
        )  # primary: frequency desc, secondary: centrality desc
        name = tags[int(members[order[0]])]
        if name in used:
            for candidate in members[order]:
                if tags[int(candidate)] not in used:
                    name = tags[int(candidate)]
                    break
        used.add(name)
        names[int(label)] = name
    return names


def assign_by_similarity(
    vectors: np.ndarray,
    labels: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    """Assign vectors to the cluster of their nearest embedded tag."""
    normalised = vectors / np.where(
        np.linalg.norm(vectors, axis=1, keepdims=True) == 0,
        1.0,
        np.linalg.norm(vectors, axis=1, keepdims=True),
    )
    queries = query / np.where(
        np.linalg.norm(query, axis=1, keepdims=True) == 0,
        1.0,
        np.linalg.norm(query, axis=1, keepdims=True),
    )
    best = np.argmax(queries @ normalised.T, axis=1)
    return labels[best]


# --------------------------------------------------------------------------- #
# theme map
# --------------------------------------------------------------------------- #
def load_theme_map(cfg: Config) -> pd.DataFrame | None:
    """Load the shared ``data/processed/theme_map.csv`` if it exists."""
    path = cfg.theme_map_path
    if not path.exists():
        return None
    frame = pd.read_csv(path, dtype=str).fillna("")
    missing = {"source", "raw_tag", "theme"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    logger.info("loaded %d theme_map rows from %s", len(frame), path.name)
    return frame


def build_theme_map(
    cfg: Config,
    *,
    force: bool = False,
    n_themes: int | None = None,
    limit_tags: int | None = None,
    client: object | None = None,
) -> ThemeBuildResult:
    """Build (or load) the theme map and write ``theme_map.csv``.

    Args:
        cfg: Application config.
        force: Recompute even when ``theme_map.csv`` already exists.
        n_themes: Override ``taxonomy.themes``.
        limit_tags: Keep only the N most frequent tags (development aid).
        client: Inject an Ollama client (tests).

    Returns:
        A :class:`ThemeBuildResult`.
    """
    existing = load_theme_map(cfg)
    if existing is not None and not force:
        themes = sorted(existing["theme"].unique().tolist())
        logger.info("reusing existing theme map (%d themes)", len(themes))
        return ThemeBuildResult(
            theme_map=existing,
            themes=themes,
            n_raw_tags=len(existing),
            source="theme_map.csv",
        )

    source_tags = collect_source_tags(cfg)
    if source_tags.empty:
        raise RuntimeError("no source tags found - run `entdec data` first")

    grouped = (
        source_tags.groupby("normalised", sort=False)
        .agg(
            frequency=("frequency", "sum"),
            variants=("raw_tag", lambda values: list(dict.fromkeys(values))),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["frequency", "normalised"], ascending=[False, True]).reset_index(
        drop=True
    )
    if limit_tags:
        grouped = grouped.head(int(limit_tags)).reset_index(drop=True)

    tags = grouped["normalised"].tolist()
    frequencies = grouped["frequency"].to_numpy()

    target_themes = int(n_themes or cfg.get("taxonomy.themes", 80))
    min_frequency = int(cfg.get("taxonomy.min_tag_frequency", 3))
    max_cluster = int(cfg.get("taxonomy.max_tags_for_clustering", 6000))

    clusterable = frequencies >= min_frequency
    if int(clusterable.sum()) > max_cluster:
        # Keep the `max_cluster` most frequent tags; the rest are assigned later.
        cutoff = int(np.sort(frequencies)[::-1][max_cluster - 1])
        clusterable = frequencies >= max(cutoff, min_frequency)
    cluster_idx = np.flatnonzero(clusterable)
    if cluster_idx.size == 0:
        cluster_idx = np.arange(len(tags))
    logger.info(
        "%d of %d tags enter clustering (min_frequency=%d)",
        cluster_idx.size,
        len(tags),
        min_frequency,
    )

    vectors, cache_hit = embed_tag_names(cfg, tags, force=force, client=client)

    cluster_vectors = vectors[cluster_idx]
    cluster_tags_list = [tags[int(i)] for i in cluster_idx]
    labels = cluster_tags(
        cluster_vectors, cluster_tags_list, frequencies[cluster_idx], target_themes
    )
    cluster_names = name_clusters(
        labels, cluster_tags_list, frequencies[cluster_idx], cluster_vectors
    )
    theme_per_cluster = {label: cluster_names[label] for label in np.unique(labels)}

    theme_of_tag: dict[str, str] = {
        tags[int(idx)]: theme_per_cluster[int(labels[position])]
        for position, idx in enumerate(cluster_idx)
    }

    rest_idx = np.setdiff1d(np.arange(len(tags)), cluster_idx)
    if rest_idx.size:
        assigned = assign_by_similarity(cluster_vectors, labels, vectors[rest_idx])
        for position, idx in enumerate(rest_idx):
            theme_of_tag[tags[int(idx)]] = theme_per_cluster[int(assigned[position])]
    logger.info(
        "themes assigned: %d by clustering, %d by nearest clustered tag",
        cluster_idx.size,
        int(rest_idx.size),
    )

    rows: list[dict[str, str]] = []
    for _, record in grouped.iterrows():
        theme = theme_of_tag.get(str(record["normalised"]))
        if not theme:
            continue
        for variant in record["variants"]:
            rows.append({"source": "", "raw_tag": str(variant), "theme": theme})

    # Attach the true source per raw tag (a variant can appear in both domains).
    raw_to_source = (
        source_tags.groupby("raw_tag")["source"]
        .agg(lambda values: "/".join(sorted(set(values))))
        .to_dict()
    )
    for row in rows:
        row["source"] = raw_to_source.get(row["raw_tag"], "")

    theme_map = pd.DataFrame(rows, columns=["source", "raw_tag", "theme"])
    theme_map = theme_map.drop_duplicates(["raw_tag"]).sort_values(
        ["theme", "raw_tag"], ignore_index=True
    )
    path = cfg.theme_map_path
    theme_map.to_csv(path, index=False)
    logger.info(
        "wrote %s (%d rows, %d themes)", path.name, len(theme_map), theme_map["theme"].nunique()
    )

    return ThemeBuildResult(
        theme_map=theme_map,
        themes=sorted(theme_map["theme"].unique().tolist()),
        n_raw_tags=len(theme_map),
        n_clustered=int(cluster_idx.size),
        n_assigned_by_similarity=int(rest_idx.size),
        cache_hit=cache_hit,
        source="clustered",
    )


def theme_lookup(theme_map: pd.DataFrame) -> dict[str, str]:
    """Map a normalised raw tag to its theme."""
    return {
        normalise_tag(raw): str(theme)
        for raw, theme in zip(theme_map["raw_tag"], theme_map["theme"], strict=False)
    }


__all__ = [
    "GAME_THEMES_PARQUET",
    "ITEM_GENRES_PARQUET",
    "MOVIE_THEMES_PARQUET",
    "TAG_ALIASES",
    "THEME_MAP_CSV",
    "ThemeBuildResult",
    "assign_by_similarity",
    "build_theme_map",
    "cluster_tags",
    "collect_source_tags",
    "embed_tag_names",
    "load_theme_map",
    "name_clusters",
    "normalise_tag",
    "theme_lookup",
]
