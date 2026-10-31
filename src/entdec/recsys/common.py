"""Shared plumbing for the recommender models (PROJECT_SPEC.md section 7).

The evaluation rules are:

* split per user by time (already materialised in the Parquet ``split`` column
  by Phase 1): last positive = test, second-last = validation, rest = train;
* rank the **full sampled catalogue of the domain**, excluding the items the
  user already has in train.

A :class:`DomainData` bundles everything a model needs for one domain, and
:class:`ScoredCandidates` is the model output: ``TopK = model(user_key) -> ids
and scores``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

DOMAINS: dict[str, dict[str, str]] = {
    "movie": {
        "items": "movies.parquet",
        "interactions": "movie_ratings.parquet",
        "item_key": "movie_id",
        "themes": "movie_themes.parquet",
    },
    "game": {
        "items": "games.parquet",
        "interactions": "game_reviews.parquet",
        "item_key": "app_id",
        "themes": "game_themes.parquet",
    },
}


@dataclass
class DomainData:
    """Everything a recommender needs for one domain, in row-aligned arrays."""

    domain: str
    items: pd.DataFrame
    interactions: pd.DataFrame
    item_key: str  # column name of the raw item id
    catalog: list[int]  # raw item ids, aligned with all matrices below
    catalog_index: dict[int, int]  # raw item id -> matrix row
    user_ids: list[int]  # raw user ids, aligned with the held-out sets
    user_indices: pd.Series  # raw user id -> row of the (user x item) matrix
    matrix: Any  # scipy sparse matrix: users x items, binary "liked in train"
    likes: dict[int, list[int]]  # raw user id -> raw item ids liked in train
    valid_targets: dict[int, int]  # raw user id -> raw item id held out
    test_targets: dict[int, int]  # raw user id -> raw item id held out
    themes: pd.DataFrame  # item themes with weights (raw item id, theme, weight)
    genres: pd.DataFrame  # canonical genres per raw item id (movies and games)
    embeddings: Any  # np.ndarray (n_items, dim) aligned with `catalog`, or None
    meta: dict[str, pd.DataFrame] = field(default_factory=dict)  # items / interactions

    def item_pos(self, raw_item_id: int) -> int | None:
        """Matrix row of a raw item id (None when outside the catalogue)."""
        return self.catalog_index.get(int(raw_item_id))


def _load_themes(cfg: Config, domain: str) -> pd.DataFrame:
    path = cfg.paths.processed / DOMAINS[domain]["themes"]
    if not path.exists():
        return pd.DataFrame(columns=["item_id", "theme", "weight"])
    frame = pd.read_parquet(path)
    key = "item_id" if "item_id" in frame.columns else (
        "movie_id" if "movie_id" in frame.columns else "app_id"
    )
    return frame.rename(columns={key: "item_id"})[["item_id", "theme", "weight"]]


def _load_embeddings(cfg: Config) -> dict[tuple[str, int], np.ndarray]:
    path = cfg.paths.processed / "item_embeddings.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["domain", "item_id", "embedding"])
    return {
        (str(row.domain), int(row.item_id)): np.asarray(row.embedding, dtype=np.float32)
        for row in frame.itertuples(index=False)
    }


def load_domain(cfg: Config, domain: str, *, with_embeddings: bool = True) -> DomainData:
    """Build the :class:`DomainData` view for one domain (from the Phase 1 Parquet)."""
    spec = DOMAINS[domain]
    processed = cfg.paths.processed
    items = pd.read_parquet(processed / spec["items"])
    interactions = pd.read_parquet(processed / spec["interactions"])

    catalog = items[spec["item_key"]].astype(int).tolist()
    catalog_index = {raw: pos for pos, raw in enumerate(catalog)}

    train = interactions[interactions["split"] == "train"]
    positives = train[train["is_positive"]]
    valid = interactions[interactions["split"] == "valid"]
    test = interactions[interactions["split"] == "test"]

    # user x item binary matrix over "liked in train"
    users_in_train = positives["uid"].astype(int)
    user_ids = sorted(users_in_train.unique().tolist())
    user_pos = {raw: pos for pos, raw in enumerate(user_ids)}

    import scipy.sparse as sp

    rows = users_in_train.map(user_pos).to_numpy()
    cols = positives[spec["item_key"]].map(catalog_index)
    keep = cols.notna().to_numpy()
    cols = cols[keep].astype(int).to_numpy()
    rows = rows[keep]
    matrix = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(user_ids), len(catalog)),
        dtype=np.float32,
    )

    likes: dict[int, list[int]] = {}
    for user, frame in positives.groupby("uid"):
        raw_items = frame[spec["item_key"]].astype(int)
        likes[int(user)] = [int(r) for r in raw_items if catalog_index.get(int(r)) is not None]

    def heldout(frame: pd.DataFrame) -> dict[int, int]:
        # one row per user by construction (last/second-last positive)
        out: dict[int, int] = {}
        for row in frame.itertuples(index=False):
            user = int(row.uid)
            raw = getattr(row, spec["item_key"])
            if user_pos.get(user) is not None and catalog_index.get(int(raw)) is not None:
                out[user] = int(raw)
        return out

    embedding_map = _load_embeddings(cfg) if with_embeddings else {}
    dim = len(next(iter(embedding_map.values()))) if embedding_map else 0
    embeddings = np.zeros((len(catalog), dim), dtype=np.float32)
    if dim:
        found = 0
        for pos, raw in enumerate(catalog):
            vec = embedding_map.get((domain, int(raw)))
            if vec is not None:
                embeddings[pos] = vec
                found += 1
        if found < len(catalog):
            logger.warning("%s: %d/%d items lack embeddings", domain, found, len(catalog))

    genres_domain = (
        pd.read_parquet(processed / "item_genres.parquet")
        if (processed / "item_genres.parquet").exists()
        else pd.DataFrame(columns=["domain", "item_id", "genre"])
    )
    genres = genres_domain[genres_domain["domain"] == domain][["item_id", "genre"]].copy()
    genres["item_id"] = genres["item_id"].astype(int)

    return DomainData(
        domain=domain,
        items=items,
        interactions=interactions,
        item_key=spec["item_key"],
        catalog=catalog,
        catalog_index=catalog_index,
        user_ids=user_ids,
        user_indices=pd.Series(user_pos),
        matrix=matrix,
        likes=likes,
        valid_targets=heldout(valid),
        test_targets=heldout(test),
        themes=_load_themes(cfg, domain),
        genres=genres,
        embeddings=embeddings if dim else None,
        meta={"items": items, "interactions": interactions},
    )


@dataclass
class ModelOutput:
    """Scores for one user over the full catalogue."""

    scores: np.ndarray  # aligned with DomainData.catalog
    name: str


ModelFn = Callable[[DomainData, int], ModelOutput]


def exclude_seen(scores: np.ndarray, data: DomainData, user: int) -> np.ndarray:
    """Zero out (-> -inf) catalogue positions the user already liked in train."""
    scores = scores.astype(np.float64, copy=True)
    liked = data.likes.get(int(user), ())
    for raw in liked:
        pos = data.item_pos(raw)
        if pos is not None:
            scores[pos] = -np.inf
    return scores


def top_k_positions(scores: np.ndarray, k: int) -> list[int]:
    """Indices of the k largest finite scores, best first."""
    finite = np.isfinite(scores)
    if not finite.any():
        return []
    masked = np.where(finite, scores, -np.inf)
    k = min(k, int(finite.sum()))
    top = np.argpartition(-masked, k - 1)[:k]
    return top[np.argsort(-masked[top], kind="stable")].tolist()


def catalogue_titles(data: DomainData) -> pd.DataFrame:
    """Raw item id -> title, for reports."""
    key = data.item_key
    return data.items.set_index(key)["title"]
