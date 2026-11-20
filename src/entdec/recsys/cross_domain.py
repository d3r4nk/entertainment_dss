"""Cross-domain recommendations (section 7).

``recommend_cross_domain(source_domain, liked_items, target_domain)`` builds a
Theme/Genre profile from the liked items of one domain and recommends items of
the other via **shared themes + embedding similarity**. Every result carries the
shared themes as evidence so the chatbot/LLM can quote them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from entdec.config import Config
from entdec.logging_setup import get_logger
from entdec.recsys.common import DOMAINS, DomainData, load_domain
from entdec.recsys.kg_scoring import item_theme_vectors

logger = get_logger(__name__)


@dataclass
class CrossDomainResult:
    """One recommendation with its evidence."""

    item_id: int
    title: str
    score: float
    theme_score: float
    embedding_score: float
    shared_themes: list[str]


@dataclass
class CrossDomainScores:
    """Raw theme, embedding, and combined cross-domain scores."""

    theme_profiles: np.ndarray
    theme_scores: np.ndarray
    embedding_scores: np.ndarray
    combined_scores: np.ndarray


class CrossDomainScorer:
    """Vectorized implementation of the scoring used by the ``cross`` command."""

    def __init__(self, source: DomainData, target: DomainData) -> None:
        self.source = source
        self.target = target
        self.source_vectors = item_theme_vectors(source)
        self.target_vectors = item_theme_vectors(target)
        self.theme_names = sorted(
            {theme for vector in self.source_vectors.values() for theme in vector}
            | {theme for vector in self.target_vectors.values() for theme in vector}
        )
        self.theme_positions = {theme: pos for pos, theme in enumerate(self.theme_names)}
        self.target_theme_matrix = self._theme_matrix(
            target.catalog, self.target_vectors, self.theme_positions
        )
        n_themed = max(1, sum(bool(vector) for vector in self.target_vectors.values()))
        document_frequency = np.zeros(len(self.theme_names), dtype=np.float64)
        for vector in self.target_vectors.values():
            for theme in vector:
                document_frequency[self.theme_positions[theme]] += 1
        self.theme_idf = np.zeros(len(self.theme_names), dtype=np.float64)
        present = document_frequency > 0
        self.theme_idf[present] = np.log(n_themed / document_frequency[present])
        self.weighted_target_themes = self.target_theme_matrix * self.theme_idf

    @staticmethod
    def _theme_matrix(
        catalog: list[int],
        vectors: dict[int, dict[str, float]],
        positions: dict[str, int],
    ) -> np.ndarray:
        matrix = np.zeros((len(catalog), len(positions)), dtype=np.float64)
        for row, item_id in enumerate(catalog):
            for theme, weight in vectors.get(int(item_id), {}).items():
                matrix[row, positions[theme]] = float(weight)
        return matrix

    def score_sources(
        self,
        source_ids: list[int],
        *,
        source_embedding_overrides: dict[int, np.ndarray] | None = None,
        target_embedding_overrides: dict[int, np.ndarray] | None = None,
    ) -> CrossDomainScores:
        """Score source items against the entire target catalogue.

        Overrides support the title-masked gold-pair diagnostic without mutating
        the cached catalogue embeddings.
        """
        embedding_dim = (
            self.target.embeddings.shape[1] if self.target.embeddings is not None else 0
        )
        profiles = np.zeros((len(source_ids), len(self.theme_names)), dtype=np.float64)
        embedding_profiles = np.zeros((len(source_ids), embedding_dim), dtype=np.float32)
        for row, item_id in enumerate(source_ids):
            vector = self.source_vectors.get(int(item_id), {})
            for theme, weight in vector.items():
                profiles[row, self.theme_positions[theme]] += float(weight)
            total = profiles[row].sum()
            if total > 0:
                profiles[row] /= total

            position = self.source.item_pos(int(item_id))
            if position is not None and self.source.embeddings is not None:
                override = (source_embedding_overrides or {}).get(int(item_id))
                embedding_profiles[row] = (
                    override if override is not None else self.source.embeddings[position]
                )
        norms = np.linalg.norm(embedding_profiles, axis=1, keepdims=True)
        embedding_profiles /= np.where(norms > 0, norms, 1.0)

        theme_scores = profiles @ self.weighted_target_themes.T
        target_embeddings = self.target.embeddings
        embedding_scores = np.zeros(
            (len(source_ids), len(self.target.catalog)), dtype=np.float64
        )
        if target_embeddings is not None and embedding_profiles.shape[1]:
            if target_embedding_overrides:
                target_embeddings = target_embeddings.copy()
                for item_id, vector in target_embedding_overrides.items():
                    position = self.target.item_pos(item_id)
                    if position is not None:
                        target_embeddings[position] = vector
            embedding_scores = embedding_profiles @ target_embeddings.T

        theme_max = theme_scores.max(axis=1, keepdims=True)
        embedding_max = embedding_scores.max(axis=1, keepdims=True)
        normalised_themes = np.divide(
            theme_scores,
            theme_max,
            out=theme_scores.copy(),
            where=theme_max > 0,
        )
        normalised_embeddings = np.divide(
            embedding_scores,
            embedding_max,
            out=embedding_scores.copy(),
            where=embedding_max > 0,
        )
        combined = 0.6 * normalised_themes + 0.4 * normalised_embeddings
        combined = np.where(np.isfinite(combined), combined, -np.inf)
        return CrossDomainScores(
            theme_profiles=profiles,
            theme_scores=theme_scores,
            embedding_scores=embedding_scores,
            combined_scores=combined,
        )

    def shared_themes(self, source_ids: int | list[int], target_id: int) -> list[str]:
        """Return the highest-contributing shared themes, matching CLI evidence."""
        ids = [source_ids] if isinstance(source_ids, int) else source_ids
        source: dict[str, float] = {}
        for source_id in ids:
            for theme, weight in self.source_vectors.get(int(source_id), {}).items():
                source[theme] = source.get(theme, 0.0) + weight
        target = self.target_vectors.get(int(target_id), {})
        total = sum(source.values())
        profile = {theme: weight / total for theme, weight in source.items()} if total else {}
        shared = [theme for theme in target if theme in profile]
        shared.sort(
            key=lambda theme: -(
                profile[theme]
                * target[theme]
                * self.theme_idf[self.theme_positions[theme]]
            )
        )
        return shared[:5]


def _titles(data: pd.DataFrame, key: str) -> dict[int, str]:
    return {
        int(item_id): str(title)
        for item_id, title in zip(data[key], data["title"], strict=False)
    }


def resolve_items(data: pd.DataFrame, liked_items: list[str], key: str) -> dict[int, str]:
    """Match free-text titles to catalogue ids (contains, then fuzzy)."""
    titles = _titles(data, key)
    resolved: dict[int, str] = {}
    for wanted in liked_items:
        wanted_lower = wanted.strip().lower()
        direct = [
            (raw, title)
            for raw, title in titles.items()
            if wanted_lower in title.lower() or title.lower() in wanted_lower
        ]
        if direct:
            raw, title = direct[0]
            resolved[raw] = title
            continue
        # graceful degradation: closest fuzzy title (seed item may be a base
        # game missing from the snapshot - try its best match anyway)
        from rapidfuzz import process

        matches = process.extract(
            wanted,
            titles,
            limit=1,
            score_cutoff=60.0,
        )
        if matches:
            raw = int(matches[0][2])
            resolved[raw] = titles[raw]
            logger.info("fuzzy-resolved catalogue title")
    return resolved


def recommend_cross_domain(
    cfg: Config,
    source_domain: str,
    liked_items: list[str],
    target_domain: str,
    *,
    top_k: int | None = None,
) -> list[CrossDomainResult]:
    """Recommend items in ``target_domain`` from liked items of ``source_domain``."""
    top_k = int(top_k or cfg.get("cross_domain.top_k", 8))
    source = load_domain(cfg, source_domain, with_embeddings=True)
    target = load_domain(cfg, target_domain, with_embeddings=True)

    key = DOMAINS[source_domain]["item_key"]
    resolved = resolve_items(source.items, liked_items, key)
    if not resolved:
        logger.warning("none of %s resolved in the %s catalogue", liked_items, source_domain)
        return []
    logger.info("resolved: %s", resolved)

    scorer = CrossDomainScorer(source, target)
    scores = scorer.score_sources(list(resolved))
    target_titles = _titles(target.items, DOMAINS[target_domain]["item_key"])
    theme_scores = scores.theme_scores[0]
    embed_scores = scores.embedding_scores[0]
    combined = scores.combined_scores[0]

    results: list[CrossDomainResult] = []
    k = min(top_k, int((combined > 0).sum()))
    if k == 0:
        return []
    top_pos = np.argpartition(-combined, k - 1)[:k]
    for pos in top_pos[np.argsort(-combined[top_pos])]:
        raw = target.catalog[int(pos)]
        results.append(
            CrossDomainResult(
                item_id=int(raw),
                title=target_titles.get(int(raw), str(raw)),
                score=float(combined[pos]),
                theme_score=float(theme_scores[pos]),
                embedding_score=float(embed_scores[pos]),
                shared_themes=scorer.shared_themes(list(resolved), int(raw)),
            )
        )
    return results


# --------------------------------------------------------------------------- #
# persona report
# --------------------------------------------------------------------------- #
def write_persona_report(cfg: Config, *, top_k: int | None = None) -> Path:
    """Run the configured personas and write the profile-specific report."""
    personas = cfg.get("cross_domain.personas", []) or []
    top_k = int(top_k or cfg.get("cross_domain.top_k", 8))
    lines = [
        "# Cross-domain persona examples",
        "",
        "Theme/genre profile + embedding similarity from one domain into the other.",
        "Every row shows the shared themes that produced the match (the evidence).",
        "",
    ]
    for persona in personas:
        name = persona["name"]
        src, tgt = persona["source_domain"], persona["target_domain"]
        liked = list(persona["liked_items"])
        lines.append(f"## {name}")
        lines.append("")
        results = recommend_cross_domain(cfg, src, liked, tgt, top_k=top_k)
        src_label = "movie" if src == "movie" else "game"
        tgt_label = "movie" if tgt == "movie" else "game"
        lines.append(f"Liked {src_label}: {', '.join(liked)} -> recommended {tgt_label}s:")
        if not results:
            lines.append("(no result - the seed item was not in the sampled catalogue)")
        for rank, r in enumerate(results, start=1):
            evidence = ", ".join(r.shared_themes) if r.shared_themes else "embedding only"
            lines.append(
                f"{rank}. **{r.title}** (theme {r.theme_score:.3f}, embed {r.embedding_score:.3f})"
                f" - shared themes: {evidence}"
            )
        lines.append("")
    path = cfg.report_path("cross_domain_examples.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s (%d personas)", path.name, len(personas))
    return path


def run(cfg: Config, args: argparse.Namespace | None = None) -> int:
    """CLI entry point: ``python -m entdec --profile tiny cross``."""
    top_k = getattr(args, "top_k", None) if args else None
    path = write_persona_report(cfg, top_k=top_k)
    logger.info("cross-domain report at reports/%s", path.name)
    return 0


__all__ = [
    "CrossDomainResult",
    "CrossDomainScores",
    "CrossDomainScorer",
    "recommend_cross_domain",
    "resolve_items",
    "run",
    "write_persona_report",
]
