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
from entdec.recsys.common import DOMAINS, load_domain
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
            raw, _score, _title = int(matches[0][2]), float(matches[0][1]), matches[0][0]
            resolved[raw] = titles[raw]
            logger.info("fuzzy-resolved %r -> %r", wanted, titles[raw])
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

    # --- theme profile -------------------------------------------------------
    item_vectors = item_theme_vectors(source)
    theme_profile: dict[str, float] = {}
    for raw in resolved:
        for theme, weight in item_vectors.get(int(raw), {}).items():
            theme_profile[theme] = theme_profile.get(theme, 0.0) + float(weight)
    if theme_profile:
        total = sum(theme_profile.values())
        theme_profile = {t: w / total for t, w in theme_profile.items()}

    # --- target-side theme vectors + embedding means --------------------------
    target_vectors = item_theme_vectors(target)
    target_titles = _titles(target.items, DOMAINS[target_domain]["item_key"])

    # TF-IDF style down-weighting: themes that appear in a large share of the
    # target catalogue ("classic", "strange", ...) carry little information,
    # otherwise every profile degenerates into them.
    n_themed = max(1, sum(1 for v in target_vectors.values() if v))
    df: dict[str, int] = {}
    for vector in target_vectors.values():
        for theme in vector:
            df[theme] = df.get(theme, 0) + 1
    idf = {t: float(np.log(n_themed / d)) for t, d in df.items()}

    theme_scores = np.zeros(len(target.catalog), dtype=np.float64)
    shared: dict[int, list[str]] = {}
    for pos, raw in enumerate(target.catalog):
        vector = target_vectors.get(int(raw))
        if not vector:
            continue
        overlap = [t for t in vector if t in theme_profile]
        if not overlap:
            continue
        contrib = {
            t: theme_profile[t] * vector[t] * idf.get(t, 0.0) for t in overlap
        }
        theme_scores[pos] = sum(contrib.values())
        shared[pos] = sorted(overlap, key=lambda t: -contrib[t])[:5]

    embed_scores = np.zeros(len(target.catalog), dtype=np.float64)
    if source.embeddings is not None and target.embeddings is not None:
        source_positions = [
            source.item_pos(raw) for raw in resolved if source.item_pos(raw) is not None
        ]
        if source_positions:
            profile = source.embeddings[source_positions].mean(axis=0)
            norm = np.linalg.norm(profile)
            if norm > 0:
                emb = target.embeddings
                embed_scores = emb @ (profile / norm)

    def _norm(v: np.ndarray) -> np.ndarray:
        hi = v.max()
        return v / hi if hi > 0 else v

    combined = 0.6 * _norm(theme_scores) + 0.4 * _norm(embed_scores)
    combined = np.where(np.isfinite(combined), combined, -np.inf)

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
                shared_themes=shared.get(int(pos), []),
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


__all__ = ["CrossDomainResult", "recommend_cross_domain", "resolve_items", "run", "write_persona_report"]
