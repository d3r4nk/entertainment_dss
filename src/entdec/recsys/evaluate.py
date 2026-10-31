"""Evaluation pipeline (section 7): train every model, tune hybrid and anonymous-chat
weights on validation, score the test split and write ``reports/<profile>/metrics.md`` +
``data/processed/<profile>/metrics.csv``.

The hybrid weight tuner never re-scores users per candidate: the normalised
per-model score vectors of a fixed, seeded sample of validation users are
computed once (:meth:`Evaluator.prepare_tuning`) and each grid candidate is
only a weighted sum plus NDCG@10.

CLI: ``python -m entdec --profile tiny recsys [train|evaluate|tune|report|all]``
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from entdec.config import Config
from entdec.logging_setup import get_logger
from entdec.recsys import content, hybrid, itemknn, kg_scoring, mf, popularity
from entdec.recsys.common import (
    DOMAINS,
    DomainData,
    exclude_seen,
    load_domain,
    top_k_positions,
)
from entdec.recsys.metrics import (
    hit_rate_at_k,
    intra_list_diversity,
    item_coverage,
    ndcg_at_k,
    recall_at_k,
    theme_dissimilarity_fn,
)

logger = get_logger(__name__)

ABLATION_MODELS: dict[str, str | None] = {
    "pop": "popularity",
    "knn": "itemknn",
    "mf": "mf",
    "content": "content",
    "kg": "kg",
    "content+kg": "content_kg",
    "hybrid-all": "hybrid",
}
CHAT_MODEL_NAMES = ("popularity", "content", "kg")


@dataclass
class PerModelResult:
    """Aggregated metrics for one model on one domain."""

    domain: str
    model: str
    recall_at_10: float
    recall_at_20: float
    ndcg_at_10: float
    ndcg_at_20: float
    hit_rate_at_10: float
    coverage: float
    diversity: float
    n_users: int
    duration_s: float


# --------------------------------------------------------------------------- #
# moving parts
# --------------------------------------------------------------------------- #
def theme_sets(data: DomainData) -> dict[int, frozenset[str]]:
    out: dict[int, frozenset[str]] = {}
    for row in data.themes.itertuples(index=False):
        out.setdefault(int(row.item_id), set()).add(str(row.theme))
    return {k: frozenset(v) for k, v in out.items()}


def rank_user(
    scores: np.ndarray, data: DomainData, user: int, k: int
) -> list[int]:
    """Scores -> top-k raw item ids, excluding train items."""
    scores = exclude_seen(scores, data, user)
    positions = top_k_positions(scores, k)
    return [data.catalog[pos] for pos in positions]


HELDOUT: dict[str, dict[int, int]] = {}


def heldout(data: DomainData, split: str) -> dict[int, int]:
    return data.test_targets if split == "test" else data.valid_targets


class Evaluator:
    """Fits the models once per domain, then scores users on a split."""

    def __init__(self, cfg: Config, data: DomainData) -> None:
        self.cfg = cfg
        self.data = data
        self.theme_lookup = theme_sets(data)
        self.distance_fn = theme_dissimilarity_fn(self.theme_lookup)
        self.k_values = [int(k) for k in cfg.get("evaluation.k_values", [10, 20])]
        self.fitted: dict[str, object] = {}
        self.catalog_positions = {raw: pos for pos, raw in enumerate(data.catalog)}

    def fit(self, *, with_mf: bool = True) -> None:
        data = self.data
        self.fitted["itemknn"] = itemknn.ItemKNNModel(data)
        if with_mf:
            self.fitted["mf"] = mf.fit(data)
        logger.info("%s: itemknn/mf fitted", data.domain)

    def score_user(self, user: int, blend_weights: Mapping[str, float] | None = None):
        """Normalised base-parts + every ablation score for one user."""
        parts = hybrid.normalised_scores(self.data, user, self.fitted)
        out: dict[str, np.ndarray] = {
            "pop": parts["popularity"],
            "knn": parts["itemknn"],
            "mf": parts["mf"],
            "content": parts["content"],
            "kg": parts["kg"],
        }
        out["content+kg"] = 0.5 * parts["content"] + 0.5 * parts["kg"]
        out["hybrid-all"] = hybrid.blend(parts, blend_weights or {})
        return out

    def evaluate(
        self,
        split: str = "test",
        *,
        weights: Mapping[str, float] | None = None,
        limit_users: int | None = None,
    ) -> list[PerModelResult]:
        data = self.data
        targets = heldout(data, split)
        users = [u for u in data.user_ids if u in targets]
        if limit_users:
            users = users[:limit_users]
        max_k = max(self.k_values)
        results: dict[str, list[float]] = {}
        coverage_lists: dict[str, list[list[int]]] = {}
        started = time.perf_counter()

        for i, user in enumerate(users):
            parts = self.score_user(user, weights)
            for model_name, scores in parts.items():
                ranked = rank_user(scores, data, user, max_k)
                stats = results.setdefault(model_name, [0.0] * 6)
                target = targets[user]
                stats[0] += recall_at_k(ranked, target, 10)
                stats[1] += recall_at_k(ranked, target, 20)
                stats[2] += ndcg_at_k(ranked, target, 10)
                stats[3] += ndcg_at_k(ranked, target, 20)
                stats[4] += hit_rate_at_k(ranked, target, 10)
                coverage_lists.setdefault(model_name, []).append(ranked)
            if (i + 1) % 200 == 0:
                logger.info("%s: %d/%d users scored", data.domain, i + 1, len(users))

        duration = time.perf_counter() - started
        n_users = len(users)
        out: list[PerModelResult] = []
        for model_name, (r10, r20, n10, n20, hr, _unused) in results.items():
            lists = coverage_lists.get(model_name, [])  # raw item ids, best first
            positions_lists = [
                [self.catalog_positions[raw] for raw in ranked if raw in self.catalog_positions]
                for ranked in lists
            ]
            # coverage counts distinct items over integer item space (positions);
            # diversity compares THEME SETS and needs the raw item ids used by
            # the theme lookup.
            coverage = item_coverage(positions_lists, len(data.catalog))
            diversity = (
                float(
                    np.mean(
                        [intra_list_diversity(ranked, self.distance_fn, k=10) for ranked in lists]
                    )
                )
                if lists
                else 0.0
            )
            out.append(
                PerModelResult(
                    domain=data.domain,
                    model=model_name,
                    recall_at_10=r10 / n_users if n_users else 0.0,
                    recall_at_20=r20 / n_users if n_users else 0.0,
                    ndcg_at_10=n10 / n_users if n_users else 0.0,
                    ndcg_at_20=n20 / n_users if n_users else 0.0,
                    hit_rate_at_10=hr / n_users if n_users else 0.0,
                    coverage=coverage,
                    diversity=diversity,
                    n_users=n_users,
                    duration_s=duration / max(1, len(results)),
                )
            )
        out.sort(key=lambda r: r.model)
        logger.info("%s/%s evaluated %d users in %.1fs", data.domain, split, n_users, duration)
        return out

    def unseen_mask(self, user: int) -> np.ndarray:
        """True where the catalogue position was not liked in train."""
        mask = np.ones(len(self.data.catalog), dtype=bool)
        for raw in self.data.likes.get(int(user), ()):
            pos = self.data.item_pos(raw)
            if pos is not None:
                mask[pos] = False
        return mask

    def prepare_tuning(self, max_users: int | None = None) -> list[tuple[int, dict[str, np.ndarray], np.ndarray]]:
        """Cache base-model parts + unseen mask for a fixed validation sample.

        The sample is deterministic: users are ordered by id, then a seeded
        ``np.random.default_rng(cfg.seed)`` choice picks ``max_users`` of them
        (no sampling at all when the pool is smaller).
        """
        data = self.data
        users = [u for u in data.user_ids if u in data.valid_targets]
        if max_users and len(users) > max_users:
            rng = np.random.default_rng(self.cfg.seed)
            chosen = np.sort(rng.choice(len(users), size=max_users, replace=False))
            users = [users[i] for i in chosen]
        cache: list[tuple[int, dict[str, np.ndarray], np.ndarray]] = []
        for user in users:
            # includes the pop/knn/mf/content/kg parts + the fixed content+kg blend
            cache.append((user, self.score_user(user), self.unseen_mask(user)))
        self._tuning_cache = cache
        logger.info("%s: tuning cache ready for %d users", data.domain, len(cache))
        return cache

    def validation_score(self, weights: Mapping[str, float]) -> float:
        """Mean NDCG@10 of the hybrid blend on the cached validation sample.

        One candidate costs |sample| weighted sums + top-10 rankings -\n        the per-model scores are already cached by :meth:`prepare_tuning`.
        """
        cache = getattr(self, "_tuning_cache", None)
        if not cache:
            cache = self.prepare_tuning()
        total = 0.0
        for user, parts, mask in cache:
            scores = np.where(mask, hybrid.blend(parts, weights), -np.inf)
            ranked = [self.data.catalog[pos] for pos in top_k_positions(scores, 10)]
            total += ndcg_at_k(ranked, self.data.valid_targets[user], 10)
        return total / max(len(cache), 1)

    def prepare_chat_tuning(
        self, max_users: int | None = None
    ) -> list[tuple[int, dict[str, np.ndarray], np.ndarray]]:
        """Cache anonymous-chat score parts from train likes and validation masks."""
        data = self.data
        users = [user for user in data.user_ids if user in data.valid_targets]
        if max_users and len(users) > max_users:
            rng = np.random.default_rng(self.cfg.seed)
            chosen = np.sort(rng.choice(len(users), size=max_users, replace=False))
            users = [users[index] for index in chosen]

        popularity_scores = hybrid.min_max_normalise(popularity.popularity_scores(data))
        cache: list[tuple[int, dict[str, np.ndarray], np.ndarray]] = []
        for user in users:
            parts = {
                "popularity": popularity_scores,
                "content": hybrid.min_max_normalise(content.scores(data, user)),
                "kg": hybrid.min_max_normalise(kg_scoring.kg_scores(data, user)),
            }
            cache.append((user, parts, self.unseen_mask(user)))
        self._chat_tuning_cache = cache
        logger.info("%s: chat tuning cache ready for %d users", data.domain, len(cache))
        return cache

    def chat_validation_score(self, weights: Mapping[str, float]) -> float:
        """Mean validation NDCG@10 for anonymous chat queries from train likes."""
        cache = getattr(self, "_chat_tuning_cache", None)
        if cache is None:
            cache = self.prepare_chat_tuning()
        total = 0.0
        for user, parts, mask in cache:
            scores = np.where(mask, hybrid.blend(parts, weights), -np.inf)
            ranked = [self.data.catalog[pos] for pos in top_k_positions(scores, 10)]
            total += ndcg_at_k(ranked, self.data.valid_targets[user], 10)
        return total / max(len(cache), 1)


# --------------------------------------------------------------------------- #
# report writing
# --------------------------------------------------------------------------- #
def results_frame(results: list[PerModelResult]) -> pd.DataFrame:
    rows = [asdict(r) for r in results]
    return pd.DataFrame(rows)


def model_display_order() -> list[str]:
    return list(ABLATION_MODELS)


def write_reports(
    cfg: Config,
    results: list[PerModelResult],
    weights_by_domain: Mapping[str, Mapping[str, float]],
    durations: Mapping[str, float],
) -> tuple[Path, Path]:
    """Write metrics to the active profile's report and processed directories."""
    frame = results_frame(results)
    csv_path = cfg.processed_path("metrics.csv")
    frame.to_csv(csv_path, index=False)

    lines = [
        "# Ablation metrics (per domain)",
        "",
        f"Generated {pd.Timestamp.utcnow().isoformat()} on profile `{cfg.profile.name}`.",
        "",
        "Split: last positive = test, second-last = validation; ranked against the",
        "full sampled catalogue excluding train items. Diversity is mean pairwise",
        "theme-Jaccard distance inside the top-10; coverage is over the catalogue.",
        "",
    ]
    order = model_display_order()
    for domain in DOMAINS:
        sub = frame[frame["domain"] == domain].copy()
        if sub.empty:
            continue
        sub["order"] = sub["model"].map(lambda m: order.index(m) if m in order else 99)
        sub = sub.sort_values("order")
        lines.append(f"## {domain}s")
        lines.append("")
        header = (
            "| model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 |"
            " Coverage | Div@10 | fit+tune+eval s |"
        )
        lines.append(header)
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, row in sub.iterrows():
            lines.append(
                f"| {row['model']} | {row['recall_at_10']:.4f} | {row['recall_at_20']:.4f} |"
                f" {row['ndcg_at_10']:.4f} | {row['ndcg_at_20']:.4f} |"
                f" {row['hit_rate_at_10']:.4f} | {row['coverage']:.3f} |"
                f" {row['diversity']:.3f} | {durations.get(row['domain'], 0.0):.1f} |"
            )
        lines.append("")

    for domain, weights in weights_by_domain.items():
        lines.append(f"## Hybrid weights - {domain} (validation-tuned)")
        lines.append("")
        lines.append("| weight | " + " | ".join(hybrid.MODEL_NAMES) + " |")
        lines.append("|---" * (len(hybrid.MODEL_NAMES) + 1) + "|")
        lines.append(
            "| value | "
            + " | ".join(f"{weights[m]:.2f}" for m in hybrid.MODEL_NAMES)
            + " |"
        )
        lines.append("")

    md_path = cfg.report_path("metrics.md")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s and %s", md_path.name, csv_path.name)
    return md_path, csv_path


def hybrid_weights_from_cfg(cfg: Config, domain: str) -> dict[str, float]:
    """Resolve the hybrid weights for one domain.

    Lookup order: ``hybrid_weights.<profile>.<domain>`` in the generated
    ``config/tuned_weights.yaml``, then the flat fallback block in
    ``config/settings.yaml``.
    """
    tuned = cfg.load_tuned_weights(domain)
    if tuned:
        return tuned
    if not cfg.hybrid_weights:
        return dict.fromkeys(hybrid.MODEL_NAMES, 0.0)
    return {name: float(w) for name, w in cfg.hybrid_weights.items()}


def chat_weights_from_cfg(cfg: Config, domain: str) -> dict[str, float]:
    """Resolve chat weights, separate from full-history recommender weights."""
    tuned = cfg.load_tuned_chat_weights(domain)
    if tuned:
        return tuned
    full_weights = hybrid_weights_from_cfg(cfg, domain)
    return {name: float(full_weights.get(name, 0.0)) for name in CHAT_MODEL_NAMES}


def save_hybrid_weights(cfg: Config, weights: Mapping[str, float], domain: str) -> None:
    """Persist full recommender weights into ``config/tuned_weights.yaml``.

    ``settings.yaml`` is never machine-written so its hand-edited comments
    survive; chat weights are stored beside the domain block under ``chat``.
    """
    path = cfg.tuned_weights_path
    data: dict[str, Any] = load_yaml_data(path)
    node = data.setdefault(cfg.profile.name, {})
    node[domain] = {name: round(float(w), 4) for name, w in weights.items()}
    text = (
        "# GENERATED by `python -m entdec recsys` - per-profile/domain model weights,\n"
        "# validation-tuned (config/tuned_weights.yaml). Do not hand-edit;\n"
        "# settings.yaml holds the fallback weights instead.\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    )
    path.write_text(text, encoding="utf-8")
    logger.info("saved hybrid weights for %s to %s", domain, path.name)


def save_chat_weights(cfg: Config, weights: Mapping[str, float], domain: str) -> None:
    """Persist validation-tuned anonymous-chat weights without altering full weights."""
    path = cfg.tuned_weights_path
    data: dict[str, Any] = load_yaml_data(path)
    profile = data.setdefault(cfg.profile.name, {})
    chat = profile.setdefault("chat", {})
    chat[domain] = {
        name: round(float(weights.get(name, 0.0)), 4) for name in CHAT_MODEL_NAMES
    }
    text = (
        "# GENERATED by `python -m entdec recsys` - per-profile/domain hybrid and chat weights,\n"
        "# validation-tuned (config/tuned_weights.yaml). Do not hand-edit.\n"
        + yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    )
    path.write_text(text, encoding="utf-8")
    logger.info("saved chat weights for %s to %s", domain, path.name)


def tune_chat_weights(
    evaluate_fn: Callable[[Mapping[str, float]], float],
    *,
    popularity_weights: tuple[float, ...] = (0.0, 0.05, 0.1),
) -> tuple[dict[str, float], float]:
    """Tune popularity/content/KG on a simplex with popularity capped at 0.1."""
    best_weights: dict[str, float] | None = None
    best_score = float("-inf")
    for pop_weight in popularity_weights:
        remaining = round(1.0 - pop_weight, 4)
        for content_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            content_weight = round(remaining * content_fraction, 4)
            weights = {
                "popularity": pop_weight,
                "content": content_weight,
                "kg": round(remaining - content_weight, 4),
            }
            score = float(evaluate_fn(weights))
            if best_weights is None or score > best_score:
                best_weights, best_score = weights, score
    return best_weights or {"popularity": 0.05, "content": 0.475, "kg": 0.475}, best_score


def load_yaml_data(path: Path) -> dict:
    import yaml

    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run_pipeline(
    cfg: Config,
    *,
    tune: bool = True,
    limit_users: int | None = None,
    domains: tuple[str, ...] = ("movie", "game"),
) -> list[PerModelResult]:
    """Full recsys pass: fit, (tune hybrid), evaluate, report."""
    all_results: list[PerModelResult] = []
    durations: dict[str, float] = {}

    domain_weights: dict[str, dict[str, float]] = {}
    for domain in domains:
        data = load_domain(cfg, domain)
        if not data.user_ids:
            logger.warning("%s: no users with positives - skipped", domain)
            continue
        evaluator = Evaluator(cfg, data)
        fit_started = time.perf_counter()
        evaluator.fit()

        def tune_fn(_data: DomainData, weights: Mapping[str, float], _ev=evaluator) -> float:
            return _ev.validation_score(weights)

        if tune:
            evaluator.prepare_tuning(max_users=int(cfg.get("tuning.sample_users", 500)))
            result = hybrid.tune_weights(
                data,
                tune_fn,
                max_candidates=int(cfg.get("tuning.max_candidates", 400)),
            )
            weights = result.weights
            save_hybrid_weights(cfg, weights, domain)
            evaluator.prepare_chat_tuning(max_users=int(cfg.get("tuning.sample_users", 500)))
            chat_weights, chat_ndcg = tune_chat_weights(evaluator.chat_validation_score)
            save_chat_weights(cfg, chat_weights, domain)
            logger.info(
                "%s: tuned chat weights %s (validation ndcg@10=%.4f)",
                domain,
                chat_weights,
                chat_ndcg,
            )
        domain_weights[domain] = weights
        for r in evaluator.evaluate("test", weights=weights, limit_users=limit_users):
            all_results.append(r)
        durations[domain] = time.perf_counter() - fit_started

    write_reports(cfg, all_results, domain_weights, durations)
    return all_results


def run(cfg: Config, args: argparse.Namespace | None = None) -> int:  # noqa: ARG001
    """CLI entry point."""
    started = time.perf_counter()
    limit = getattr(args, "limit_users", None) if args else None
    results = run_pipeline(cfg, limit_users=limit)
    if not results:
        logger.error("no evaluation results produced - check the splits")
        return 1
    logger.info("recsys done in %.1fs - %d rows", time.perf_counter() - started, len(results))
    return 0


__all__ = [
    "ABLATION_MODELS",
    "CHAT_MODEL_NAMES",
    "Evaluator",
    "PerModelResult",
    "results_frame",
    "run",
    "run_pipeline",
    "chat_weights_from_cfg",
    "save_hybrid_weights",
    "save_chat_weights",
    "tune_chat_weights",
    "write_reports",
]
