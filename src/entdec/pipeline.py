"""``python -m entdec all --profile tiny`` - the whole pipeline in one command.

Each phase keeps its own ``run(cfg, args)`` entry point; this module only
sequences them in the spec's order (section 9) and reports how long each step
took. Every step is idempotent and honours its cache, so the command can be
re-run to pick up where a previous run stopped (`--force` rebuilds from
scratch). Ollama is used only by ``taxonomy`` and ``embed``; both are skipped
with ``--skip-llm``, which then relies on the artefacts of an earlier run.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

from entdec.config import Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class Step:
    """One phase of the pipeline."""

    name: str
    run: Callable[[], int]


@dataclass
class StepOutcome:
    """Timing and exit code of a finished phase."""

    name: str
    seconds: float
    code: int


def _graph_client(cfg: Config):
    from entdec.graph.neo4j_client import Neo4jClient

    return Neo4jClient.from_config(cfg)


def build_steps(
    cfg: Config, *, force: bool = False, skip_llm: bool = False, no_graph: bool = False
) -> list[Step]:
    """Sequence the phases; LLM-dependent ones are optional."""
    from entdec.data.pipeline import run as data_run
    from entdec.graph.store import ensure_graph
    from entdec.nlp.embedder import run as embed_run
    from entdec.nlp.textdocs import run as texts_run
    from entdec.recsys.cross_domain import run as cross_run
    from entdec.recsys.evaluate import run as recsys_run
    from entdec.taxonomy.pipeline import run as taxonomy_run

    steps: list[Step] = [
        Step("data", lambda: data_run(cfg, argparse.Namespace(force=force))),
    ]

    if skip_llm:
        logger.warning("--skip-llm: taxonomy and embed are skipped (existing artefacts are reused)")
    else:
        from entdec.nlp.ollama_client import OllamaClient

        client = OllamaClient.from_config(cfg)
        steps.append(
            Step(
                "taxonomy",
                lambda: taxonomy_run(
                    cfg,
                    argparse.Namespace(force=force, themes=None, limit=None),
                    client=client,
                ),
            )
        )
    steps.append(Step("texts", lambda: texts_run(cfg, argparse.Namespace(force=force))))
    if not skip_llm:
        steps.append(
            Step(
                "embed",
                lambda: embed_run(
                    cfg,
                    argparse.Namespace(force=force, limit=None, domain=None),
                    client=client,
                ),
            )
        )

    graph_disabled = no_graph or cfg.env.entdec_no_graph
    if graph_disabled:
        logger.info("Neo4j disabled; all will use local Parquet data")
    else:
        def ensure_profile_graph() -> int:
            from entdec.graph.store import GraphStoreError

            try:
                ensure_graph(cfg, force=force)
            except GraphStoreError as exc:
                logger.error("unable to select graph profile: %s", exc)
                return 1
            return 0

        steps.append(Step("graph use", ensure_profile_graph))
    steps.extend(
        [
            Step(
                "recsys",
                lambda: recsys_run(
                    cfg, argparse.Namespace(limit_users=None, models=None, action="all")
                ),
            ),
            Step("cross", lambda: cross_run(cfg, argparse.Namespace(top_k=None))),
        ]
    )
    return steps


def run(cfg: Config, args: argparse.Namespace | None = None) -> int:
    """Run every phase in order; the first failure stops the pipeline."""
    force = bool(getattr(args, "force", False))
    skip_llm = bool(getattr(args, "skip_llm", False))
    no_graph = bool(getattr(args, "no_graph", False)) or cfg.env.entdec_no_graph
    started = time.perf_counter()
    outcomes: list[StepOutcome] = []
    for step in build_steps(cfg, force=force, skip_llm=skip_llm, no_graph=no_graph):
        logger.info("=== %s ===", step.name)
        step_started = time.perf_counter()
        code = int(step.run() or 0)
        elapsed = time.perf_counter() - step_started
        outcomes.append(StepOutcome(step.name, elapsed, code))
        logger.info("=== %s finished in %.1fs (exit %d) ===", step.name, elapsed, code)
        if code:
            logger.error("step %r failed with exit code %d - stopping", step.name, code)
            break
    total = time.perf_counter() - started
    logger.info("pipeline summary (profile=%s):", cfg.profile.name)
    for outcome in outcomes:
        logger.info("  %-14s %7.1fs  exit %d", outcome.name, outcome.seconds, outcome.code)
    logger.info("  %-14s %7.1fs", "TOTAL", total)
    return 0 if all(outcome.code == 0 for outcome in outcomes) and outcomes else 1


__all__ = ["Step", "StepOutcome", "build_steps", "run"]
