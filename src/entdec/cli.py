"""Command line interface: ``python -m entdec <step> [--profile tiny|default]``.

Every pipeline step is a sub-command. Handlers live in their phase module and
expose ``run(cfg, args) -> int``. Commands are imported lazily so that a missing
optional dependency only breaks the step that needs it.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence

from entdec.config import Config, ProfileName, load_config
from entdec.logging_setup import get_logger, setup_logging

logger = get_logger(__name__)

# name -> (python module holding run()/register(), one-line help)
COMMAND_MODULES: dict[str, tuple[str, str]] = {
    "envcheck": ("", "inspect the environment, tooling and raw datasets"),
    "data": ("entdec.data.pipeline", "load, clean, sample and split both datasets"),
    "taxonomy": ("entdec.taxonomy.pipeline", "build genres + clustered themes + theme_map.csv"),
    "texts": ("entdec.nlp.textdocs", "build item documents and chunks"),
    "embed": ("entdec.nlp.embedder", "embed item chunks with Ollama (resumable)"),
    "graph": ("entdec.graph.loader", "switch profiles and manage Neo4j graph dumps"),
    "recsys": ("entdec.recsys.evaluate", "train models, evaluate, tune weights"),
    "cross": ("entdec.recsys.cross_domain", "cross-domain persona recommendations"),
    "chat": ("entdec.chatbot.cli", "interactive chatbot (CLI)"),
    "api": ("entdec.api.main", "serve the FastAPI app"),
    "ui": ("entdec.ui.app", "run the Streamlit UI"),
    "all": ("entdec.pipeline", "run the whole pipeline for a profile"),
}

# Command specific arguments, added on top of the shared options.
_EXTRA_ARGS: dict[str, list[tuple[tuple[str, ...], dict]]] = {
    "data": [
        (("--force",), {"action": "store_true", "help": "rebuild even if outputs exist"}),
    ],
    "taxonomy": [
        (("--force",), {"action": "store_true", "help": "recompute theme clustering"}),
        (("--themes",), {"type": int, "default": None, "help": "override theme count"}),
        (("--limit",), {"type": int, "default": None, "help": "limit source tags (dev)"}),
    ],
    "texts": [
        (("--force",), {"action": "store_true", "help": "rebuild item texts"}),
    ],
    "embed": [
        (("--limit",), {"type": int, "default": None, "help": "embed at most N items"}),
        (("--force",), {"action": "store_true", "help": "ignore the resume cache"}),
        (("--domain",), {"choices": ["movie", "game"], "default": None, "help": "only one domain"}),
    ],
    "graph": [
        (
            ("action",),
            {
                "choices": [
                    "up", "down", "init", "load", "verify", "similar", "reset",
                    "status", "dump", "restore", "use",
                ],
                "help": "graph operation (use selects the active profile)",
            },
        ),
        (("--force",), {"action": "store_true", "help": "overwrite dump or recompute SIMILAR_TO"}),
    ],
    "recsys": [
        (
            ("action",),
            {
                "choices": ["train", "evaluate", "tune", "report", "all"],
                "nargs": "?",
                "default": "all",
                "help": "recsys step to run (default: all)",
            },
        ),
        (("--models",), {"type": str, "default": None, "help": "comma separated model names"}),
        (("--limit-users",), {"type": int, "default": None, "help": "evaluate at most N users"}),
    ],
    "cross": [
        (("--top-k",), {"type": int, "default": None, "help": "results per persona"}),
    ],
    "chat": [
        (("--question",), {"type": str, "default": None, "help": "ask one question and exit"}),
        (("--language",), {"type": str, "default": None, "help": "answer language (default vi)"}),
        (("--no-graph",), {"action": "store_true", "help": "skip Neo4j; use local Parquet retrieval"}),
    ],
    "api": [
        (("--host",), {"type": str, "default": "127.0.0.1"}),
        (("--port",), {"type": int, "default": 8000}),
        (("--reload",), {"action": "store_true"}),
        (("--no-graph",), {"action": "store_true", "help": "skip Neo4j; use local Parquet retrieval"}),
    ],
    "ui": [
        (("--port",), {"type": int, "default": 8501}),
        (("--no-graph",), {"action": "store_true", "help": "skip Neo4j; use local Parquet retrieval"}),
    ],
    "all": [
        (("--skip-llm",), {"action": "store_true", "help": "skip Ollama dependent steps"}),
        (("--force",), {"action": "store_true", "help": "rebuild everything from scratch"}),
        (("--no-graph",), {"action": "store_true", "help": "skip Neo4j; use local Parquet retrieval"}),
    ],
    "envcheck": [
        (("--rows",), {"action": "store_true", "help": "also count rows of raw CSV files (slow)"}),
    ],
}


def build_parser() -> argparse.ArgumentParser:
    """Build the top level argument parser."""
    parser = argparse.ArgumentParser(
        prog="entdec",
        description="Entertainment_decision - movies + games recommender pipeline.",
    )
    parser.add_argument(
        "--profile",
        choices=["tiny", "default"],
        default=None,
        help="sampling profile (default: ENTDEC_PROFILE or tiny)",
    )
    parser.add_argument("--log-level", default="INFO", help="logging level (default INFO)")

    subparsers = parser.add_subparsers(dest="command", metavar="<step>")
    for name, (_, help_text) in COMMAND_MODULES.items():
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        sub.add_argument(
            "--profile", choices=["tiny", "default"], default=argparse.SUPPRESS
        )
        sub.add_argument("--log-level", default=argparse.SUPPRESS)
        for flags, kwargs in _EXTRA_ARGS.get(name, []):
            sub.add_argument(*flags, **kwargs)
    return parser


def _envcheck_handler(cfg: Config, args: argparse.Namespace) -> int:
    from entdec.envcheck import run_env_check

    report = run_env_check(cfg)
    return 0 if report.ok else 1


_SPECIAL_HANDLERS: dict[str, Callable[[Config, argparse.Namespace], int]] = {
    "envcheck": _envcheck_handler,
}


def _resolve_handler(command: str) -> Callable[[Config, argparse.Namespace], int]:
    if command in _SPECIAL_HANDLERS:
        return _SPECIAL_HANDLERS[command]
    module_name, _ = COMMAND_MODULES[command]
    module = importlib.import_module(module_name)
    if not hasattr(module, "run"):
        raise AttributeError(f"{module_name} does not expose run(cfg, args)")
    return module.run


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.command:
        parser.print_help()
        return 2

    setup_logging(args.log_level or "INFO")
    profile: ProfileName | None = args.profile
    try:
        cfg = load_config(profile)
    except KeyError as exc:
        logger.error("%s", exc)
        return 2

    cfg.paths.ensure()

    try:
        handler = _resolve_handler(args.command)
    except (ImportError, AttributeError) as exc:
        logger.error("Command %r is not available: %s", args.command, exc)
        return 2

    logger.info("=== entdec %s (profile=%s) ===", args.command, cfg.profile.name)
    try:
        return int(handler(cfg, args) or 0)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        logger.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
