"""``python -m entdec --profile tiny chat`` - CLI for the chatbot pipeline."""

from __future__ import annotations

import argparse

from entdec.chatbot.session import ChatSession
from entdec.config import Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)


def _print_result(result) -> None:
    """Friendly terminal output."""
    print()
    print(result.answer)
    print()
    if result.items:
        print("Top items (evidence):")
        for i, item in enumerate(result.items[:10], start=1):
            shared = ", ".join(item.shared_themes[:3]) or ", ".join(item.shared_genres[:3])
            tail = f"  [{shared}]" if shared else ""
            print(f"  {i}. {item.title} ({result.domains_queried[0] if len(result.domains_queried) == 1 else 'movie/game'}){tail}")
    meta = []
    if not result.used_graph:
        meta.append("Neo4j offline -> local parquet retrieval")
    if not result.intent_llm:
        meta.append("intent = keyword rules")
    if not result.answer_llm:
        meta.append("answer = deterministic template")
    for line in meta:
        print(f"  ({line})")
    print(f"  timing: {', '.join(f'{k}={v:.1f}s' for k, v in result.durations.items())}")


def run(cfg: Config, args: argparse.Namespace | None = None) -> int:
    """Interactive chat; ``--question`` runs a single turn and exits."""
    from entdec.chatbot.pipeline import ChatPipeline
    from entdec.graph.store import GraphStoreError, ensure_graph

    question = getattr(args, "question", None) if args else None
    language = getattr(args, "language", None) if args else None
    graph_enabled = not (
        bool(getattr(args, "no_graph", False)) or cfg.env.entdec_no_graph
    )
    try:
        ensure_graph(cfg, no_graph=not graph_enabled)
    except GraphStoreError as exc:
        logger.error("unable to select graph profile: %s", exc)
        return 1
    pipeline = ChatPipeline(cfg, use_graph=graph_enabled)
    memory_enabled = cfg.memory_enabled and not bool(getattr(args, "no_memory", False))

    if question:
        session = ChatSession(memory_turns=int(cfg.get("llm.memory_turns", 4))) if memory_enabled else None
        result = pipeline.run(question, language=language, session=session)
        _print_result(result)
        return 0

    print("Entertainment chatbot - ask about movies or games ('exit' to stop).")
    if memory_enabled:
        print("Memory is active for this chat. Commands: /reset, /history.")
    else:
        print("Conversation memory is disabled for this chat.")
    print("Sample: 'Tôi thích Inception, gợi ý vài game có cốt truyện hay'")
    session = ChatSession(memory_turns=int(cfg.get("llm.memory_turns", 4))) if memory_enabled else None
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in {"exit", "quit", ":q"}:
            return 0
        if line.lower() == "/reset":
            if session is not None:
                session.reset()
                print("Conversation reset.")
            else:
                print("Conversation memory is disabled.")
            continue
        if line.lower() == "/history":
            if session is not None and session.history_text:
                print(session.history_text)
            else:
                print("No conversation history.")
            continue
        if not line:
            continue
        try:
            result = pipeline.run(line, language=language, session=session)
        except Exception as exc:  # noqa: BLE001 - keep chatting on failures
            print(f"(error: {exc})")
            continue
        _print_result(result)


__all__ = ["run"]
