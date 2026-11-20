"""Service-backed multi-turn memory evaluation."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import yaml

from entdec.config import Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)


def _grounding_violations(answer: str, allowed: set[str], catalogue: set[str]) -> int:
    scrubbed = answer
    for title in sorted(allowed, key=len, reverse=True):
        scrubbed = scrubbed.replace(title, "")
    return int(any(title in scrubbed for title in catalogue - allowed))


def evaluate(cfg: Config) -> Path:
    """Run all scripted conversations against configured Ollama/Neo4j services."""
    data_path = cfg.path("config", "chat_eval_conversations.yaml")
    conversations = yaml.safe_load(data_path.read_text(encoding="utf-8"))["conversations"]

    from entdec.chatbot.pipeline import ChatPipeline
    from entdec.chatbot.session import ChatSession

    pipeline = ChatPipeline(cfg)
    if not pipeline.client.health():
        raise RuntimeError("Ollama is unavailable; chat memory evaluation needs live services")

    catalogue = set()
    for domain in ("movie", "game"):
        catalogue.update(pipeline._ranker(domain).titles_by_id().values())

    ref_hits = ref_total = 0
    repeat_items = recommendation_items = 0
    switch_hits = switch_total = 0
    grounding_errors = fallback_turns = total_turns = 0
    graph_turns = 0
    latencies: list[float] = []

    for conversation in conversations:
        session = ChatSession(memory_turns=int(cfg.get("llm.memory_turns", 4)))
        previous_result = None
        recommended_before: set[tuple[str, int]] = set()
        final_result = None
        for index, question in enumerate(conversation["turns"]):
            result = pipeline.run(question, language=conversation.get("language"), session=session)
            final_result = result
            total_turns += 1
            graph_turns += int(result.used_graph)
            latencies.append(float(result.durations.get("total_s", 0.0)))
            fallback_turns += int(not result.answer_llm)
            allowed = {item.title for item in result.items} | set(result.resolved_references)
            grounding_errors += _grounding_violations(result.answer, allowed, catalogue)

            if index in conversation.get("recommendation_turns", []):
                current = {(item.domain, item.item_id) for item in result.items}
                repeat_items += len(current & recommended_before)
                recommendation_items += len(current)
                recommended_before.update(current)

            if index == 1:
                ref_total += 1
                position = int(conversation.get("expected_reference_position", 2))
                if previous_result is not None and len(previous_result.items) >= position:
                    expected = previous_result.items[position - 1].title
                    ref_hits += int(expected in result.resolved_references)

            previous_result = result

        expected_domain = conversation.get("expected_topic_domain")
        if expected_domain:
            switch_total += 1
            switch_hits += int(
                final_result is not None and final_result.domains_queried == [expected_domain]
            )

    avg_latency = statistics.mean(latencies) if latencies else 0.0
    p95_latency = sorted(latencies)[min(len(latencies) - 1, int(len(latencies) * 0.95))] if latencies else 0.0
    repeat_rate = repeat_items / recommendation_items if recommendation_items else 0.0
    ref_accuracy = ref_hits / ref_total if ref_total else 0.0
    switch_accuracy = switch_hits / switch_total if switch_total else 0.0
    fallback_rate = fallback_turns / total_turns if total_turns else 0.0

    report = [
        f"# Chat memory evaluation ({cfg.profile.name})",
        "",
        f"Scripted conversations: {len(conversations)} ({total_turns} turns)",
        (
            f"Services: live Ollama; Neo4j graph used for {graph_turns}/{total_turns} turns, "
            f"local Parquet used for {total_turns - graph_turns}/{total_turns}."
        ),
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Reference-resolution accuracy | {ref_hits}/{ref_total} ({ref_accuracy:.1%}) |",
        f"| Repeat-recommendation rate | {repeat_items}/{recommendation_items} ({repeat_rate:.1%}) |",
        f"| Topic-switch correctness | {switch_hits}/{switch_total} ({switch_accuracy:.1%}) |",
        f"| Grounding violations | {grounding_errors} |",
        f"| Fallback-template rate | {fallback_turns}/{total_turns} ({fallback_rate:.1%}) |",
        f"| Mean latency per turn | {avg_latency:.2f} s |",
        f"| P95 latency per turn | {p95_latency:.2f} s |",
        "",
        "Per-turn latencies (seconds): " + ", ".join(f"{value:.2f}" for value in latencies),
        "",
        "No user messages or session contents are persisted by this command beyond this report.",
    ]
    output = Path(cfg.paths.reports) / "chat_memory_eval.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report) + "\n", encoding="utf-8")
    return output


def run(cfg: Config, _args: argparse.Namespace | None = None) -> int:
    try:
        output = evaluate(cfg)
    except Exception as exc:
        logger.error("chat memory evaluation failed: %s", exc)
        return 1
    print(f"Chat memory evaluation written to {output}")
    return 0


__all__ = ["evaluate", "run"]
