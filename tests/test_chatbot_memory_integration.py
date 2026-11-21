"""Live-service multi-turn memory integration coverage."""

from __future__ import annotations

import pytest

from entdec.config import get_config

pytestmark = pytest.mark.integration


def _services_available() -> bool:
    cfg = get_config("tiny", use_cache=False)
    from entdec.graph.neo4j_client import Neo4jClient
    from entdec.nlp.ollama_client import OllamaClient

    try:
        ollama_ok = OllamaClient.from_config(cfg).health()
    except Exception:  # noqa: BLE001
        ollama_ok = False
    try:
        with Neo4jClient.from_config(cfg) as client:
            graph_ok = client.is_ready()
    except Exception:  # noqa: BLE001
        graph_ok = False
    return ollama_ok and graph_ok


def test_three_scripted_conversations_remain_grounded() -> None:
    if not _services_available():
        pytest.skip("Ollama and/or the Neo4j container are not available")

    from entdec.chatbot.pipeline import ChatPipeline
    from entdec.chatbot.session import ChatSession
    from entdec.llm.generator import is_vietnamese

    pipeline = ChatPipeline(get_config("tiny", use_cache=False))
    scripts = [
        [
            "Tôi thích Inception, gợi ý vài game có cốt truyện hay",
            "cho tôi thêm vài game nữa",
            "cái thứ hai là gì?",
        ],
        [
            "Gợi ý vài phim hài nhẹ nhàng",
            "cái thứ hai là gì?",
            "cho tôi thêm vài phim nữa",
        ],
        [
            "Tôi thích Hades, gợi ý phim phiêu lưu",
            "phim thứ hai là gì?",
            "giờ gợi ý game nhẹ nhàng hơn",
        ],
    ]

    for messages in scripts:
        session = ChatSession(memory_turns=4)
        first = pipeline.run(messages[0], session=session)
        assert len(first.items) >= 2
        second = pipeline.run(messages[1], session=session)
        third = pipeline.run(messages[2], session=session)
        first_ids = {(item.domain, item.item_id) for item in first.items}
        second_ids = {(item.domain, item.item_id) for item in second.items}
        if "thứ hai" in messages[1].casefold() or "second one" in messages[1].casefold():
            assert first.items[1].title in second.resolved_references
            assert second.items[0].item_id == first.items[1].item_id
        else:
            assert second.items
            assert not first_ids.intersection(second_ids)
        third_ids = {(item.domain, item.item_id) for item in third.items}
        assert not first_ids.intersection(third_ids)
        for result in (first, second, third):
            assert is_vietnamese(result.answer)
            allowed = {item.title for item in result.items} | set(result.resolved_references)
            scrubbed = result.answer
            for title in sorted(allowed, key=len, reverse=True):
                scrubbed = scrubbed.replace(title, "")
            assert all(
                title not in scrubbed
                for title in pipeline._ranker("movie").titles_by_id().values()
                if title not in allowed
            )
            assert all(
                title not in scrubbed
                for title in pipeline._ranker("game").titles_by_id().values()
                if title not in allowed
            )
