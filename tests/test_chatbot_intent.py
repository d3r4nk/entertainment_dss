"""Unit tests for the chatbot's intent parsing and explanation generation.

Ollama is mocked over HTTP with ``httpx.MockTransport`` (spec section 11:
unit tests mock Ollama; real-service runs are ``integration``).
"""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from entdec.llm import generator
from entdec.llm import intent as intent_mod
from entdec.llm.intent import Intent, canonical_genres, parse_intent, rules_intent
from entdec.nlp.ollama_client import OllamaClient

GOOD_INTENT = {
    "domain": "game",
    "reference_titles": ["Inception"],
    "genres": [],
    "themes": ["story rich"],
    "keywords": ["cốt truyện"],
    "exclude": [],
    "language": "vi",
}


def make_client(handler: Callable[[httpx.Request], httpx.Response]) -> OllamaClient:
    return OllamaClient(
        host="http://mock",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )


def intent_handler(payload: dict | None, *, fail_request: bool = False):
    """Handler that answers /api/generate with an intent-shaped JSON."""

    def _handle(request: httpx.Request) -> httpx.Response:
        if fail_request:
            return httpx.Response(500, text="boom")
        body = json.loads(request.read())
        assert body["stream"] is False
        assert body.get("format") == "json"
        assert body["options"].get("temperature") == 0.0
        assert body.get("think") is False
        if payload is None:
            return httpx.Response(200, json={"response": "I am not JSON at all."})
        return httpx.Response(200, json={"response": json.dumps(payload)})

    return _handle


def gen_handler(answer: str, *, code: int = 200):
    """Handler for the explanation turn (``/api/chat`` because of the prefill)."""

    def _handle(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        assert request.url.path == "/api/chat"
        assert body["options"].get("temperature") == 0.2
        assert body.get("think") is False
        assert body["messages"][-1]["role"] == "assistant"  # prefilled reply start
        prefill = body["messages"][-1]["content"]
        # the answer arrives without the prefilled prefix
        return httpx.Response(code, json={"message": {"content": answer[len(prefill) :]}})

    return _handle


# --------------------------------------------------------------------------- #
# pydantic schema
# --------------------------------------------------------------------------- #
def test_intent_defaults_and_normalisation() -> None:
    parsed = Intent.model_validate(
        {"domain": "FILMS", "reference_titles": "Black Swan", "language": "vietnamese"}
    )
    assert parsed.domain == "movie"
    assert parsed.reference_titles == ["Black Swan"]
    assert parsed.language == "vi"
    assert parsed.parsed_by == "rules"


def test_intent_rejects_bad_domain_shape() -> None:
    with pytest.raises(ValueError):  # pydantic: int not coercible to domain literal
        Intent.model_validate({"domain": 42})


# --------------------------------------------------------------------------- #
# LLM parsing: success, retry, fallback
# --------------------------------------------------------------------------- #
def test_parse_intent_valid_json() -> None:
    client = make_client(intent_handler(GOOD_INTENT))
    result = parse_intent(client, "Tôi thích Inception, gợi ý vài game có cốt truyện hay")
    assert result.parsed_by == "llm"
    assert result.domain == "game"
    assert "Inception" in result.reference_titles


def test_parse_intent_retries_once_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            return httpx.Response(200, json={"response": "not { broken"})
        return intent_handler(GOOD_INTENT)(request)

    client = make_client(handler)
    result = parse_intent(client, "nhạc phim buồn")
    assert calls["n"] == 2
    assert result.domain == "game"


def test_parse_intent_falls_back_to_rules_after_two_failures() -> None:
    client = make_client(intent_handler(None))
    result = parse_intent(client, "Muốn xem phim hài nhẹ nhàng cuối tuần")
    assert result.parsed_by == "rules"
    assert result.domain == "movie"


def test_parse_intent_unreachable_ollama_uses_rules() -> None:
    client = make_client(intent_handler(GOOD_INTENT, fail_request=True))
    result = parse_intent(client, "game hành động nào hay?")
    assert result.parsed_by == "rules"


def test_llm_json_embedded_in_prose_is_extrated() -> None:
    client = OllamaClient(
        host="http://mock",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"response": "Here you go:\n```json\n" + json.dumps(GOOD_INTENT) + "\n```"},
            )
        ),
        max_retries=1,
    )
    result = parse_intent(client, "any question")
    assert result.reference_titles == ["Inception"]


# --------------------------------------------------------------------------- #
# rule fallback
# --------------------------------------------------------------------------- #
def test_rules_intent_detects_domain_language_genres() -> None:
    parsed = rules_intent("Tôi thích Inception, gợi ý vài game có cốt truyện hay")
    assert parsed.domain == "game"
    assert parsed.language == "vi"
    assert parsed.themes == ["story rich"]

    parsed_movie = rules_intent("Muốn xem phim hài nhẹ nhàng cuối tuần")
    assert parsed_movie.domain == "movie"
    assert "Comedy" in parsed_movie.genres
    assert parsed_movie.themes == ["cosy"]


def test_rules_intent_english_default() -> None:
    parsed = rules_intent("Recommend a fancy open world game")
    assert parsed.language == "en"
    assert parsed.domain == "game"


def test_canonical_genres_maps_known_words() -> None:
    intent = Intent(genres=["Comedy", "kinh dị", "nonsense-word"], domain="movie")
    assert canonical_genres(intent) == ["Comedy", "Horror"]


def test_normalize_theme_names_lowercases_and_dedups() -> None:
    names = intent_mod.normalize_theme_names(["Story Rich", "story rich", "  Psychological "])
    assert names == ["story rich", "psychological"]


def test_normalize_theme_names_ignores_unknown_without_fuzzy_hit() -> None:
    mapped = intent_mod.normalize_theme_names(["zzzz-unknown"], known=["story rich"])
    assert mapped == ["zzzz-unknown"]


# --------------------------------------------------------------------------- #
# deterministic fallback for the explanation
# --------------------------------------------------------------------------- #
def test_fallback_answer_lists_only_retrieved() -> None:
    evidence = ["Journey: themes: exploration (match 0.92)", "GRIS: themes: exploration (match 0.41)"]
    answer = generator.fallback_answer(evidence, "vi")
    assert "Journey" in answer and "GRIS" in answer
    assert "Portal 2" not in answer  # no unretrieved item is ever mentioned


def test_generate_answer_keeps_only_parsed_candidate_lines() -> None:
    reply = (
        "Okay, let's plan this answer.\n"
        "Life is Strange | Câu chuyện tuổi trẻ đầy cảm xúc, chọn lựa khó khăn.\n"
        "Wait, I should double-check the evidence.\n"
    )
    client = make_client(gen_handler(reply))
    evidence = ["Life is Strange: chủ đề story rich; điểm 0.80"]
    answer, used_fallback = generator.generate_answer(
        client, "số 1 game story rich", evidence, "vi", ["Life is Strange"]
    )
    assert answer == "1. Life is Strange: Câu chuyện tuổi trẻ đầy cảm xúc, chọn lựa khó khăn."
    assert used_fallback is False


def test_generate_answer_rejects_meta_reasoning_and_english() -> None:
    # lines that talk about the task, or are written in English, are unusable
    reply = (
        "Life is Strange | The user wants story rich games, so pick this candidate.\n"
        "Journey | A short emotional journey through a desert.\n"
    )
    client = make_client(gen_handler(reply))
    evidence = ["Life is Strange: chủ đề story rich", "Journey: chủ đề exploration"]
    answer, used_fallback = generator.generate_answer(
        client, "gợi ý game", evidence, "vi", ["Life is Strange", "Journey"]
    )
    assert used_fallback is True
    assert "Journey" in answer


def test_generate_answer_falls_back_on_error() -> None:
    client = make_client(gen_handler("ignored", code=500))
    evidence = ["Journey: themes: exploration (match 0.62)"]
    answer, used_fallback = generator.generate_answer(
        client, "gợi ý", evidence, "vi"
    )
    assert used_fallback is True
    assert "Journey" in answer  # the retrieved title is still quoted, nothing invented
