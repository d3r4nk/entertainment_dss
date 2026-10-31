"""Chatbot pipeline unit tests (spec section 8/11): rerank + grounding.

No Ollama, no Neo4j: the catalogue comes from tiny Parquet fixtures, Ollama is
mocked over HTTP with ``httpx.MockTransport`` and the graph backend is replaced
by an object that always raises, forcing the local retrieval path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import pytest

import entdec.chatbot.pipeline as pipe_mod
from entdec.chatbot.pipeline import (
    Candidate,
    ChatPipeline,
    ChatRanker,
    add_genre_evidence,
    add_same_similar_evidence,
    fuzzy_resolve,
    initialise_candidates,
    normalise_chat_weights,
)
from entdec.config import Config
from entdec.nlp.ollama_client import OllamaClient

DIM = 8
TITLES = {100: "Inception (2010)", 101: "Apples (2010)", 102: "Zebra Crossing (2010)"}
INTENT_JSON = {"domain": "movie", "reference_titles": ["Zebra Crossing"], "genres": [], "themes": ["cosy"], "keywords": [], "exclude": [], "language": "vi"}


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _mock_client(handler) -> OllamaClient:
    return OllamaClient(host="http://mock", transport=httpx.MockTransport(handler), max_retries=1)


class _BrokenGraph:
    """Stands in for GraphRetriever: constructing it fails -> local retrieval."""

    def __init__(self, cfg: Config) -> None:  # pragma: no cover - trivial
        raise ConnectionError("no graph in unit tests")


@pytest.fixture
def catalogue_cfg(tmp_path: Path, monkeypatch) -> Config:
    """Config + processed Parquet for ``load_domain('movie')`` (3 items)."""
    cfg = _catalogue_cfg_impl(tmp_path)
    monkeypatch.setattr("entdec.config.PROJECT_ROOT", tmp_path)
    yield cfg


def _catalogue_cfg_impl(tmp_path: Path) -> Config:
    import yaml as _yaml  # local import keeps the module surface small

    from entdec.config import DEFAULT_SETTINGS_PATH, load_yaml

    data = load_yaml(DEFAULT_SETTINGS_PATH)
    data.setdefault("nlp", {})["embed_dim"] = DIM
    data.setdefault("llm", {})["retrieval_k"] = 3
    data.setdefault("llm", {})["rerank_k"] = 3
    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(_yaml.safe_dump(data), encoding="utf-8")

    from entdec.config import EnvSettings, ProfileConfig, build_paths

    data_dir = tmp_path / "data"
    env = EnvSettings(
        entdec_data_dir=data_dir,
        entdec_processed_dir=data_dir / "processed",
        entdec_reports_dir=tmp_path / "reports",
        entdec_cache_dir=tmp_path / ".cache",
    )
    profile_cfg = ProfileConfig(name="tiny", **data["profiles"]["tiny"])
    cfg = Config(
        env=env,
        yaml_data=data,
        profile=profile_cfg,
        paths=build_paths(env, profile_cfg.name),
        settings_path=settings_path,
    )
    cfg.paths.ensure()
    processed = cfg.paths.processed

    pd.DataFrame({"movie_id": list(TITLES), "title": list(TITLES.values())}).to_parquet(
        processed / "movies.parquet", index=False
    )
    # only positives reach the interaction matrix
    pd.DataFrame(
        {
            "uid": [1, 1, 2],
            "movie_id": [100, 101, 102],
            "is_positive": [True, True, True],
            "split": ["train", "train", "train"],
        }
    ).to_parquet(processed / "movie_ratings.parquet", index=False)
    pd.DataFrame(
        {
            "item_id": [100, 101, 101, 102],
            "theme": ["story rich", "story rich", "cosy", "cosy"],
            "weight": [1.0, 0.9, 0.4, 1.0],
        }
    ).to_parquet(processed / "movie_themes.parquet", index=False)
    eye = np.eye(DIM, dtype=np.float32)
    pd.DataFrame(
        {
            "domain": ["movie"] * 3,
            "item_id": [100, 101, 102],
            "embedding": [eye[i].tolist() for i in range(3)],
        }
    ).to_parquet(processed / "item_embeddings.parquet", index=False)
    pd.DataFrame(
        {"domain": ["movie"] * 2, "item_id": [100, 102], "genre": ["Thriller", "Comedy"]}
    ).to_parquet(processed / "item_genres.parquet", index=False)
    return cfg


@pytest.fixture
def ranker(catalogue_cfg: Config) -> ChatRanker:
    return ChatRanker(catalogue_cfg, "movie")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_fuzzy_resolve_maps_titles(ranker: ChatRanker) -> None:
    pool = {title: raw for raw, title in ranker.titles_by_id().items()}
    resolved = fuzzy_resolve(["inception", "Zebra Crossing (2010)"], pool)
    assert resolved == {TITLES[100]: 100, TITLES[102]: 102}
    # gibberish is skipped, not guessed
    assert fuzzy_resolve(["zzzz-not-a-title"], pool) == {}


def test_initialise_candidates_merges_best_scores(ranker: ChatRanker) -> None:
    pool: dict[int, Candidate] = {}
    initialise_candidates(pool, ranker, 100, {"themes": 0.5})
    assert pool[100].title == TITLES[100]
    assert pool[100].position == 0
    initialise_candidates(pool, ranker, 100, {"themes": 0.1})
    assert pool[100].scores["themes"] == 0.5, "best score must survive"
    initialise_candidates(pool, ranker, 999, {"themes": 1.0})  # unknown id ignored
    assert len(pool) == 1


def test_genre_and_similarity_evidence(ranker: ChatRanker) -> None:
    pool: dict[int, Candidate] = {}
    initialise_candidates(pool, ranker, 100, {})
    initialise_candidates(pool, ranker, 101, {})
    add_genre_evidence(pool, ranker, ["Thriller"])
    assert pool[100].shared_genres == ["Thriller"]
    assert pool[101].shared_genres == []

    add_same_similar_evidence(pool, ranker, [102], {TITLES[102]: 102})
    # embeddings are orthogonal: 100 and 101 have no similarity to 102
    assert pool[100].similar_to == {}
    assert pool[101].similar_to == {}


def test_ranker_uses_only_anonymous_chat_weights(ranker: ChatRanker) -> None:
    assert set(ranker.weights) == {"popularity", "content", "kg"}
    total = sum(ranker.weights.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert all(weight >= 0 for weight in ranker.weights.values())


def test_zero_or_unavailable_chat_weights_use_safe_fallback() -> None:
    weights = normalise_chat_weights(
        {"popularity": 0.0, "content": 0.0, "kg": 0.0, "itemknn": 1.0, "mf": 1.0}
    )
    assert weights == pytest.approx(
        {"popularity": 0.05, "content": 0.475, "kg": 0.475}
    )
    without_content = normalise_chat_weights(
        {"popularity": 0.0, "content": 1.0, "kg": 0.0},
        {"popularity", "kg"},
    )
    assert without_content == pytest.approx({"popularity": 0.05 / 0.525, "kg": 0.475 / 0.525})


def test_no_graph_pipeline_does_not_construct_graph_retriever(
    catalogue_cfg: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pipe_mod,
        "GraphRetriever",
        lambda _cfg: pytest.fail("Neo4j must not be constructed when disabled"),
    )
    pipeline = ChatPipeline(catalogue_cfg, client=_mock_client(lambda _request: None), use_graph=False)

    references = pipeline.resolve_references(["Inception"])

    assert references == {"Inception (2010)": ("movie", 100)}


def test_rerank_ranks_pool_and_excludes_references(ranker: ChatRanker) -> None:
    pool: dict[int, Candidate] = {}
    for raw in (100, 101, 102):
        initialise_candidates(pool, ranker, raw, {})
    query = np.zeros(DIM, dtype=np.float32)
    # user liked item 101 in the past; explicitly references item 102
    ranked = ranker.rerank(
        pool,
        liked_ids=[101],
        query_vector=query,
        excluded_ids={102},
        k=3,
    )
    ranked_ids = [c.item_id for c in ranked]
    assert 101 not in ranked_ids, "item seen in likes must be excluded"
    assert 102 not in ranked_ids, "explicit exclusion must win"
    assert ranked_ids == [100]
    assert ranked[0].scores["total"] > 0


def test_rerank_sets_shared_themes(ranker: ChatRanker) -> None:
    pool: dict[int, Candidate] = {}
    for raw in (100, 101):
        initialise_candidates(pool, ranker, raw, {})
    # an empty user profile claims no shared themes (honest evidence)
    ranked = ranker.rerank(pool, liked_ids=[], query_vector=None, k=2)
    assert all(c.shared_themes == [] for c in ranked)

    # extra_themes is what run() passes from the parsed intent
    ranked = ranker.rerank(
        pool, liked_ids=[], query_vector=None, extra_themes=["story rich", "cosy"], k=2
    )
    themes = {c.item_id: c.shared_themes for c in ranked}
    assert themes[100] == ["story rich"]
    # item 101's own weights order the listed themes (0.9 vs 0.4)
    assert themes[101] == ["story rich", "cosy"]


# --------------------------------------------------------------------------- #
# end to end: grounding (the mocked LLM may only mention retrieved items)
# --------------------------------------------------------------------------- #
class _EchoHandler:
    """Answers /api/embed deterministically and /api/generate by echoing the
    candidate lines of the prompt (so the test can verify grounding)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.read())
        if request.url.path == "/api/embed":
            vectors = []
            for _text in body["input"]:
                seed = 1.0
                vectors.append([seed / (i + 1) for i in range(DIM)])
            return httpx.Response(200, json={"embeddings": vectors})
        assert body["stream"] is False
        if request.url.path == "/api/chat":
            # answer turn: quote exactly the candidate titles given in the prompt,
            # in the shape the real prompt asks for (one "TITLE | sentence" line)
            prompt = body["messages"][1]["content"]
            self.prompts.append(prompt)
            titles = re.findall(r"^([^|\n]+) \| ", prompt, flags=re.MULTILINE)
            lines = [
                f"{rank}. {title.strip()} | Phù hợp với yêu cầu của bạn"
                for rank, title in enumerate(titles, start=1)
            ]
            return httpx.Response(200, json={"message": {"content": "\n".join(lines)}})
        self.prompts.append(body["prompt"])
        assert body.get("format") == "json"
        return httpx.Response(200, json={"response": json.dumps(INTENT_JSON)})


def test_run_grounding_no_unretrieved_items(catalogue_cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(pipe_mod, "GraphRetriever", _BrokenGraph)
    handler = _EchoHandler()
    pipeline = ChatPipeline(catalogue_cfg, client=_mock_client(handler))

    result = pipeline.run(" phim nhẹ nhàng như Zebra Crossing ")

    assert result.used_graph is False
    assert result.intent_llm is True
    assert result.answer_llm is True
    assert result.domains_queried == ["movie"]
    retrieved_titles = {item.title for item in result.items}
    # grounding: the mocked LLM echoes the evidence lines, so every title in
    # the answer belongs to a retrieved title, nothing else leaks in
    for title in retrieved_titles:
        assert title in result.answer
    for title in TITLES.values():
        if title not in retrieved_titles:
            assert title not in result.answer, "no unretrieved title may leak in"
    # the explicitly referenced (disliked) item must not be recommended
    assert TITLES[102] not in retrieved_titles


def test_run_no_candidates_returns_polite_no_answer(catalogue_cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(pipe_mod, "GraphRetriever", _BrokenGraph)

    def _handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        if request.url.path == "/api/embed":
            return httpx.Response(200, json={"embeddings": [[0.0] * DIM]})
        if body.get("format") == "json":
            payload = {**INTENT_JSON, "themes": ["nonexistent-theme"], "reference_titles": []}
            return httpx.Response(200, json={"response": json.dumps(payload)})
        return httpx.Response(200, json={"response": ""})

    pipeline = ChatPipeline(catalogue_cfg, client=_mock_client(_handler))
    result = pipeline.run("gợi ý gì đó đi")
    # nothing matches the request, but the local backend still backfills with
    # popular items rather than returning nothing; the empty LLM reply sends
    # the turn to the deterministic fallback template
    assert result.items, "popular backfill keeps the chat useful"
    assert result.answer_llm is False, "an empty LLM answer must not be returned"
    assert all(item.title in result.answer for item in result.items)


def test_recommend_uses_rule_intent_without_llm(catalogue_cfg: Config, monkeypatch) -> None:
    monkeypatch.setattr(pipe_mod, "GraphRetriever", _BrokenGraph)
    handler = _EchoHandler()
    pipeline = ChatPipeline(catalogue_cfg, client=_mock_client(handler))

    result = pipeline.recommend("Muốn xem phim hài nhẹ nhàng cuối tuần", domain="movie")

    assert result.answer_llm is False
    assert result.intent.parsed_by == "rules"
    assert result.domains_queried == ["movie"]
    assert handler.prompts == [], "recommend must not call the generation model"
    assert result.items, "the catalogue is non-empty so at least one item is recommended"
