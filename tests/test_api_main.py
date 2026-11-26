"""FastAPI unit tests (spec section 8/11): /health, /recommend, /chat.

The ChatPipeline is replaced by a deterministic stub; Neo4j/Ollama health is
monkeypatched, so no real service is touched (that is the integration suite).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest


class _FakePipeline:
    """Records calls and returns a minimal ChatResult-shaped mapping."""

    def __init__(self) -> None:
        self.chat_calls: list[str] = []
        self.sessions: list[object] = []
        self.rec_calls: list[dict[str, object]] = []

    def run(
        self,
        message: str,
        *,
        language: str | None = None,
        top_k: int | None = None,
        session=None,
    ):
        self.chat_calls.append(message)
        self.sessions.append(session)
        return SimpleNamespace(
            as_dict=lambda: {
                "question": message,
                "language": language or "vi",
                "answer": "answer for " + message,
                "used_graph": True,
                "intent_llm": True,
                "answer_llm": True,
                "items": [
                    {"id": i, "title": f"item{i}", "score": 0.5, "evidence": f"item{i}: match"}
                    for i in range(5)
                ],
            }
        )

    def recommend(self, query: str, *, domain: str | None = None, k: int | None = None):
        self.rec_calls.append({"query": query, "domain": domain, "k": k})
        return SimpleNamespace(
            as_dict=lambda: {
                "question": query,
                "items": [
                    {"id": i, "title": f"g{i}", "score": 1.0 - i / 10, "evidence": f"g{i}"}
                    for i in range(5)
                ],
            }
        )


@pytest.fixture
def pipeline_api(make_config, monkeypatch):
    """FastAPI TestClient with a stubbed pipeline and healthy-looking services."""
    from fastapi.testclient import TestClient

    from entdec.api.main import create_app
    from entdec.graph import neo4j_client
    from entdec.nlp.ollama_client import OllamaClient

    def _make(*, neo4j_ok: bool = True, no_graph: bool = False):
        monkeypatch.setattr(OllamaClient, "health", lambda self: True)
        monkeypatch.setattr(
            OllamaClient,
            "loaded_models",
            lambda self: [{"name": "qwen3:4b"}, {"name": "qwen3-embedding:4b"}],
        )
        if neo4j_ok:
            monkeypatch.setattr(neo4j_client.Neo4jClient, "is_ready", lambda self: True, raising=False)
        else:
            def _no_neo4j(cls, cfg_):
                raise ConnectionError("no neo4j in unit tests")

            monkeypatch.setattr(neo4j_client.Neo4jClient, "from_config", classmethod(_no_neo4j))

        app = create_app(make_config("tiny"), no_graph=no_graph)
        pipeline = _FakePipeline()
        app.state.pipeline = pipeline
        return TestClient(app), pipeline

    yield _make
    # monkeypatch fixture undoes everything at teardown


def test_health_ok(pipeline_api) -> None:
    client, _pipeline = pipeline_api()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["neo4j"] is True and body["ollama"] is True
    assert body["profile"] == "tiny"


def test_poster_endpoint_returns_image_and_rejects_other_domains(pipeline_api) -> None:
    from concurrent.futures import Future

    client, _pipeline = pipeline_api(no_graph=True)
    requested = []

    class _PosterService:
        def poster_for(self, item):
            requested.append(item)
            image = Future()
            image.set_result(b"poster-bytes")
            return image

    client.app.state.recommendation_media = _PosterService()

    response = client.get("/media/movie/123")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "public, max-age=3600"
    assert response.content == b"poster-bytes"
    assert requested == [{"domain": "movie", "id": "123"}]
    assert client.get("/media/book/123").status_code == 404


def test_health_degraded_when_ollama_down(pipeline_api, monkeypatch) -> None:
    from entdec.nlp.ollama_client import OllamaClient

    client, _pipeline = pipeline_api()
    monkeypatch.setattr(OllamaClient, "health", lambda self: False)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["ollama"] is False


def test_health_degraded_without_neo4j(pipeline_api) -> None:
    client, _pipeline = pipeline_api(neo4j_ok=False)
    body = client.get("/health").json()
    assert body["status"] == "degraded"
    assert body["neo4j"] is False


def test_health_skips_neo4j_when_graph_is_disabled(pipeline_api) -> None:
    client, _pipeline = pipeline_api(neo4j_ok=False, no_graph=True)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["neo4j"] is False
    assert body["graph_enabled"] is False


def test_chat_calls_pipeline_and_requires_message(pipeline_api) -> None:
    client, pipeline = pipeline_api()
    response = client.post("/chat", json={"message": "Tôi thích Inception, gợi ý game story rich"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("answer for")
    assert pipeline.chat_calls == ["Tôi thích Inception, gợi ý game story rich"]

    assert client.post("/chat", json={"message": "   "}).status_code == 400


def test_chat_session_round_trip_and_delete(pipeline_api) -> None:
    client, pipeline = pipeline_api()
    first = client.post("/chat", json={"message": "recommend a movie"}).json()
    session_id = first["session_id"]
    second = client.post(
        "/chat", json={"message": "something else", "session_id": session_id}
    ).json()
    assert second["session_id"] == session_id
    assert pipeline.sessions[0] is pipeline.sessions[1]

    deleted = client.delete(f"/session/{session_id}")
    assert deleted.status_code == 200
    assert deleted.json() == {"session_id": session_id, "deleted": True}


def test_chat_logs_do_not_include_user_text(pipeline_api, caplog) -> None:
    caplog.set_level("INFO")
    client, _pipeline = pipeline_api()
    private_text = "private phrase for logging test"
    client.post("/chat", json={"message": private_text})
    assert private_text not in caplog.text


def test_session_store_expires_and_caps_sessions(monkeypatch) -> None:
    import entdec.api.main as api_main

    now = [100.0]
    monkeypatch.setattr(api_main.time, "monotonic", lambda: now[0])
    store = api_main.SessionStore(ttl_minutes=1, max_sessions=1, memory_turns=4)
    original = store.get("old")
    now[0] += 61
    assert store.get("old") is not original
    store.get("new")
    assert "old" not in store._sessions


def test_chat_without_memory_returns_null_session_id(pipeline_api, monkeypatch, make_config) -> None:
    from fastapi.testclient import TestClient

    from entdec.api.main import create_app

    cfg = make_config("tiny")
    cfg.env.entdec_memory_enabled = False
    app = create_app(cfg, no_graph=True)
    pipeline = _FakePipeline()
    app.state.pipeline = pipeline
    response = TestClient(app).post("/chat", json={"message": "hello"})
    assert response.status_code == 200
    assert response.json()["session_id"] is None
    assert pipeline.sessions == [None]


def test_recommend_truncates_and_validates(pipeline_api) -> None:
    client, pipeline = pipeline_api()
    response = client.post(
        "/recommend", json={"query": "game story rich", "domain": "game", "k": 2}
    )
    assert response.status_code == 200
    items = response.json()["items"]
    assert len(items) == 2, "k=2 truncates the reranked list"
    scores = [item["score"] for item in items]
    assert scores == sorted(scores, reverse=True), "ranked best-first"
    assert pipeline.rec_calls[0]["domain"] == "game"

    assert client.post("/recommend", json={"query": "x", "domain": "anime"}).status_code == 400
    assert client.post("/recommend", json={"query": " "}).status_code == 400
    assert client.post("/recommend", json={"query": "x", "k": 0}).status_code == 422
