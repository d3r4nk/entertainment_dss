"""FastAPI service (section 8): ``python -m entdec --profile tiny api``.

Endpoints:
* ``GET  /health``    - Neo4j + Ollama + required models.
* ``POST /recommend`` - retrieval + hybrid rerank only (no LLM phrasing),
  each item with its code-built evidence.
* ``POST /chat``      - the full chatbot pipeline; the answer is phrased by
  qwen3:4b from the retrieved evidence only.

One :class:`ChatPipeline` is cached on ``app.state`` (graph client, parquet
data and fitted rerankers are reused across requests). A module lock keeps
embedding/generation out of each other's way so only one Ollama model is
resident in VRAM at a time.
"""

from __future__ import annotations

import argparse
import threading
import time
import uuid
from collections import OrderedDict

from pydantic import BaseModel, Field

from entdec.chatbot.session import ChatSession

_LLM_LOCK = threading.Lock()


class SessionStore:
    """Bounded in-memory session store with sliding TTL."""

    def __init__(self, ttl_minutes: int, max_sessions: int, memory_turns: int) -> None:
        self.ttl_seconds = max(1, int(ttl_minutes)) * 60
        self.max_sessions = max(1, int(max_sessions))
        self.memory_turns = max(1, int(memory_turns))
        self._sessions: OrderedDict[str, tuple[ChatSession, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str | None = None) -> ChatSession:
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if session_id is not None and session_id in self._sessions:
                session, _ = self._sessions.pop(session_id)
                self._sessions[session_id] = (session, now)
                return session
            new_id = session_id or str(uuid.uuid4())
            session = ChatSession(session_id=new_id, memory_turns=self.memory_turns)
            self._sessions[new_id] = (session, now)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def _expire(self, now: float) -> None:
        expired = [
            key
            for key, (_session, touched) in self._sessions.items()
            if now - touched >= self.ttl_seconds
        ]
        for key in expired:
            self._sessions.pop(key, None)


class ChatIn(BaseModel):
    """Payload for ``/chat``."""

    message: str
    language: str | None = Field(default=None)
    session_id: str | None = Field(default=None)


class RecommendIn(BaseModel):
    """Payload for ``/recommend``."""

    query: str
    domain: str | None = Field(default=None)  # optional: movie | game
    k: int = Field(default=10, ge=1, le=20)


def create_app(cfg=None, *, no_graph: bool = False):
    """Build the FastAPI application (state cached on ``app.state``)."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("fastapi is required: pip install fastapi uvicorn") from exc

    from entdec.chatbot.pipeline import ChatPipeline
    from entdec.config import get_config
    from entdec.graph.neo4j_client import Neo4jClient
    from entdec.logging_setup import get_logger
    from entdec.nlp.ollama_client import OllamaClient

    logger = get_logger(__name__)
    cfg = cfg or get_config()
    graph_enabled = not (no_graph or cfg.env.entdec_no_graph)

    app = FastAPI(title="Entertainment_decision recommender", version="0.1.0")
    app.state.graph_enabled = graph_enabled
    app.state.pipeline = ChatPipeline(cfg, use_graph=graph_enabled)
    app.state.sessions = SessionStore(
        ttl_minutes=int(cfg.get("llm.session_ttl_minutes", 60)),
        max_sessions=int(cfg.get("llm.max_sessions", 100)),
        memory_turns=int(cfg.get("llm.memory_turns", 4)),
    )
    app.state.memory_enabled = cfg.memory_enabled

    @app.get("/health")
    def health() -> dict:
        neo4j_ok = False
        if app.state.graph_enabled:
            try:
                with Neo4jClient.from_config(cfg) as client:
                    neo4j_ok = client.is_ready()
            except Exception:  # noqa: BLE001 - health must not throw
                neo4j_ok = False
        ollama = OllamaClient.from_config(cfg)
        ollama_ok = ollama.health()
        loaded = [model.get("name") for model in ollama.loaded_models()]
        status = "ok" if (not app.state.graph_enabled or neo4j_ok) and ollama_ok else "degraded"
        return {
            "status": status,
            "neo4j": neo4j_ok,
            "graph_enabled": app.state.graph_enabled,
            "ollama": ollama_ok,
            "loaded_models": loaded,
            "profile": cfg.profile.name,
        }

    @app.post("/chat")
    def chat(body: ChatIn) -> dict:
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="message must not be empty")
        logger.info("/chat message_chars=%d language=%s", len(body.message), body.language or "auto")
        session = (
            app.state.sessions.get(body.session_id)
            if app.state.memory_enabled
            else None
        )
        with _LLM_LOCK:
            result = app.state.pipeline.run(
                body.message, language=body.language, session=session
            )
        payload = result.as_dict()
        payload["answer_llm_locked"] = True
        payload["session_id"] = session.session_id if session is not None else None
        return payload

    @app.delete("/session/{session_id}")
    def delete_session(session_id: str) -> dict:
        deleted = app.state.sessions.delete(session_id)
        return {"session_id": session_id, "deleted": deleted}

    @app.post("/recommend")
    def recommend(body: RecommendIn) -> dict:
        if not body.query.strip():
            raise HTTPException(status_code=400, detail="query must not be empty")
        if body.domain not in {None, "movie", "game"}:
            raise HTTPException(status_code=400, detail="domain must be movie or game")
        logger.info("/recommend query_chars=%d domain=%s", len(body.query), body.domain)
        with _LLM_LOCK:
            result = app.state.pipeline.recommend(
                body.query, domain=body.domain, k=body.k
            )
        payload = result.as_dict()
        payload["items"] = payload["items"][: body.k]
        return payload

    return app


def run(cfg, args: argparse.Namespace | None = None) -> int:  # pragma: no cover
    """``python -m entdec api`` - serve the app with uvicorn."""
    import uvicorn

    from entdec.graph.store import GraphStoreError, ensure_graph
    from entdec.logging_setup import get_logger

    graph_enabled = not (
        bool(getattr(args, "no_graph", False)) or cfg.env.entdec_no_graph
    )
    try:
        ensure_graph(cfg, no_graph=not graph_enabled)
    except GraphStoreError as exc:
        get_logger(__name__).error("unable to select graph profile: %s", exc)
        return 1
    app = create_app(cfg, no_graph=not graph_enabled)
    uvicorn.run(
        app,
        host=getattr(args, "host", None) or "127.0.0.1",
        port=int(getattr(args, "port", None) or 8000),
    )
    return 0


__all__ = ["ChatIn", "RecommendIn", "create_app", "run"]
