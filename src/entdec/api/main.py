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
import json
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any

from fastapi import Query, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from entdec.auth.conversations import ConversationNotFound
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
    conversation_id: str | None = Field(default=None)


class RecommendIn(BaseModel):
    """Payload for ``/recommend``."""

    query: str
    domain: str | None = Field(default=None)  # optional: movie | game
    k: int = Field(default=10, ge=1, le=20)


class RegisterIn(BaseModel):
    email: str
    password: str
    confirm_password: str


class RegisterOut(BaseModel):
    enrollment_token: str
    totp_uri: str
    totp_secret: str
    qr_code_png_base64: str


class Confirm2FAIn(BaseModel):
    enrollment_token: str
    code: str


class RecoveryCodesOut(BaseModel):
    recovery_codes: list[str]


class LoginIn(BaseModel):
    email: str
    password: str


class Login2FAIn(BaseModel):
    preauth_token: str
    code: str


class ChangePasswordIn(BaseModel):
    current_password: str
    new_password: str
    confirm_password: str


class PreAuthOut(BaseModel):
    preauth_token: str
    requires_2fa: bool = True


class SessionOut(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserOut(BaseModel):
    id: str
    email: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    meta: dict[str, Any] | None = None
    created_at: str


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str


class ConversationDetailOut(ConversationOut):
    messages: list[MessageOut]


class ConversationListOut(BaseModel):
    items: list[ConversationOut]
    total: int
    offset: int
    limit: int


class ConversationCreateIn(BaseModel):
    title: str = "New chat"


class ConversationRenameIn(BaseModel):
    title: str


class ChatTurnIn(BaseModel):
    message: str = Field(min_length=1)
    language: str | None = None


class ChatTurnOut(BaseModel):
    conversation_id: str
    title: str
    user_message: str
    result: dict[str, Any]


class AccountStateIn(BaseModel):
    payload: dict[str, Any] = Field(default_factory=dict)


def create_app(cfg=None, *, no_graph: bool = False):
    """Build the FastAPI application (state cached on ``app.state``)."""
    try:
        from fastapi import Depends, FastAPI, HTTPException
        from fastapi.middleware.cors import CORSMiddleware
        from sqlalchemy.exc import OperationalError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("fastapi and SQLAlchemy are required to serve the API") from exc

    from entdec.auth.conversations import ConversationService
    from entdec.auth.database import init_db, make_engine, make_session_factory
    from entdec.auth.models import AccountState
    from entdec.auth.service import AuthError, AuthService
    from entdec.config import get_config
    from entdec.graph.neo4j_client import Neo4jClient
    from entdec.logging_setup import get_logger
    from entdec.nlp.ollama_client import OllamaClient
    from entdec.ui.recommendation_media import RecommendationMedia

    logger = get_logger(__name__)
    cfg = cfg or get_config()
    graph_enabled = not (no_graph or cfg.env.entdec_no_graph)

    app = FastAPI(title="Entertainment_decision recommender", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:8501",
            "http://127.0.0.1:8501",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.state.graph_enabled = graph_enabled
    app.state.pipeline = None
    app.state.recommendation_media = RecommendationMedia(
        cfg.raw_path("movie", "links.csv"),
        cfg.env.tmdb_read_access_token,
    )

    def pipeline_for_request():
        if app.state.pipeline is None:
            try:
                from entdec.chatbot.pipeline import ChatPipeline

                app.state.pipeline = ChatPipeline(cfg, use_graph=graph_enabled)
            except Exception as exc:  # AI assets/services are optional for account APIs.
                logger.warning("AI recommender is unavailable: %s", exc)
                raise HTTPException(
                    status_code=503,
                    detail="Môi trường AI chưa sẵn sàng. Hội thoại và dữ liệu tài khoản vẫn sử dụng được.",
                ) from exc
        return app.state.pipeline
    app.state.sessions = SessionStore(
        ttl_minutes=int(cfg.get("llm.session_ttl_minutes", 60)),
        max_sessions=int(cfg.get("llm.max_sessions", 100)),
        memory_turns=int(cfg.get("llm.memory_turns", 4)),
    )
    app.state.memory_enabled = cfg.memory_enabled
    app.state.auth_enabled = not cfg.env.entdec_auth_disabled
    if app.state.auth_enabled:
        if not cfg.env.entdec_db_url:
            raise RuntimeError(
                "ENTDEC_DB_URL is required when API authentication is enabled; "
                "configure .env and run `python -m entdec auth init-db`."
            )
        app.state.auth_engine = make_engine(cfg.env.entdec_db_url)
        if app.state.auth_engine.dialect.name == "sqlite":
            init_db(app.state.auth_engine)
        else:
            from sqlalchemy import text
            from sqlalchemy.exc import SQLAlchemyError

            try:
                with app.state.auth_engine.connect() as connection:
                    connection.execute(text("SELECT 1"))
            except SQLAlchemyError as exc:
                raise RuntimeError(
                    "Cannot connect to ENTDEC_DB_URL; start MySQL and run "
                    "`python -m entdec auth init-db`."
                ) from exc
        app.state.auth_sessions = make_session_factory(app.state.auth_engine)
        app.state.auth_service = AuthService(
            app.state.auth_sessions,
            cfg.env.entdec_secret_key or "",
            session_days=int(cfg.get("auth.session_lifetime_days", 7)),
        )
    else:
        app.state.auth_engine = make_engine("sqlite://")
        init_db(app.state.auth_engine)
        app.state.auth_sessions = make_session_factory(app.state.auth_engine)
        app.state.auth_service = None
    app.state.conversations = ConversationService(
        app.state.auth_sessions,
        memory_turns=int(cfg.get("llm.memory_turns", 4)),
    )

    @app.exception_handler(AuthError)
    async def auth_error_handler(_request: Request, exc: AuthError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(ConversationNotFound)
    async def conversation_not_found_handler(
        _request: Request, _exc: ConversationNotFound
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Conversation not found."})

    @app.exception_handler(OperationalError)
    async def database_unavailable_handler(
        _request: Request, _exc: OperationalError
    ) -> JSONResponse:
        logger.warning("authentication database is temporarily unavailable")
        return JSONResponse(
            status_code=503,
            content={"detail": "Authentication database is temporarily unavailable."},
            headers={"Retry-After": "5"},
        )

    def request_ip(request: Request) -> str | None:
        return request.client.host if request.client else None

    def bearer_token(request: Request) -> str:
        header = request.headers.get("authorization", "")
        scheme, separator, token = header.partition(" ")
        if not separator or scheme.casefold() != "bearer" or not token.strip():
            from entdec.auth.service import AuthError

            raise AuthError(401, "Authentication required.")
        return token.strip()

    def current_user(request: Request) -> dict[str, str]:
        if not app.state.auth_enabled:
            return {"id": "dev-anonymous", "email": "local development"}
        return app.state.auth_service.authenticate(bearer_token(request), ip=request_ip(request))

    def require_auth_enabled():
        if not app.state.auth_enabled:
            raise HTTPException(status_code=404, detail="Authentication is disabled.")
        return app.state.auth_service

    @app.post("/auth/register", response_model=RegisterOut, status_code=201)
    def auth_register(body: RegisterIn, request: Request) -> dict:
        service = require_auth_enabled()
        if body.password != body.confirm_password:
            raise HTTPException(status_code=422, detail="Passwords do not match.")
        return service.register(body.email, body.password, ip=request_ip(request))

    @app.post(
        "/auth/register/confirm-2fa",
        response_model=RecoveryCodesOut,
    )
    def auth_confirm_2fa(body: Confirm2FAIn, request: Request) -> dict:
        service = require_auth_enabled()
        return {
            "recovery_codes": service.confirm_registration(
                body.enrollment_token, body.code, ip=request_ip(request)
            )
        }

    @app.post("/auth/login", response_model=PreAuthOut)
    def auth_login(body: LoginIn, request: Request) -> dict:
        service = require_auth_enabled()
        return {"preauth_token": service.login(body.email, body.password, ip=request_ip(request))}

    @app.post("/auth/login/2fa", response_model=SessionOut)
    def auth_login_2fa(body: Login2FAIn, request: Request) -> dict:
        service = require_auth_enabled()
        token = service.login_2fa(
            body.preauth_token,
            body.code,
            user_agent=request.headers.get("user-agent"),
            ip=request_ip(request),
        )
        return {
            "access_token": token,
            "token_type": "bearer",
            "expires_in": int(cfg.get("auth.session_lifetime_days", 7)) * 86400,
        }

    @app.post("/auth/logout", status_code=204)
    def auth_logout(
        request: Request,
        _user: dict = Depends(current_user),  # noqa: B008
    ) -> Response:
        if app.state.auth_enabled:
            app.state.auth_service.logout(bearer_token(request), ip=request_ip(request))
        return Response(status_code=204)

    @app.get("/auth/me", response_model=UserOut)
    def auth_me(user: dict = Depends(current_user)) -> dict:  # noqa: B008
        return user

    @app.post("/auth/password", status_code=204)
    def auth_change_password(
        body: ChangePasswordIn,
        request: Request,
        user: dict = Depends(current_user),  # noqa: B008
    ) -> Response:
        if body.new_password != body.confirm_password:
            raise HTTPException(status_code=422, detail="Passwords do not match.")
        if not app.state.auth_enabled:
            raise HTTPException(status_code=404, detail="Authentication is disabled.")
        app.state.auth_service.change_password(
            user["id"],
            body.current_password,
            body.new_password,
            ip=request_ip(request),
        )
        return Response(status_code=204)

    @app.get("/account/state")
    def get_account_state(user: dict = Depends(current_user)) -> dict:
        require_auth_enabled()
        with app.state.auth_sessions() as db:
            row = db.get(AccountState, user["id"])
            if row is None:
                return {"payload": {}, "updated_at": None}
            try:
                payload = json.loads(row.payload_json)
            except (TypeError, ValueError):
                payload = {}
            return {"payload": payload, "updated_at": row.updated_at.isoformat()}

    @app.put("/account/state")
    def put_account_state(
        body: AccountStateIn, user: dict = Depends(current_user)
    ) -> dict:
        require_auth_enabled()
        from datetime import UTC, datetime

        with app.state.auth_sessions() as db, db.begin():
            row = db.get(AccountState, user["id"])
            if row is None:
                row = AccountState(user_id=user["id"], payload_json="{}")
                db.add(row)
            row.payload_json = json.dumps(body.payload, ensure_ascii=False)
            row.updated_at = datetime.now(UTC)
            db.flush()
            return {"payload": body.payload, "updated_at": row.updated_at.isoformat()}

    @app.get("/conversations", response_model=ConversationListOut)
    def list_conversations(
        offset: int = 0,
        limit: int = Query(default=20, ge=1, le=100),
        search: str | None = None,
        user: dict = Depends(current_user),  # noqa: B008
    ) -> dict:
        return app.state.conversations.list(
            user["id"], offset=max(0, offset), limit=limit, search=search
        )

    @app.post("/conversations", response_model=ConversationOut, status_code=201)
    def create_conversation(
        body: ConversationCreateIn,
        user: dict = Depends(current_user),  # noqa: B008
    ) -> dict:
        return app.state.conversations.create(user["id"], body.title)

    @app.delete("/conversations", status_code=200)
    def delete_all_conversations(user: dict = Depends(current_user)) -> dict:  # noqa: B008
        deleted = app.state.conversations.delete_all(user["id"])
        return {"deleted": deleted}

    @app.get("/conversations/{conversation_id}", response_model=ConversationDetailOut)
    def get_conversation(
        conversation_id: str,
        user: dict = Depends(current_user),  # noqa: B008
    ) -> dict:
        return app.state.conversations.get(user["id"], conversation_id)

    @app.patch("/conversations/{conversation_id}", response_model=ConversationOut)
    def rename_conversation(
        conversation_id: str,
        body: ConversationRenameIn,
        user: dict = Depends(current_user),  # noqa: B008
    ) -> dict:
        return app.state.conversations.rename(user["id"], conversation_id, body.title)

    @app.delete("/conversations/{conversation_id}", status_code=204)
    def delete_conversation(
        conversation_id: str,
        user: dict = Depends(current_user),  # noqa: B008
    ) -> Response:
        app.state.conversations.delete(user["id"], conversation_id)
        return Response(status_code=204)

    @app.post("/conversations/{conversation_id}/messages", response_model=ChatTurnOut)
    def conversation_message(
        conversation_id: str,
        body: ChatTurnIn,
        user: dict = Depends(current_user),  # noqa: B008
    ) -> dict:
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="message must not be empty")
        logger.info("/conversations message_chars=%d", len(body.message))
        with _LLM_LOCK:
            return app.state.conversations.run_turn(
                user["id"],
                conversation_id,
                body.message,
                pipeline_for_request(),
                language=body.language,
            )

    from entdec.auth.query_runs import QueryRuns, statistics

    def trace_filters(status=None, profile=None, conversation_id=None, domain=None,
                      answer_source=None, since=None, until=None):
        return dict(status=status, profile=profile, conversation_id=conversation_id,
                    domain=domain, answer_source=answer_source, since=since, until=until)

    @app.get("/traces")
    def traces(offset: int = 0, limit: int = 50,
               filters: dict = Depends(trace_filters), user: dict = Depends(current_user)):
        items = QueryRuns(app.state.conversations).list(user["id"], **filters)
        offset, limit = max(0, offset), max(1, min(limit, 200))
        return {"items": items[offset:offset + limit], "total": len(items),
                "offset": offset, "limit": limit}

    @app.get("/traces/stats")
    def trace_stats(filters: dict = Depends(trace_filters), user: dict = Depends(current_user)):
        return statistics(QueryRuns(app.state.conversations).list(user["id"], **filters))

    @app.get("/traces/{trace_id}")
    def trace_detail(trace_id: str, user: dict = Depends(current_user)):
        return QueryRuns(app.state.conversations).get(user["id"], trace_id)

    @app.get("/traces/{trace_id}/events")
    def trace_events(trace_id: str, after_seq: int = 0, user: dict = Depends(current_user)):
        return QueryRuns(app.state.conversations).get(user["id"], trace_id, max(0, after_seq))["events"]

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

    @app.get("/media/{domain}/{item_id}")
    def recommendation_poster(domain: str, item_id: str) -> Response:
        if domain not in ("movie", "game"):
            raise HTTPException(status_code=404, detail="Poster not found.")
        image = app.state.recommendation_media.poster_for({"domain": domain, "id": item_id})
        if image is None:
            raise HTTPException(status_code=404, detail="Poster not found.")
        try:
            content = image.result(timeout=8)
        except FutureTimeoutError as exc:
            raise HTTPException(status_code=504, detail="Poster source timed out.") from exc
        if content is None:
            raise HTTPException(status_code=404, detail="Poster not found.")
        return Response(
            content=content,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.post("/chat")
    def chat(body: ChatIn, request: Request) -> dict:
        if not body.message.strip():
            raise HTTPException(status_code=400, detail="message must not be empty")
        logger.info("/chat message_chars=%d language=%s", len(body.message), body.language or "auto")
        if app.state.auth_enabled:
            user = current_user(request)
            conversation_id = body.conversation_id or body.session_id
            if conversation_id is None:
                conversation_id = app.state.conversations.create(user["id"])["id"]
            with _LLM_LOCK:
                turn = app.state.conversations.run_turn(
                    user["id"],
                    conversation_id,
                    body.message,
                    pipeline_for_request(),
                    language=body.language,
                )
            payload = dict(turn["result"])
            payload["conversation_id"] = conversation_id
            payload["session_id"] = conversation_id
            payload["answer_llm_locked"] = True
            return payload
        session = (
            app.state.sessions.get(body.session_id)
            if app.state.memory_enabled
            else None
        )
        with _LLM_LOCK:
            result = pipeline_for_request().run(
                body.message, language=body.language, session=session
            )
        payload = result.as_dict()
        payload["answer_llm_locked"] = True
        payload["session_id"] = session.session_id if session is not None else None
        return payload

    @app.delete("/session/{session_id}")
    def delete_session(session_id: str, request: Request) -> dict:
        if app.state.auth_enabled:
            user = current_user(request)
            app.state.conversations.delete(user["id"], session_id)
            return {"session_id": session_id, "deleted": True}
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
            result = pipeline_for_request().recommend(
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


__all__ = ["ChangePasswordIn", "ChatIn", "RecommendIn", "create_app", "run"]
