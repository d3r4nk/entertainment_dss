"""MySQL integration checks for account and durable conversation storage."""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace

import pyotp
import pytest
from sqlalchemy import select, text
from sqlalchemy.exc import SQLAlchemyError

from entdec.auth.conversations import ConversationService
from entdec.auth.database import init_db, make_engine, make_session_factory
from entdec.auth.models import SchemaVersion
from entdec.auth.service import AuthService
from entdec.config import EnvSettings

pytestmark = pytest.mark.integration


def _mysql_engine():
    url = EnvSettings().entdec_db_url
    if not url:
        pytest.skip("Set ENTDEC_DB_URL and start MySQL with `entdec auth db-up`.")
    engine = make_engine(url)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        engine.dispose()
        pytest.skip("MySQL is not reachable; start it with `entdec auth db-up`.")
    return engine


def test_mysql_schema_is_idempotent_and_auth_history_flow() -> None:
    engine = _mysql_engine()
    email = f"integration-{uuid.uuid4().hex}@example.test"
    service = None
    try:
        init_db(engine)
        init_db(engine)
        sessions = make_session_factory(engine)
        secret_key = EnvSettings().entdec_secret_key or ""
        if not secret_key:
            pytest.skip("Set ENTDEC_SECRET_KEY in .env to run auth integration tests.")
        service = AuthService(sessions, secret_key)
        registration = service.register(email, "strong-integration-password")
        recovery_codes = service.confirm_registration(
            registration["enrollment_token"],
            pyotp.TOTP(registration["totp_secret"]).now(),
        )
        preauth = service.login(email, "strong-integration-password")
        next_step = (int(time.time()) // 30 + 1) * 30
        token = service.login_2fa(preauth, pyotp.TOTP(registration["totp_secret"]).at(next_step))
        assert service.authenticate(token)["email"] == email
        assert len(recovery_codes) == 10

        class FakePipeline:
            def run(self, message, *, language=None, session=None):
                session.add_turn(
                    message,
                    {"domain": "movie", "genres": [], "themes": []},
                    {},
                    [],
                    "mocked answer",
                )
                return SimpleNamespace(
                    as_dict=lambda: {
                        "answer": "mocked answer",
                        "answer_llm": False,
                        "items": [],
                        "resolved_references": [],
                        "intent": {"domain": "movie"},
                        "durations": {},
                    }
                )

        conversations = ConversationService(sessions)
        conversation = conversations.create(service.authenticate(token)["id"])
        conversations.run_turn(
            service.authenticate(token)["id"],
            conversation["id"],
            "recommend a movie",
            FakePipeline(),
        )
        listing = conversations.list(service.authenticate(token)["id"])
        assert listing["total"] == 1
        assert (
            conversations.get(service.authenticate(token)["id"], conversation["id"])["messages"][1][
                "content"
            ]
            == "mocked answer"
        )
        with sessions() as db:
            assert db.scalar(select(SchemaVersion.version)) == 2
        if engine.dialect.name == "mysql":
            with engine.connect() as connection:
                precision = connection.execute(
                    text(
                        "SELECT datetime_precision FROM information_schema.columns "
                        "WHERE table_schema=DATABASE() AND table_name='messages' "
                        "AND column_name='created_at'"
                    )
                ).scalar_one()
            assert precision == 6
    finally:
        if service is not None:
            service.delete_user(email)
        engine.dispose()
