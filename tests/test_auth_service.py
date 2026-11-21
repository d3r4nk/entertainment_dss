"""SQLite-backed authentication and conversation-service tests."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pyotp
import pytest
from cryptography.fernet import Fernet
from sqlalchemy import select

from entdec.auth.conversations import ConversationNotFound, ConversationService
from entdec.auth.database import init_db, make_engine, make_session_factory
from entdec.auth.models import (
    AuthSession,
    RecoveryCode,
    User,
)
from entdec.auth.service import AuthError, AuthService


@pytest.fixture
def auth_context():
    engine = make_engine("sqlite://")
    init_db(engine)
    sessions = make_session_factory(engine)
    clock = [time.time()]
    service = AuthService(
        sessions,
        Fernet.generate_key().decode("ascii"),
        session_days=7,
        clock=lambda: clock[0],
    )
    yield service, sessions, clock
    engine.dispose()


def _enroll(service: AuthService, email: str = "alice@example.com"):
    registration = service.register(email, "strong-password-24")
    now = time.time()
    recovery_codes = service.confirm_registration(
        registration["enrollment_token"],
        pyotp.TOTP(registration["totp_secret"]).at(now),
    )
    return registration, recovery_codes


def _login(service: AuthService, email: str, secret: str, at: float) -> str:
    preauth = service.login(email, "strong-password-24")
    return service.login_2fa(preauth, pyotp.TOTP(secret).at(at))


def test_registration_hashes_password_and_encrypts_totp_seed(auth_context) -> None:
    service, sessions, clock = auth_context
    registration = service.register("Alice@Example.com", "strong-password-24")
    recovery = service.confirm_registration(
        registration["enrollment_token"],
        pyotp.TOTP(registration["totp_secret"]).at(clock[0]),
    )

    with sessions() as db:
        user = db.scalar(select(User).where(User.email == "alice@example.com"))
        stored_codes = db.scalars(select(RecoveryCode).where(RecoveryCode.user_id == user.id)).all()
    assert user is not None and user.is_active and user.totp_enabled
    assert user.password_hash != "strong-password-24"
    assert registration["totp_secret"] not in user.totp_secret_encrypted
    assert len(recovery) == len(stored_codes) == 10
    assert all(code.code_hash not in recovery for code in stored_codes)


@pytest.mark.parametrize(
    "password",
    ["short", "password123", "alice@example.com", "alice"],
)
def test_registration_rejects_weak_passwords(auth_context, password: str) -> None:
    service, _sessions, _clock = auth_context
    with pytest.raises(AuthError) as error:
        service.register("alice@example.com", password)
    assert error.value.status_code == 422


def test_password_then_totp_login_replay_and_session_revocation(auth_context) -> None:
    service, sessions, clock = auth_context
    registration, _codes = _enroll(service)
    clock[0] += 30
    preauth = service.login("ALICE@example.com", "strong-password-24")
    token = service.login_2fa(preauth, pyotp.TOTP(registration["totp_secret"]).at(clock[0]))
    assert service.authenticate(token)["email"] == "alice@example.com"

    replay_preauth = service.login("alice@example.com", "strong-password-24")
    with pytest.raises(AuthError, match="Invalid authentication code"):
        service.login_2fa(replay_preauth, pyotp.TOTP(registration["totp_secret"]).at(clock[0]))
    service.logout(token)
    with pytest.raises(AuthError, match="Authentication required"):
        service.authenticate(token)
    with sessions() as db:
        stored = db.scalar(select(AuthSession).where(AuthSession.user_id.is_not(None)))
        assert stored is not None
        assert stored.token_hash != token


def test_wrong_password_and_second_factor_lock_account(auth_context) -> None:
    service, sessions, _clock = auth_context
    _enroll(service)
    for attempt in range(5):
        with pytest.raises(AuthError) as error:
            service.login("alice@example.com", f"wrong-password-{attempt}")
    assert error.value.status_code == 423
    with sessions() as db:
        user = db.scalar(select(User).where(User.email == "alice@example.com"))
        assert user.failed_login_count == 5
        assert user.locked_until is not None


def test_wrong_totp_also_locks_account(auth_context) -> None:
    service, sessions, _clock = auth_context
    _enroll(service)
    for attempt in range(5):
        preauth = service.login("alice@example.com", "strong-password-24")
        with pytest.raises(AuthError) as error:
            service.login_2fa(preauth, f"{attempt:06d}")
    assert error.value.status_code == 423
    with sessions() as db:
        user = db.scalar(select(User).where(User.email == "alice@example.com"))
        assert user.failed_login_count == 5


def test_recovery_code_is_single_use(auth_context) -> None:
    service, _sessions, clock = auth_context
    registration, codes = _enroll(service)
    preauth = service.login("alice@example.com", "strong-password-24")
    token = service.login_2fa(preauth, codes[0])
    assert service.authenticate(token)["id"]

    retry = service.login("alice@example.com", "strong-password-24")
    with pytest.raises(AuthError, match="Invalid authentication code"):
        service.login_2fa(retry, codes[0])
    assert registration["totp_secret"]
    clock[0] += 30


def test_expired_session_is_rejected(auth_context) -> None:
    service, _sessions, clock = auth_context
    registration, _codes = _enroll(service)
    clock[0] += 30
    token = _login(service, "alice@example.com", registration["totp_secret"], clock[0])
    clock[0] += 8 * 24 * 60 * 60
    with pytest.raises(AuthError, match="Authentication required"):
        service.authenticate(token)


def test_conversation_state_round_trip_and_user_isolation(auth_context) -> None:
    _auth, sessions, _clock = auth_context
    with sessions() as db, db.begin():
        db.add_all(
            [
                User(
                    id="user-a",
                    email="a@example.com",
                    password_hash="hash",
                    totp_secret_encrypted=None,
                    totp_enabled=True,
                    is_active=True,
                ),
                User(
                    id="user-b",
                    email="b@example.com",
                    password_hash="hash",
                    totp_secret_encrypted=None,
                    totp_enabled=True,
                    is_active=True,
                ),
            ]
        )

    conversations = ConversationService(sessions, memory_turns=3)
    first = conversations.create("user-a")
    second = conversations.create("user-a")

    class FakePipeline:
        def __init__(self) -> None:
            self.prior_turns: list[int] = []

        def run(self, message, *, language=None, session=None):
            self.prior_turns.append(len(session.turns))
            session.add_turn(
                message,
                {"domain": "game", "genres": [], "themes": ["story rich"]},
                {"Portal 2": ("game", 22)},
                [{"id": 22, "domain": "game", "title": "Portal 2"}],
                "Grounded answer",
            )
            return SimpleNamespace(
                as_dict=lambda: {
                    "answer": "Grounded answer",
                    "answer_llm": False,
                    "items": [{"title": "Portal 2", "evidence": "story rich"}],
                    "resolved_references": ["Portal 2"],
                    "intent": {"domain": "game"},
                    "durations": {"total": 0.1},
                }
            )

    pipeline = FakePipeline()
    conversations.run_turn("user-a", first["id"], "recommend a game", pipeline)
    conversations.run_turn("user-a", first["id"], "more like that", pipeline)
    conversations.run_turn("user-a", second["id"], "another chat", pipeline)
    assert pipeline.prior_turns == [0, 1, 0]
    saved = conversations.get("user-a", first["id"])
    assert [message["role"] for message in saved["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert saved["messages"][1]["meta"]["items"][0]["title"] == "Portal 2"
    assert (
        json.loads(_conversation_state(sessions, first["id"]))["turns"][-1]["user_message"]
        == "more like that"
    )
    with pytest.raises(ConversationNotFound):
        conversations.get("user-b", first["id"])
    with pytest.raises(ConversationNotFound):
        conversations.rename("user-b", first["id"], "steal")
    with pytest.raises(ConversationNotFound):
        conversations.delete("user-b", first["id"])
    assert conversations.rename("user-a", first["id"], "renamed")["title"] == "renamed"
    conversations.delete("user-a", first["id"])
    assert conversations.list("user-a")["total"] == 1
    assert conversations.delete_all("user-a") == 1


def _conversation_state(sessions, conversation_id: str) -> str:
    from entdec.auth.models import Conversation

    with sessions() as db:
        return db.get(Conversation, conversation_id).state_json
