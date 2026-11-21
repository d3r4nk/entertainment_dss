"""FastAPI auth and persistent conversation endpoint tests."""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import pyotp
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from entdec.api.main import create_app


class _MemoryPipeline:
    def __init__(self) -> None:
        self.prior_turns: list[int] = []

    def run(self, message, *, language=None, session=None):
        self.prior_turns.append(len(session.turns))
        session.add_turn(
            message,
            {"domain": "game", "genres": [], "themes": ["story rich"]},
            {},
            [{"id": 22, "domain": "game", "title": "Portal 2"}],
            "Grounded answer",
        )
        return SimpleNamespace(
            as_dict=lambda: {
                "question": message,
                "answer": "Grounded answer",
                "language": language or "en",
                "intent": {"domain": "game"},
                "used_graph": False,
                "intent_llm": True,
                "answer_llm": False,
                "items": [{"id": 22, "title": "Portal 2", "evidence": "story rich"}],
                "resolved_references": [],
                "durations": {"total": 0.1},
            }
        )


def _client(make_config):
    cfg = make_config("tiny")
    cfg.env.entdec_auth_disabled = False
    cfg.env.entdec_db_url = "sqlite://"
    cfg.env.entdec_secret_key = Fernet.generate_key().decode("ascii")
    app = create_app(cfg, no_graph=True)
    pipeline = _MemoryPipeline()
    app.state.pipeline = pipeline
    return TestClient(app), pipeline


def _register_and_login(client: TestClient, email: str) -> str:
    registered = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": "strong-password-24",
            "confirm_password": "strong-password-24",
        },
    )
    assert registered.status_code == 201
    details = registered.json()
    code = pyotp.TOTP(details["totp_secret"]).now()
    confirmed = client.post(
        "/auth/register/confirm-2fa",
        json={"enrollment_token": details["enrollment_token"], "code": code},
    )
    assert confirmed.status_code == 200
    assert len(confirmed.json()["recovery_codes"]) == 10
    preauth = client.post(
        "/auth/login",
        json={"email": email, "password": "strong-password-24"},
    )
    assert preauth.status_code == 200
    next_step = (int(time.time()) // 30 + 1) * 30
    login_code = pyotp.TOTP(details["totp_secret"]).at(next_step)
    authenticated = client.post(
        "/auth/login/2fa",
        json={"preauth_token": preauth.json()["preauth_token"], "code": login_code},
    )
    assert authenticated.status_code == 200
    return authenticated.json()["access_token"]


def test_auth_endpoints_and_user_scoped_conversation_memory(make_config) -> None:
    client, pipeline = _client(make_config)
    token_a = _register_and_login(client, "Alice@Example.com")
    token_b = _register_and_login(client, "bob@example.com")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    me = client.get("/auth/me", headers=headers_a)
    assert me.status_code == 200
    assert me.json()["email"] == "alice@example.com"
    assert client.get("/auth/me").status_code == 401

    first = client.post("/conversations", headers=headers_a, json={})
    second = client.post("/conversations", headers=headers_a, json={})
    assert first.status_code == second.status_code == 201
    conversation_id = first.json()["id"]
    assert client.get(f"/conversations/{conversation_id}", headers=headers_b).status_code == 404

    messages = f"/conversations/{conversation_id}/messages"
    assert (
        client.post(messages, headers=headers_a, json={"message": "recommend"}).status_code == 200
    )
    assert client.post(messages, headers=headers_a, json={"message": "more"}).status_code == 200
    other_id = second.json()["id"]
    assert (
        client.post(
            f"/conversations/{other_id}/messages",
            headers=headers_a,
            json={"message": "new context"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            f"/conversations/{other_id}/messages",
            headers=headers_a,
            json={"message": "a follow-up"},
        ).status_code
        == 200
    )
    assert pipeline.prior_turns == [0, 1, 0, 1]

    detail = client.get(f"/conversations/{conversation_id}", headers=headers_a).json()
    assert [message["role"] for message in detail["messages"]] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]
    assert detail["messages"][1]["meta"]["items"][0]["title"] == "Portal 2"
    assert detail["messages"][1]["meta"]["used_graph"] is False
    assert detail["messages"][1]["meta"]["intent_llm"] is True
    assert detail["messages"][1]["meta"]["fallback"] is True
    assert (
        client.patch(
            f"/conversations/{conversation_id}",
            headers=headers_a,
            json={"title": "renamed"},
        ).json()["title"]
        == "renamed"
    )
    assert client.delete(f"/conversations/{conversation_id}", headers=headers_b).status_code == 404
    assert client.delete(f"/conversations/{conversation_id}", headers=headers_a).status_code == 204
    assert client.post("/auth/logout", headers=headers_a).status_code == 204
    assert client.get("/auth/me", headers=headers_a).status_code == 401


def test_history_survives_api_recreation(make_config, tmp_path: Path) -> None:
    cfg = make_config("tiny")
    cfg.env.entdec_auth_disabled = False
    cfg.env.entdec_db_url = f"sqlite:///{tmp_path.as_posix()}/auth-smoke.sqlite"
    cfg.env.entdec_secret_key = Fernet.generate_key().decode("ascii")
    app = create_app(cfg, no_graph=True)
    app.state.pipeline = _MemoryPipeline()

    with TestClient(app) as client:
        token = _register_and_login(client, "restart@example.com")
        headers = {"Authorization": f"Bearer {token}"}
        conversation_ids = [
            client.post("/conversations", headers=headers, json={}).json()["id"] for _ in range(2)
        ]
        for conversation_id in conversation_ids:
            for message in ("first question", "follow-up"):
                response = client.post(
                    f"/conversations/{conversation_id}/messages",
                    headers=headers,
                    json={"message": message},
                )
                assert response.status_code == 200

    restarted_app = create_app(cfg, no_graph=True)
    restarted_app.state.pipeline = _MemoryPipeline()
    with TestClient(restarted_app) as restarted_client:
        headers = {"Authorization": f"Bearer {token}"}
        assert restarted_client.get("/auth/me", headers=headers).status_code == 200
        for conversation_id in conversation_ids:
            history = restarted_client.get(
                f"/conversations/{conversation_id}", headers=headers
            ).json()
            assert [message["role"] for message in history["messages"]] == [
                "user",
                "assistant",
                "user",
                "assistant",
            ]


def test_auth_errors_are_generic_for_unknown_and_wrong_password(make_config) -> None:
    client, _pipeline = _client(make_config)
    _register_and_login(client, "known@example.com")
    unknown = client.post(
        "/auth/login",
        json={"email": "unknown@example.com", "password": "wrong-password"},
    )
    wrong = client.post(
        "/auth/login",
        json={"email": "known@example.com", "password": "wrong-password"},
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json()["detail"] == wrong.json()["detail"] == "Invalid email or password."


