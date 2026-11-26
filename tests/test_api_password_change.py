"""Password changes are authenticated and revoke every existing session."""

from __future__ import annotations

import time
from pathlib import Path

import pyotp
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from entdec.api.main import create_app


def _register_login(client: TestClient, email: str) -> str:
    registered = client.post(
        "/auth/register",
        json={
            "email": email,
            "password": "strong-password-24",
            "confirm_password": "strong-password-24",
        },
    ).json()
    confirmed = client.post(
        "/auth/register/confirm-2fa",
        json={
            "enrollment_token": registered["enrollment_token"],
            "code": pyotp.TOTP(registered["totp_secret"]).now(),
        },
    )
    assert confirmed.status_code == 200
    preauth = client.post(
        "/auth/login", json={"email": email, "password": "strong-password-24"}
    ).json()["preauth_token"]
    next_step = (int(time.time()) // 30 + 1) * 30
    return client.post(
        "/auth/login/2fa",
        json={"preauth_token": preauth, "code": pyotp.TOTP(registered["totp_secret"]).at(next_step)},
    ).json()["access_token"]


def test_password_change_validates_and_revokes_sessions(make_config, tmp_path: Path) -> None:
    cfg = make_config("tiny")
    cfg.env.entdec_auth_disabled = False
    cfg.env.entdec_db_url = f"sqlite:///{tmp_path.as_posix()}/password.sqlite"
    cfg.env.entdec_secret_key = Fernet.generate_key().decode("ascii")
    app = create_app(cfg, no_graph=True)

    with TestClient(app) as client:
        token = _register_login(client, "password-change@example.com")
        headers = {"Authorization": f"Bearer {token}"}
        changed = client.post(
            "/auth/password",
            headers=headers,
            json={
                "current_password": "strong-password-24",
                "new_password": "new-strong-password-25",
                "confirm_password": "new-strong-password-25",
            },
        )
        assert changed.status_code == 204
        assert client.get("/auth/me", headers=headers).status_code == 401

        login = client.post(
            "/auth/login",
            json={"email": "password-change@example.com", "password": "new-strong-password-25"},
        )
        assert login.status_code == 200


def test_password_change_rejects_wrong_current_and_confirmation(make_config, tmp_path: Path) -> None:
    cfg = make_config("tiny")
    cfg.env.entdec_auth_disabled = False
    cfg.env.entdec_db_url = f"sqlite:///{tmp_path.as_posix()}/password-errors.sqlite"
    cfg.env.entdec_secret_key = Fernet.generate_key().decode("ascii")
    app = create_app(cfg, no_graph=True)

    with TestClient(app) as client:
        token = _register_login(client, "password-errors@example.com")
        headers = {"Authorization": f"Bearer {token}"}
        wrong_current = client.post(
            "/auth/password",
            headers=headers,
            json={
                "current_password": "wrong-password-24",
                "new_password": "new-strong-password-25",
                "confirm_password": "new-strong-password-25",
            },
        )
        assert wrong_current.status_code == 401

        mismatch = client.post(
            "/auth/password",
            headers=headers,
            json={
                "current_password": "strong-password-24",
                "new_password": "new-strong-password-25",
                "confirm_password": "another-strong-password-25",
            },
        )
        assert mismatch.status_code == 422

        invalid_new = client.post(
            "/auth/password",
            headers=headers,
            json={
                "current_password": "strong-password-24",
                "new_password": "short",
                "confirm_password": "short",
            },
        )
        assert invalid_new.status_code == 422

        assert client.get("/auth/me", headers=headers).status_code == 200
