"""Transient authentication database outage response coverage."""

from __future__ import annotations

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

from entdec.api.main import create_app


def test_database_operational_error_returns_clean_503(make_config) -> None:
    cfg = make_config("tiny")
    cfg.env.entdec_auth_disabled = False
    cfg.env.entdec_db_url = "sqlite://"
    cfg.env.entdec_secret_key = Fernet.generate_key().decode("ascii")
    app = create_app(cfg, no_graph=True)

    def database_unavailable(*_args, **_kwargs):
        raise OperationalError("SELECT 1", {}, OSError("database unavailable"))

    app.state.auth_service.authenticate = database_unavailable
    with TestClient(app) as client:
        response = client.get(
            "/auth/me",
            headers={"Authorization": "Bearer outage-probe"},
        )
    assert response.status_code == 503
    assert response.json() == {
        "detail": "Authentication database is temporarily unavailable."
    }
    assert "traceback" not in response.text.lower()
