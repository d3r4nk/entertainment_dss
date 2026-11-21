"""Tests for safe local auth environment initialization."""

from __future__ import annotations

from cryptography.fernet import Fernet
from sqlalchemy import select

from entdec.auth import cli as auth_cli
from entdec.auth.database import init_db, make_engine, make_session_factory
from entdec.auth.models import SchemaVersion


def test_missing_env_is_created_with_strong_credentials(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(auth_cli, "PROJECT_ROOT", tmp_path)
    auth_cli._create_local_env_if_missing()
    content = (tmp_path / ".env").read_text(encoding="utf-8")
    values = dict(line.split("=", 1) for line in content.splitlines() if "=" in line)

    assert len(values["MYSQL_ROOT_PASSWORD"]) >= 32
    assert len(values["MYSQL_PASSWORD"]) >= 32
    Fernet(values["ENTDEC_SECRET_KEY"].encode("ascii"))
    assert values["ENTDEC_DB_URL"].endswith("127.0.0.1:3307/${MYSQL_DATABASE}?charset=utf8mb4")


def test_existing_env_is_never_replaced(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(auth_cli, "PROJECT_ROOT", tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP_THIS=value\n", encoding="utf-8")

    auth_cli._create_local_env_if_missing()

    assert env_file.read_text(encoding="utf-8") == "KEEP_THIS=value\n"


def test_schema_initialization_is_idempotent() -> None:
    engine = make_engine("sqlite://")
    try:
        init_db(engine)
        init_db(engine)
        with make_session_factory(engine)() as session:
            assert session.scalar(select(SchemaVersion.version)) == 2
    finally:
        engine.dispose()
