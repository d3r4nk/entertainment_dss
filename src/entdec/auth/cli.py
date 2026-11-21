"""Local auth database and account administration commands."""

from __future__ import annotations

import json
import secrets
import subprocess
import time
from typing import Any

from cryptography.fernet import Fernet

from entdec.config import PROJECT_ROOT, Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


def _engine(cfg: Config):
    from entdec.auth.database import make_engine

    url = cfg.env.entdec_db_url
    if not url:
        raise RuntimeError("ENTDEC_DB_URL is missing; configure the MySQL connection in .env.")
    return make_engine(url)


def _compose(args: list[str], *, timeout: float = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _create_local_env_if_missing() -> None:
    """Create secure first-run credentials without ever replacing an existing file."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        return
    content = "\n".join(
        [
            "# Generated local settings. This file is git-ignored; keep it private.",
            "NEO4J_URI=bolt://localhost:7688",
            "NEO4J_USER=neo4j",
            f"NEO4J_PASSWORD={secrets.token_hex(32)}",
            "NEO4J_DATABASE=neo4j",
            "OLLAMA_HOST=http://localhost:11434",
            "OLLAMA_EMBED_MODEL=qwen3-embedding:4b",
            "OLLAMA_GEN_MODEL=qwen3:4b",
            "ENTDEC_PROFILE=tiny",
            f"MYSQL_ROOT_PASSWORD={secrets.token_hex(32)}",
            "MYSQL_DATABASE=entdec",
            "MYSQL_USER=entdec",
            f"MYSQL_PASSWORD={secrets.token_hex(32)}",
            "ENTDEC_DB_URL=mysql+pymysql://${MYSQL_USER}:${MYSQL_PASSWORD}@127.0.0.1:3307/${MYSQL_DATABASE}?charset=utf8mb4",
            f"ENTDEC_SECRET_KEY={Fernet.generate_key().decode('ascii')}",
            "ENTDEC_AUTH_DISABLED=0",
            "",
        ]
    )
    try:
        with env_path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(content)
    except FileExistsError:
        return
    print("Created a git-ignored .env with generated local credentials.")


def _start_mysql() -> bool:
    _create_local_env_if_missing()
    result = _compose(["up", "-d", "entdec-mysql"])
    if result.returncode:
        logger.error(
            "unable to start entdec-mysql: %s",
            (result.stderr or "docker compose failed").strip()[:500],
        )
        return False

    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        container = _compose(["ps", "-q", "entdec-mysql"], timeout=15)
        if container.returncode == 0 and container.stdout.strip():
            health = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.State.Health.Status}}",
                    container.stdout.strip(),
                ],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if health.returncode == 0 and health.stdout.strip() == "healthy":
                print("entdec-mysql is healthy.")
                return True
        time.sleep(2)
    logger.error("entdec-mysql did not become healthy within 180 seconds")
    return False


def run(cfg: Config, args: Any = None) -> int:
    """Dispatch auth and local account administration operations."""
    action = getattr(args, "action", None)
    if action == "db-up":
        return 0 if _start_mysql() else 1
    if action == "init-db":
        from entdec.auth.database import init_db

        engine = _engine(cfg)
        try:
            init_db(engine)
        finally:
            engine.dispose()
        print("Authentication database schema is ready.")
        return 0

    from entdec.auth.database import make_session_factory
    from entdec.auth.service import AuthService

    if not cfg.env.entdec_db_url:
        raise RuntimeError("ENTDEC_DB_URL is missing; configure .env first.")
    service_engine = _engine(cfg)
    try:
        service = AuthService(
            make_session_factory(service_engine),
            cfg.env.entdec_secret_key or "",
            session_days=int(cfg.get("auth.session_lifetime_days", 7)),
        )
        if action == "list-users":
            print(json.dumps(service.list_users(), indent=2, ensure_ascii=False))
            return 0
        email = getattr(args, "email", None)
        if not email:
            logger.error("--email is required for auth %s", action)
            return 2
        if action == "delete-user":
            if not service.delete_user(email):
                logger.error("no account was found for the supplied email")
                return 1
            print("User deleted.")
            return 0
        if action == "reset-2fa":
            enrollment = service.reset_2fa(email)
            if enrollment is None:
                logger.error("no account was found for the supplied email")
                return 1
            print(
                "Scan this URI in the authenticator, then confirm via /auth/register/confirm-2fa:"
            )
            print(f"URI: {enrollment['totp_uri']}")
            print(f"Manual secret: {enrollment['totp_secret']}")
            print(f"Enrollment token: {enrollment['enrollment_token']}")
            return 0
        logger.error("unknown auth action: %s", action)
        return 2
    finally:
        service_engine.dispose()


__all__ = ["run"]
