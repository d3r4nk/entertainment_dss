"""Database engine and idempotent schema initialization."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from entdec.auth.models import Base, SchemaVersion

SCHEMA_VERSION = 1


def make_engine(url: str, *, echo: bool = False) -> Engine:
    """Create a synchronous engine, sharing in-memory SQLite across test threads."""
    options: dict = {"echo": echo, "pool_pre_ping": True}
    if url in {"sqlite://", "sqlite:///:memory:"}:
        options.update(
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    engine = create_engine(url, **options)
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def _enable_sqlite_foreign_keys(connection, _record) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Build the standard SQLAlchemy session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_db(engine: Engine) -> None:
    """Create missing tables and record the current schema version."""
    Base.metadata.create_all(engine)
    with Session(engine) as session, session.begin():
        version = session.scalar(select(SchemaVersion.version).limit(1))
        if version is None:
            session.add(SchemaVersion(version=SCHEMA_VERSION, applied_at=datetime.now(UTC)))
        elif version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported database schema version {version}; "
                f"application requires {SCHEMA_VERSION}"
            )
