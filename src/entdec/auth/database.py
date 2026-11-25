"""Database engine and idempotent schema initialization."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Engine, create_engine, event, select, text, update
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from entdec.auth.models import Base, SchemaVersion

SCHEMA_VERSION = 4


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
    with engine.begin() as connection:
        version = connection.scalar(select(SchemaVersion.version).limit(1))
        if version is None:
            connection.execute(
                SchemaVersion.__table__.insert().values(
                    version=SCHEMA_VERSION,
                    applied_at=datetime.now(UTC),
                )
            )
        elif version == 1:
            if engine.dialect.name == "mysql":
                connection.execute(
                    text("ALTER TABLE messages MODIFY COLUMN created_at DATETIME(6) NOT NULL")
                )
                connection.execute(
                    text(
                        "UPDATE messages AS assistant "
                        "JOIN messages AS user_message "
                        "ON user_message.conversation_id = assistant.conversation_id "
                        "AND user_message.role = 'user' "
                        "AND assistant.role = 'assistant' "
                        "AND user_message.created_at = assistant.created_at "
                        "SET assistant.created_at = "
                        "TIMESTAMPADD(MICROSECOND, 1, user_message.created_at)"
                    )
                )
            connection.execute(
                update(SchemaVersion)
                .where(SchemaVersion.version == version)
                .values(version=SCHEMA_VERSION, applied_at=datetime.now(UTC))
            )
        elif version == 2:
            connection.execute(update(SchemaVersion).where(SchemaVersion.version == 2)
                               .values(version=SCHEMA_VERSION, applied_at=datetime.now(UTC)))
        elif version == 3:
            connection.execute(update(SchemaVersion).where(SchemaVersion.version == 3)
                               .values(version=SCHEMA_VERSION, applied_at=datetime.now(UTC)))
        elif version != SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported database schema version {version}; "
                f"application requires {SCHEMA_VERSION}"
            )
    _import_legacy_traces(engine)


def _import_legacy_traces(engine):
    """Idempotently preserve old message traces; never invent missing timings."""
    import json

    from entdec.auth.models import Conversation, Message, QueryEvent, QueryRun

    with Session(engine) as db, db.begin():
        for message, conversation in db.execute(
            select(Message, Conversation).join(Conversation).where(Message.meta_json.is_not(None))
        ):
            try:
                meta = json.loads(message.meta_json)
                trace = meta.get("trace")
                if not isinstance(trace, dict) or not trace.get("trace_id"):
                    continue
                run_id = str(trace["trace_id"])
                if len(run_id) > 36 or db.get(QueryRun, run_id):
                    continue
                payload = dict(trace, schema_version=1, trace_id=run_id,
                               user_id=conversation.user_id, conversation_id=conversation.id,
                               status="complete" if trace.get("status") == "ok" else "error",
                               created_at=trace.get("started_at") or message.created_at.isoformat(),
                               answer=message.content, metrics={})
                db.add(QueryRun(id=run_id, user_id=conversation.user_id,
                                conversation_id=conversation.id, owner="legacy",
                                status=payload["status"], created_at=message.created_at,
                                payload=json.dumps(payload, ensure_ascii=False)))
                db.flush()
                for seq, event in enumerate(trace.get("events", []), 1):
                    db.add(QueryEvent(run_id=run_id, seq=seq,
                                      payload=json.dumps(dict(event, seq=seq), ensure_ascii=False)))
            except (ValueError, TypeError, AttributeError):
                continue
