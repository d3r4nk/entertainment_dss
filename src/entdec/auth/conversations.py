"""Per-user conversation CRUD and durable chatbot turn handling."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from entdec.auth.models import Conversation, Message
from entdec.chatbot.session import ChatSession


class ConversationNotFound(LookupError):
    """The conversation does not exist or is not owned by the requesting user."""


class ConversationService:
    """Conversation operations scoped to an authenticated user ID."""

    def __init__(self, sessions: sessionmaker[Session], *, memory_turns: int = 4) -> None:
        self.sessions = sessions
        self.memory_turns = max(1, int(memory_turns))

    def create(self, user_id: str, title: str = "New chat") -> dict[str, Any]:
        clean_title = _title(title)
        now = datetime.now(UTC)
        conversation = Conversation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            title=clean_title,
            state_json="{}",
            created_at=now,
            updated_at=now,
        )
        with self.sessions() as db, db.begin():
            db.add(conversation)
        return _conversation_dict(conversation)

    def list(
        self,
        user_id: str,
        *,
        offset: int = 0,
        limit: int = 20,
        search: str | None = None,
    ) -> dict[str, Any]:
        query = select(Conversation).where(
            Conversation.user_id == user_id,
            Conversation.deleted_at.is_(None),
        )
        if search and search.strip():
            query = query.where(Conversation.title.ilike(f"%{_like_escape(search.strip())}%"))
        with self.sessions() as db:
            total = db.scalar(select(func.count()).select_from(query.subquery())) or 0
            rows = db.scalars(
                query.order_by(Conversation.updated_at.desc())
                .offset(max(0, offset))
                .limit(max(1, min(100, limit)))
            ).all()
            return {
                "items": [_conversation_dict(row) for row in rows],
                "total": int(total),
                "offset": max(0, offset),
                "limit": max(1, min(100, limit)),
            }

    def get(self, user_id: str, conversation_id: str) -> dict[str, Any]:
        with self.sessions() as db:
            conversation = self._owned(db, user_id, conversation_id)
            return _conversation_dict(conversation, include_messages=True)

    def rename(self, user_id: str, conversation_id: str, title: str) -> dict[str, Any]:
        clean_title = _title(title)
        with self.sessions() as db, db.begin():
            conversation = self._owned(db, user_id, conversation_id)
            conversation.title = clean_title
            conversation.updated_at = datetime.now(UTC)
        return _conversation_dict(conversation)

    def delete(self, user_id: str, conversation_id: str) -> None:
        with self.sessions() as db, db.begin():
            conversation = self._owned(db, user_id, conversation_id)
            conversation.deleted_at = datetime.now(UTC)
            conversation.updated_at = conversation.deleted_at

    def delete_all(self, user_id: str) -> int:
        with self.sessions() as db, db.begin():
            rows = db.scalars(
                select(Conversation).where(
                    Conversation.user_id == user_id,
                    Conversation.deleted_at.is_(None),
                )
            ).all()
            now = datetime.now(UTC)
            for conversation in rows:
                conversation.deleted_at = now
                conversation.updated_at = now
            return len(rows)

    def run_turn(
        self,
        user_id: str,
        conversation_id: str,
        message: str,
        pipeline: Any,
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Run and persist one turn and its updated memory in a single transaction."""
        with self.sessions() as db, db.begin():
            conversation = self._owned(db, user_id, conversation_id)
            prior_user_message_count = (
                db.scalar(
                    select(func.count())
                    .select_from(Message)
                    .where(
                        Message.conversation_id == conversation.id,
                        Message.role == "user",
                    )
                )
                or 0
            )
            try:
                state = json.loads(conversation.state_json or "{}")
            except json.JSONDecodeError as exc:
                raise RuntimeError("Stored conversation memory is invalid JSON.") from exc
            if not isinstance(state, dict):
                raise RuntimeError("Stored conversation memory must be a JSON object.")
            chat_session = ChatSession.from_dict(state, memory_turns=self.memory_turns)
            result = pipeline.run(message, language=language, session=chat_session)
            payload = result.as_dict()
            now = datetime.now(UTC)
            db.add(
                Message(
                    id=str(uuid.uuid4()),
                    conversation_id=conversation.id,
                    role="user",
                    content=message,
                    created_at=now,
                )
            )
            metadata = {
                "items": payload.get("items", []),
                "resolved_references": payload.get("resolved_references", []),
                "intent": payload.get("intent"),
                "fallback": not bool(payload.get("answer_llm")),
                "latency": payload.get("durations", {}),
                "partial": False,
            }
            db.add(
                Message(
                    id=str(uuid.uuid4()),
                    conversation_id=conversation.id,
                    role="assistant",
                    content=str(payload.get("answer", "")),
                    meta_json=json.dumps(metadata, ensure_ascii=False),
                    created_at=now + timedelta(microseconds=1),
                )
            )
            conversation.state_json = json.dumps(
                chat_session.to_dict(), ensure_ascii=False, separators=(",", ":")
            )
            conversation.updated_at = now
            if prior_user_message_count == 0:
                conversation.title = _title(message, default="New chat")
            response = {
                "conversation_id": conversation.id,
                "title": conversation.title,
                "user_message": message,
                "result": payload,
            }
        return response

    @staticmethod
    def _owned(db: Session, user_id: str, conversation_id: str) -> Conversation:
        conversation = db.scalar(
            select(Conversation).where(
                Conversation.id == conversation_id,
                Conversation.user_id == user_id,
                Conversation.deleted_at.is_(None),
            )
        )
        if conversation is None:
            raise ConversationNotFound
        return conversation


def _title(value: str, *, default: str = "New chat") -> str:
    cleaned = " ".join(value.split())[:60]
    return cleaned or default


def _like_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _conversation_dict(
    conversation: Conversation, *, include_messages: bool = False
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }
    if include_messages:
        result["messages"] = [
            {
                "id": message.id,
                "role": message.role,
                "content": message.content,
                "meta": json.loads(message.meta_json) if message.meta_json else None,
                "created_at": message.created_at.isoformat(),
            }
            for message in conversation.messages
        ]
    return result


__all__ = ["ConversationNotFound", "ConversationService"]
