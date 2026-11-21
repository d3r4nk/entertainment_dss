"""Code-built conversational state for the chatbot."""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

_POSITION_PATTERNS = (
    re.compile(r"\b(?:cái|game|phim|mục|item)\s+(?:thứ\s*)?(\d+|một|hai|ba)\b", re.I),
    re.compile(r"\b(?:the\s+)?(first|second|third|last)\s+(?:one|item|game|movie)?\b", re.I),
    re.compile(r"\b(?:đầu tiên|cuối cùng)\b", re.I),
)
_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "một": 1,
    "hai": 2,
    "ba": 3,
    "đầu tiên": 1,
}
_ANAPHORA = re.compile(
    r"\b(?:nó|cái đó|game đó|phim đó|that one|it|this one)\b", re.I
)
_LIKE = re.compile(r"\b(?:tôi thích|mình thích|i like|i liked|love)\b", re.I)
_DISLIKE = re.compile(r"\b(?:không thích|ghét|đã chơi rồi|đã xem rồi|"
                      r"don't like|do not like|played it already|already played|"
                      r"seen it already|already seen)\b", re.I)
_REPEAT = re.compile(r"\b(?:lặp lại|nói lại|repeat|again|show again)\b", re.I)
_LIGHTER = re.compile(r"\b(?:ít bạo lực hơn|nhẹ nhàng hơn|something lighter|"
                      r"lighter|less violent|less violence)\b", re.I)
_LESS_VIOLENT = re.compile(r"\b(?:ít bạo lực hơn|less violent|less violence)\b", re.I)


@dataclass
class ChatTurn:
    """Verified, code-built facts from one completed turn."""

    user_message: str
    intent: dict[str, Any]
    resolved_references: dict[str, tuple[str, int]]
    items: list[dict[str, Any]]
    answer: str
    created_at: float = field(default_factory=time.time)


@dataclass
class ChatSession:
    """Bounded chat memory; answer text is retained for display, never context."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    language: str = "vi"
    memory_turns: int = 4
    turns: list[ChatTurn] = field(default_factory=list)
    liked_ids: dict[str, set[int]] = field(default_factory=dict)
    excluded_ids: dict[str, set[int]] = field(default_factory=dict)
    recommended_ids: dict[str, set[int]] = field(default_factory=dict)
    liked_references: dict[str, tuple[str, int]] = field(default_factory=dict)
    recent_domain: str | None = None
    recent_genres: list[str] = field(default_factory=list)
    recent_themes: list[str] = field(default_factory=list)
    recent_constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.memory_turns = max(1, int(self.memory_turns))

    def resolve_position(self, message: str) -> dict[str, tuple[str, int]]:
        """Resolve a rank or anaphoric phrase against the latest shown items."""
        if not self.turns:
            return {}
        shown = self.turns[-1].items
        if not shown:
            return {}

        position: int | None = None
        for pattern in _POSITION_PATTERNS:
            match = pattern.search(message)
            if not match:
                continue
            token = match.group(1).lower() if match.lastindex else match.group(0).lower()
            if token.isdigit():
                position = int(token)
            elif token in {"last", "cuối cùng"}:
                position = len(shown)
            else:
                position = _ORDINALS.get(token)
            break
        if position is None and _ANAPHORA.search(message):
            position = 1
        if position is None or position < 1 or position > len(shown):
            return {}
        item = shown[position - 1]
        title, domain, item_id = str(item["title"]), str(item["domain"]), int(item["id"])
        return {title: (domain, item_id)}

    def titles_in_history(self, message: str) -> dict[str, tuple[str, int]]:
        """Return prior shown titles mentioned literally in the new message."""
        lowered = message.casefold()
        resolved: dict[str, tuple[str, int]] = {}
        for turn in reversed(self.turns):
            for title, ref in turn.resolved_references.items():
                if title.casefold() in lowered:
                    resolved[title] = ref
            for item in turn.items:
                title = str(item["title"])
                if title.casefold() in lowered:
                    resolved[title] = (str(item["domain"]), int(item["id"]))
        return resolved

    def feedback_for(
        self, message: str, references: dict[str, tuple[str, int]]
    ) -> tuple[dict[str, set[int]], dict[str, set[int]], list[str]]:
        """Apply explicit feedback to verified referenced items and return constraints."""
        liked: dict[str, set[int]] = {}
        excluded: dict[str, set[int]] = {}
        if _LIKE.search(message):
            liked = _group_ids(references)
        if _DISLIKE.search(message):
            excluded = _group_ids(references)
        constraints: list[str] = []
        if _LESS_VIOLENT.search(message):
            constraints.append("avoid:violence")
        elif _LIGHTER.search(message):
            constraints.append("cosy")
        for domain, ids in liked.items():
            self.liked_ids.setdefault(domain, set()).update(ids)
            self.excluded_ids.setdefault(domain, set()).difference_update(ids)
            for title, (ref_domain, item_id) in references.items():
                if ref_domain == domain and item_id in ids:
                    self.liked_references[title] = (ref_domain, item_id)
        for domain, ids in excluded.items():
            self.excluded_ids.setdefault(domain, set()).update(ids)
            self.liked_ids.setdefault(domain, set()).difference_update(ids)
            self.liked_references = {
                title: ref
                for title, ref in self.liked_references.items()
                if ref[0] != domain or ref[1] not in ids
            }
        return liked, excluded, constraints

    def clear_recent_context(self) -> None:
        """Drop carried filters after a clear subject/domain switch."""
        self.recent_domain = None
        self.recent_genres.clear()
        self.recent_themes.clear()
        self.recent_constraints.clear()

    def add_turn(
        self,
        user_message: str,
        intent: Any,
        resolved_references: dict[str, tuple[str, int]],
        items: list[Any],
        answer: str,
    ) -> None:
        """Store verified turn facts and update bounded recommendation memory."""
        parsed = intent.model_dump() if hasattr(intent, "model_dump") else dict(intent)
        domains = parsed.get("domain", "both")
        self.recent_domain = domains if domains in {"movie", "game"} else self.recent_domain
        self.recent_genres = list(parsed.get("genres", []))[:8]
        self.recent_themes = list(parsed.get("themes", []))[:8]
        self.recent_constraints = list(dict.fromkeys(self.recent_constraints + self.recent_themes))[-8:]

        shown: list[dict[str, Any]] = []
        for position, item in enumerate(items, start=1):
            if isinstance(item, dict):
                domain = str(item.get("domain", ""))
                item_id = int(item.get("id", item.get("item_id", 0)))
                title = str(item.get("title", ""))
            else:
                domain = str(getattr(item, "domain", ""))
                item_id = int(getattr(item, "item_id", 0))
                title = str(getattr(item, "title", ""))
            if not domain and len(getattr(intent, "domains_queried", [])) == 1:
                domain = intent.domains_queried[0]
            shown.append({"position": position, "id": item_id, "domain": domain, "title": title})
            if domain:
                self.recommended_ids.setdefault(domain, set()).add(item_id)

        self.turns.append(
            ChatTurn(
                user_message=user_message,
                intent=parsed,
                resolved_references=dict(resolved_references),
                items=shown,
                answer=answer,
            )
        )
        self.turns = self.turns[-self.memory_turns :]

    def summary_for_intent(self, max_chars: int = 400) -> str:
        """Build a compact parser context from verified state, never answer text."""
        pieces: list[str] = []
        if self.recent_domain:
            pieces.append(f"domain={self.recent_domain}")
        if self.recent_genres:
            pieces.append("genres=" + ", ".join(self.recent_genres[:4]))
        if self.recent_themes:
            pieces.append("themes=" + ", ".join(self.recent_themes[:4]))
        if self.recent_constraints:
            pieces.append("constraints=" + ", ".join(self.recent_constraints[:4]))
        if self.liked_references:
            pieces.append("liked references=" + ", ".join(list(self.liked_references)[:3]))
        if self.turns and self.turns[-1].items:
            names = ", ".join(item["title"] for item in self.turns[-1].items[:3])
            pieces.append("previously shown=" + names)
        limit = min(400, max(0, int(max_chars)))
        return ("Verified conversation context: " + "; ".join(pieces))[:limit]

    @property
    def history_text(self) -> str:
        """User-displayable compact history, including answers only for display."""
        return "\n".join(
            f"You: {turn.user_message}\nAssistant: {turn.answer}" for turn in self.turns
        )

    def reset(self) -> None:
        """Clear conversational state while keeping this session identifier."""
        self.language = "vi"
        self.turns.clear()
        self.liked_ids.clear()
        self.excluded_ids.clear()
        self.recommended_ids.clear()
        self.liked_references.clear()
        self.clear_recent_context()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible state for durable conversation storage."""
        return {
            "session_id": self.session_id,
            "language": self.language,
            "memory_turns": self.memory_turns,
            "turns": [
                {
                    "user_message": turn.user_message,
                    "intent": turn.intent,
                    "resolved_references": [
                        [title, domain, item_id]
                        for title, (domain, item_id) in turn.resolved_references.items()
                    ],
                    "items": turn.items,
                    "answer": turn.answer,
                    "created_at": turn.created_at,
                }
                for turn in self.turns
            ],
            "liked_ids": _sets_to_dict(self.liked_ids),
            "excluded_ids": _sets_to_dict(self.excluded_ids),
            "recommended_ids": _sets_to_dict(self.recommended_ids),
            "liked_references": [
                [title, domain, item_id]
                for title, (domain, item_id) in self.liked_references.items()
            ],
            "recent_domain": self.recent_domain,
            "recent_genres": self.recent_genres,
            "recent_themes": self.recent_themes,
            "recent_constraints": self.recent_constraints,
        }

    @classmethod
    def from_dict(cls, state: dict[str, Any], *, memory_turns: int | None = None) -> ChatSession:
        """Rebuild bounded memory from persisted, code-generated state."""
        session = cls(
            session_id=str(state.get("session_id") or uuid.uuid4()),
            language=str(state.get("language", "vi")),
            memory_turns=int(memory_turns or state.get("memory_turns", 4)),
        )
        session.turns = [
            ChatTurn(
                user_message=str(row["user_message"]),
                intent=dict(row.get("intent", {})),
                resolved_references={
                    str(title): (str(domain), int(item_id))
                    for title, domain, item_id in row.get("resolved_references", [])
                },
                items=list(row.get("items", [])),
                answer=str(row.get("answer", "")),
                created_at=float(row.get("created_at", time.time())),
            )
            for row in state.get("turns", [])
        ][-session.memory_turns :]
        session.liked_ids = _dict_to_sets(state.get("liked_ids", {}))
        session.excluded_ids = _dict_to_sets(state.get("excluded_ids", {}))
        session.recommended_ids = _dict_to_sets(state.get("recommended_ids", {}))
        session.liked_references = {
            str(title): (str(domain), int(item_id))
            for title, domain, item_id in state.get("liked_references", [])
        }
        session.recent_domain = state.get("recent_domain")
        session.recent_genres = list(state.get("recent_genres", []))[:8]
        session.recent_themes = list(state.get("recent_themes", []))[:8]
        session.recent_constraints = list(state.get("recent_constraints", []))[-8:]
        return session


def _group_ids(references: dict[str, tuple[str, int]]) -> dict[str, set[int]]:
    grouped: dict[str, set[int]] = {}
    for domain, item_id in references.values():
        grouped.setdefault(domain, set()).add(int(item_id))
    return grouped


def _sets_to_dict(values: dict[str, set[int]]) -> dict[str, list[int]]:
    return {key: sorted(int(item_id) for item_id in item_ids) for key, item_ids in values.items()}


def _dict_to_sets(values: dict[str, list[int]]) -> dict[str, set[int]]:
    return {str(key): {int(item_id) for item_id in item_ids} for key, item_ids in values.items()}


__all__ = ["ChatSession", "ChatTurn"]
