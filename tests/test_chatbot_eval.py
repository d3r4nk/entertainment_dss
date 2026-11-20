"""Offline checks for the scripted memory evaluation dataset."""

from __future__ import annotations

from pathlib import Path

import yaml


def test_chat_memory_eval_has_twenty_three_turn_conversations() -> None:
    root = Path(__file__).resolve().parents[1]
    data = yaml.safe_load(
        (root / "config" / "chat_eval_conversations.yaml").read_text(encoding="utf-8")
    )
    conversations = data["conversations"]
    assert len(conversations) == 20
    assert all(len(conversation["turns"]) == 3 for conversation in conversations)
    messages = " ".join(
        message for conversation in conversations for message in conversation["turns"]
    )
    assert "Tôi thích" in messages
    assert "I like" in messages
