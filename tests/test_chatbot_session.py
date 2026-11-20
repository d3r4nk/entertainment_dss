"""Unit tests for code-built conversational memory."""

from __future__ import annotations

from entdec.chatbot.session import ChatSession


def _session() -> ChatSession:
    session = ChatSession(memory_turns=4)
    session.add_turn(
        "recommend games",
        {"domain": "game", "genres": [], "themes": ["story rich"]},
        {"Portal 2": ("game", 22)},
        [
            {"id": 11, "domain": "game", "title": "Hades"},
            {"id": 22, "domain": "game", "title": "Portal 2"},
            {"id": 33, "domain": "game", "title": "Journey"},
        ],
        "An answer mentioning only retrieved items.",
    )
    return session


def test_position_and_anaphora_resolve_latest_displayed_item() -> None:
    session = _session()
    assert session.resolve_position("cái thứ 2 là gì?") == {
        "Portal 2": ("game", 22)
    }
    assert session.resolve_position("the second one") == {"Portal 2": ("game", 22)}
    assert session.resolve_position("đầu tiên") == {"Hades": ("game", 11)}
    assert session.resolve_position("cuối cùng") == {"Journey": ("game", 33)}
    assert session.resolve_position("that one") == {"Hades": ("game", 11)}
    assert session.resolve_position("the ninth one") == {}


def test_feedback_updates_verified_item_aggregates_and_constraints() -> None:
    session = _session()
    refs = session.resolve_position("cái thứ hai")
    liked, excluded, constraints = session.feedback_for("Tôi thích cái thứ hai", refs)
    assert liked == {"game": {22}}
    assert excluded == {}
    assert session.liked_ids == {"game": {22}}
    assert session.liked_references == {"Portal 2": ("game", 22)}

    refs = session.resolve_position("that one")
    _liked, excluded, constraints = session.feedback_for("I've already played that one", refs)
    assert excluded == {"game": {11}}
    assert constraints == []
    assert session.excluded_ids == {"game": {11}}

    _liked, _excluded, constraints = session.feedback_for(
        "something lighter", session.resolve_position("that one")
    )
    assert constraints == ["cosy"]
    _liked, _excluded, constraints = session.feedback_for(
        "less violence", session.resolve_position("that one")
    )
    assert constraints == ["avoid:violence"]


def test_summary_is_bounded_and_never_uses_saved_answer() -> None:
    session = _session()
    summary = session.summary_for_intent()
    assert len(summary) <= 400
    assert len(session.summary_for_intent(800)) <= 400
    assert "story rich" in summary
    assert "An answer mentioning" not in summary
    assert "Portal 2" in summary


def test_reset_clears_state_but_preserves_session_identity() -> None:
    session = _session()
    session_id = session.session_id
    session.reset()
    assert session.session_id == session_id
    assert session.turns == []
    assert session.recommended_ids == {}
    assert session.liked_ids == {}
    assert session.history_text == ""


def test_turn_history_is_bounded() -> None:
    session = ChatSession(memory_turns=2)
    for index in range(3):
        session.add_turn(str(index), {"domain": "game"}, {}, [], "display only")
    assert [turn.user_message for turn in session.turns] == ["1", "2"]
