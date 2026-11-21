"""Tests for rendering the persisted recommendation context in chat history."""

from __future__ import annotations

from entdec.ui.app import _render_saved_message


class _Recorder:
    def __init__(self) -> None:
        self.values: list[str] = []

    def __enter__(self) -> _Recorder:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def write(self, value: str) -> None:
        self.values.append(value)

    def caption(self, value: str) -> None:
        self.values.append(value)

    def markdown(self, value: str) -> None:
        self.values.append(value)

    def columns(self, count: int) -> list[_Recorder]:
        return [self for _ in range(count)]

    def expander(self, _label: str) -> _Recorder:
        return self


def test_saved_assistant_message_keeps_recommendation_cards_and_evidence() -> None:
    ui = _Recorder()
    _render_saved_message(
        ui,
        {
            "role": "assistant",
            "content": "Try this game.",
            "meta": {
                "resolved_references": ["Portal"],
                "items": [
                    {
                        "title": "Portal 2",
                        "score": 0.91,
                        "shared_themes": ["puzzle", "story"],
                        "shared_genres": ["Adventure"],
                        "evidence": "Portal 2: themes puzzle, story; match 0.91",
                    }
                ],
                "used_graph": True,
                "intent_llm": True,
                "answer_llm": True,
            },
        },
    )

    rendered = "\n".join(ui.values)
    assert "**1. Portal 2**" in rendered
    assert "puzzle, story (score 0.91)" in rendered
    assert "Portal 2: themes puzzle, story; match 0.91" in rendered
    assert "Resolved references: Portal" in rendered
    assert "graph: yes · intent: LLM · answer: LLM" in rendered
