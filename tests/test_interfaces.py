"""Smoke tests for the chatbot interfaces (CLI + Streamlit launcher).

No real services: the pipeline and Streamlit are stubbed, so these run on the
tiny synthetic config only.
"""

from __future__ import annotations

import argparse
import subprocess
from typing import Any

import pytest

from entdec.chatbot import cli as chat_cli
from entdec.chatbot.pipeline import Candidate, ChatResult
from entdec.config import Config
from entdec.llm.intent import Intent
from entdec.ui import app as ui_app


class _StubPipeline:
    def __init__(self, cfg: Config, *, use_graph: bool = True) -> None:
        self.cfg = cfg
        self.questions: list[str] = []

    def run(self, question: str, *, language: str | None = None, top_k: int | None = None) -> ChatResult:
        self.questions.append(question)
        item = Candidate(
            item_id=1,
            title="Journey",
            position=0,
            scores={"total": 0.8},
            shared_themes=["exploration"],
        )
        return ChatResult(
            question=question,
            language=language or "vi",
            intent=Intent(domain="game", language=language or "vi", parsed_by="rules"),
            domains_queried=["game"],
            items=[item],
            answer="1. Journey: Hành trình khám phá đầy cảm xúc.",
            used_graph=True,
            intent_llm=False,
            answer_llm=True,
            durations={"total_s": 0.1},
        )


def test_cli_answers_one_question(
    synthetic_config: Config, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from entdec.chatbot import pipeline as pipe_mod

    monkeypatch.setattr(pipe_mod, "ChatPipeline", _StubPipeline)
    args = argparse.Namespace(question="game khám phá?", language=None, no_graph=True)
    assert chat_cli.run(synthetic_config, args) == 0
    out = capsys.readouterr().out
    assert "Hành trình khám phá" in out  # the grounded answer
    assert "Journey" in out  # the evidence list


def test_cli_empty_question_interactive_exit(
    synthetic_config: Config, monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    from entdec.chatbot import pipeline as pipe_mod

    monkeypatch.setattr(pipe_mod, "ChatPipeline", _StubPipeline)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "exit")
    assert chat_cli.run(
        synthetic_config, argparse.Namespace(question=None, language=None, no_graph=True)
    ) == 0
    assert "Entertainment chatbot" in capsys.readouterr().out


def test_ui_launches_streamlit_with_profile(
    synthetic_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_call(command: list[str]) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(subprocess, "call", fake_call)
    assert ui_app.run(synthetic_config, argparse.Namespace(port=8599, no_graph=True)) == 0
    command = calls[0]
    assert "streamlit" in command and "run" in command
    assert str(ui_app.__file__) in command
    assert command[-3:] == ["--profile", synthetic_config.profile.name, "--no-graph"]
    assert "8599" in command
