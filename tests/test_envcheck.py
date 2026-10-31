"""Tests for :mod:`entdec.envcheck`."""

from __future__ import annotations

import httpx
import pytest
from tests.conftest import write_csv

from entdec import envcheck
from entdec.config import Config


def test_list_ollama_models_unreachable_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise httpx.ConnectError("no server")

    monkeypatch.setattr(envcheck.httpx, "get", boom)
    assert envcheck.list_ollama_models("http://127.0.0.1:1") == []


def test_list_ollama_models_parses_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"models": [{"name": "qwen3:4b"}, {"name": "qwen3-embedding:4b"}]}

    monkeypatch.setattr(envcheck.httpx, "get", lambda *a, **k: FakeResponse())
    assert envcheck.list_ollama_models("http://localhost:11434") == [
        "qwen3-embedding:4b",
        "qwen3:4b",
    ]


def test_run_env_check_ok_on_synthetic(
    synthetic_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envcheck, "list_ollama_models", lambda host, timeout=5.0: [])
    report = envcheck.run_env_check(synthetic_config)
    assert report.python_version
    assert "pandas" in report.packages
    assert report.files
    assert report.ok, report.problems


def test_run_env_check_flags_missing_files(make_config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envcheck, "list_ollama_models", lambda host, timeout=5.0: [])
    cfg = make_config("tiny")  # no raw files created
    report = envcheck.run_env_check(cfg)
    assert not report.ok
    assert any("missing data file" in problem for problem in report.problems)


def test_run_env_check_detects_missing_model(
    synthetic_config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(envcheck, "list_ollama_models", lambda host, timeout=5.0: ["other:1b"])
    report = envcheck.run_env_check(synthetic_config)
    assert any("not pulled" in problem for problem in report.problems)


def test_file_info_reports_size(tmp_path) -> None:
    path = write_csv(tmp_path / "x.csv", ["a"], [(1,), (2,)])
    info = envcheck._file_info(path, expected_rows=2)
    assert info["exists"] is True
    assert info["expected_rows"] == 2
    assert info["size_mb"] >= 0.0
