"""Unit tests for the ``python -m entdec all`` sequencing (no services)."""

from __future__ import annotations

import argparse

import pytest

from entdec import pipeline
from entdec.config import Config


def _step(name: str, counter: list[str], code: int = 0) -> pipeline.Step:
    """A step that records its execution and returns a fixed exit code."""
    return pipeline.Step(
        name,
        lambda: (counter.append(name), code)[1],  # type: ignore[sub-type]
    )


def test_run_executes_all_steps_in_order(
    make_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config()  # type: ignore[operator]
    executed: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "build_steps",
        lambda cfg, *, force, skip_llm, no_graph: [
            _step(name, executed) for name in ("data", "texts", "recsys")
        ],
    )
    assert pipeline.run(cfg, argparse.Namespace(force=False, skip_llm=False)) == 0
    assert executed == ["data", "texts", "recsys"]


def test_run_stops_at_first_failing_step(
    make_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config()  # type: ignore[operator]
    executed: list[str] = []
    monkeypatch.setattr(
        pipeline,
        "build_steps",
        lambda cfg, *, force, skip_llm, no_graph: [
            _step("a", executed),
            _step("bad", executed, code=3),
            _step("after", executed),
        ],
    )
    assert pipeline.run(cfg, argparse.Namespace(force=False, skip_llm=False)) == 1
    assert executed == ["a", "bad"]


def test_run_fails_when_no_step_ran(
    make_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config()  # type: ignore[operator]
    monkeypatch.setattr(
        pipeline, "build_steps", lambda cfg, *, force, skip_llm, no_graph: []
    )
    assert pipeline.run(cfg, argparse.Namespace(force=False, skip_llm=False)) == 1


def test_build_steps_order_and_skip_llm(make_config: object) -> None:
    cfg: Config = make_config()  # type: ignore[operator]
    steps = pipeline.build_steps(cfg, force=False, skip_llm=True)
    names = [step.name for step in steps]
    assert names == [
        "data",
        "texts",
        "graph use",
        "recsys",
        "cross",
    ]
    assert not any(name in {"taxonomy", "embed"} for name in names)


def test_build_steps_omits_graph_when_disabled(make_config: object) -> None:
    cfg: Config = make_config()  # type: ignore[operator]
    steps = pipeline.build_steps(cfg, force=False, skip_llm=True, no_graph=True)
    assert "graph use" not in [step.name for step in steps]
