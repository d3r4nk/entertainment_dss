"""Offline tests for graph identity, dump metadata, and profile switching."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from entdec.config import Config
from entdec.graph import store
from entdec.graph.loader import run as graph_run
from entdec.graph.store import GraphStoreError


class MarkerClient:
    def __init__(self) -> None:
        self.marker: dict[str, Any] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        if "UNWIND labels(n)" in query:
            return [{"label": "Movie", "count": 2}, {"label": "GraphMeta", "count": 1}]
        if "RETURN type(r)" in query:
            return [{"type": "HAS_THEME", "count": 4}]
        if "RETURN count(n) AS count" in query:
            return [{"count": 3}]
        if "RETURN properties(m) AS marker" in query:
            return [{"marker": dict(self.marker)}] if self.marker else []
        if "MERGE (m:GraphMeta" in query:
            self.marker["profile"] = params["profile"]
        if "SET m.profile = $profile, m.built_at" in query:
            self.marker.update(params)
            self.marker["counts"] = params["counts"]
        return []

    def is_ready(self) -> bool:
        return True


class ReadyClient:
    def is_ready(self) -> bool:
        return True

    def wait_until_ready(self, timeout_s: float = 30) -> bool:
        return True


def test_marker_and_dump_metadata_round_trip(
    make_config: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config("tiny")  # type: ignore[operator]
    cfg.paths.root = tmp_path
    monkeypatch.setattr(store, "graph_artifact_hash", lambda _cfg: "sha256-value")
    monkeypatch.setattr(store, "graph_embed_dim", lambda _cfg: 8)
    client = MarkerClient()

    marker = store.write_graph_marker(client, cfg)

    assert marker["profile"] == "tiny"
    assert marker["artefact_hash"] == "sha256-value"
    assert marker["embed_dim"] == 8
    assert marker["counts"]["node_count"] == 3
    assert marker["counts"]["relationship_count"] == 4
    assert client.marker["profile"] == "tiny"
    stored_counts = json.loads(client.marker["counts"])
    assert stored_counts["nodes"]["GraphMeta"] == 1

    store.dump_path(cfg, "tiny").parent.mkdir(parents=True, exist_ok=True)
    store.dump_path(cfg, "tiny").write_bytes(b"test dump")
    metadata = store.write_dump_metadata(cfg, "tiny", marker)
    assert store.read_dump_metadata(cfg, "tiny") == metadata
    assert metadata == {
        "profile": "tiny",
        "created_at": metadata["created_at"],
        "node_count": 3,
        "relationship_count": 4,
        "counts": marker["counts"],
        "artefact_hash": "sha256-value",
        "embed_dim": 8,
    }


@pytest.mark.parametrize(
    ("marker", "dump", "expected"),
    [
        ({"profile": "tiny", "artefact_hash": "x"}, None, "noop"),
        ({"profile": "tiny", "artefact_hash": "old"}, {"profile": "tiny", "artefact_hash": "x"}, "restore"),
        ({"profile": "tiny", "artefact_hash": "old"}, {"profile": "default", "artefact_hash": "old"}, "build"),
        (None, None, "build"),
    ],
)
def test_graph_use_decision(
    marker: dict[str, Any] | None, dump: dict[str, Any] | None, expected: str
) -> None:
    assert store.choose_graph_action(marker, "tiny", "x", dump) == expected


def test_stop_container_allows_time_for_clean_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(
        store,
        "_run",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )

    store._stop_container()

    assert calls == [
        (
            [
                "docker",
                "compose",
                "-f",
                str(store.COMPOSE_FILE),
                "stop",
                "--timeout",
                "180",
                store.COMPOSE_SERVICE,
            ],
            {"timeout_s": 180},
        )
    ]


def test_use_profile_restores_matching_dump_with_mocked_io(
    make_config: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config("default")  # type: ignore[operator]
    active_marker = {"profile": "tiny", "artefact_hash": "tiny-hash"}
    restored_marker = {"profile": "default", "artefact_hash": "default-hash"}
    target_dump = tmp_path / "default.dump"
    target_dump.write_bytes(b"dump")
    monkeypatch.setattr(store, "graph_artifact_hash", lambda _cfg: "default-hash")
    monkeypatch.setattr(store, "dump_path", lambda _cfg, _profile: target_dump)
    monkeypatch.setattr(
        store,
        "read_dump_metadata",
        lambda _cfg, profile: {"profile": profile, "artefact_hash": "default-hash"},
    )
    monkeypatch.setattr(store, "_marker_or_infer", lambda _cfg, _client: active_marker)
    monkeypatch.setattr(store, "_marker_is_dumped", lambda *_args: True)
    calls: list[tuple[str, Any]] = []

    def restore(_cfg: Config, profile: str, **kwargs: Any) -> dict[str, Any]:
        calls.append((profile, kwargs))
        return restored_marker

    monkeypatch.setattr(store, "restore_profile", restore)
    result = store.use_profile(cfg, client=ReadyClient())

    assert result == restored_marker
    assert len(calls) == 1
    assert calls[0][0] == "default"
    assert calls[0][1]["preserve_current"] is False
    assert calls[0][1]["force"] is False
    assert isinstance(calls[0][1]["client"], ReadyClient)


def test_use_profile_builds_and_dumps_when_no_matching_dump(
    make_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config("default")  # type: ignore[operator]
    marker = {"profile": "tiny", "artefact_hash": "tiny-hash"}
    built_marker = {"profile": "default", "artefact_hash": "default-hash"}
    monkeypatch.setattr(store, "graph_artifact_hash", lambda _cfg: "default-hash")
    monkeypatch.setattr(store, "read_dump_metadata", lambda *_args: None)
    monkeypatch.setattr(store, "_marker_or_infer", lambda _cfg, _client: marker)
    monkeypatch.setattr(store, "_marker_is_dumped", lambda *_args: True)
    calls: list[str] = []
    monkeypatch.setattr(store, "build_profile", lambda *_args: built_marker)
    monkeypatch.setattr(
        store,
        "_dump_loaded",
        lambda _cfg, _client, _marker, **_kwargs: calls.append("dump") or {},
    )

    assert store.use_profile(cfg, client=ReadyClient()) == built_marker
    assert calls == ["dump"]


def test_use_profile_dumps_current_graph_before_rebuilding(
    make_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config("default")  # type: ignore[operator]
    current = {"profile": "tiny", "artefact_hash": "tiny-hash"}
    built = {"profile": "default", "artefact_hash": "default-hash"}
    monkeypatch.setattr(store, "graph_artifact_hash", lambda _cfg: "default-hash")
    monkeypatch.setattr(store, "read_dump_metadata", lambda *_args: None)
    monkeypatch.setattr(store, "_marker_or_infer", lambda _cfg, _client: current)
    monkeypatch.setattr(store, "_marker_is_dumped", lambda *_args: False)
    monkeypatch.setattr(store, "build_profile", lambda *_args: built)
    operations: list[tuple[str, str]] = []
    monkeypatch.setattr(
        store,
        "_dump_loaded",
        lambda _cfg, _client, marker, **_kwargs: operations.append(
            ("dump", marker["profile"])
        )
        or {},
    )

    store.use_profile(cfg, client=ReadyClient())

    assert operations == [("dump", "tiny"), ("dump", "default")]


def test_profile_refusal_includes_graph_use_guidance() -> None:
    with pytest.raises(GraphStoreError, match="graph use"):
        store.require_loaded_profile({"profile": "default"}, "tiny", "load")


def test_dump_refuses_a_different_loaded_profile(
    make_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config("tiny")  # type: ignore[operator]
    monkeypatch.setattr(
        store, "_marker_or_infer", lambda *_args: {"profile": "default"}
    )

    with pytest.raises(GraphStoreError, match="graph use"):
        store.dump_profile(cfg, "tiny", client=ReadyClient())


def test_graph_load_refuses_a_different_loaded_profile(
    make_config: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg: Config = make_config("tiny")  # type: ignore[operator]
    monkeypatch.setattr(
        "entdec.graph.loader.current_marker",
        lambda *_args: {"profile": "default", "artefact_hash": "x"},
    )
    monkeypatch.setattr(
        "entdec.graph.loader.load_all",
        lambda *_args, **_kwargs: pytest.fail("load must not run"),
        raising=False,
    )
    assert graph_run(cfg, type("Args", (), {"action": "load", "force": False})(), ReadyClient()) == 1


def test_container_inspection_uses_named_entdec_container_and_real_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    container = {
        "Id": "container-id",
        "Name": "/entdec-neo4j",
        "Config": {"Image": store.NEO4J_IMAGE},
        "NetworkSettings": {
            "Ports": {
                "7474/tcp": [{"HostPort": "7475"}],
                "7687/tcp": [{"HostPort": "7688"}],
            }
        },
        "Mounts": [
            {"Destination": "/data", "Type": "volume", "Name": "actual_data_volume"}
        ],
    }
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        output = json.dumps(container) if command[1] == "inspect" else "container-id\n"
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(store, "_run", run)

    assert store._container_info() == ("container-id", "actual_data_volume")
    assert all("movie-neo4j" not in " ".join(command) for command in commands)
    assert any("entdec-neo4j" in command for command in commands)
