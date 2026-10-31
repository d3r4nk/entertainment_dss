"""Profile-aware offline Neo4j dump storage and graph switching."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as parquet

from entdec.config import Config, load_config
from entdec.graph.neo4j_client import COMPOSE_FILE, Neo4jClient
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

NEO4J_IMAGE = "neo4j:5.26-community"
CONTAINER_NAME = "entdec-neo4j"
COMPOSE_SERVICE = "neo4j"
MARKER_ID = "active"
GRAPH_DUMPS_RELATIVE = Path(".cache") / "graph_dumps"


class GraphStoreError(RuntimeError):
    """Raised when a graph cannot safely be dumped, restored, or switched."""


def graph_dumps_dir(cfg: Config) -> Path:
    return cfg.paths.root / GRAPH_DUMPS_RELATIVE


def graph_artifact_hash(cfg: Config) -> str:
    """Hash every processed Parquet input used to construct the graph."""
    digest = hashlib.sha256()
    files = sorted(cfg.paths.processed.glob("*.parquet"))
    if not files:
        raise GraphStoreError(f"no processed Parquet artefacts in {cfg.paths.processed}")
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def graph_embed_dim(cfg: Config) -> int:
    path = cfg.processed_path("item_embeddings.parquet")
    if not path.exists():
        raise GraphStoreError(f"embedding artefact is missing: {path}")
    frame = pd.read_parquet(path, columns=["embedding"])
    if frame.empty:
        raise GraphStoreError(f"embedding artefact is empty: {path}")
    return len(frame.iloc[0]["embedding"])


def graph_counts(client: Any) -> dict[str, Any]:
    node_rows = client.run(
        "MATCH (n) UNWIND labels(n) AS label "
        "RETURN label, count(*) AS count ORDER BY label"
    )
    relationship_rows = client.run(
        "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS count ORDER BY type"
    )
    nodes = {str(row["label"]): int(row["count"]) for row in node_rows}
    relationships = {str(row["type"]): int(row["count"]) for row in relationship_rows}
    return {
        "nodes": nodes,
        "relationships": relationships,
        "node_count": int(client.run("MATCH (n) RETURN count(n) AS count")[0]["count"]),
        "relationship_count": sum(relationships.values()),
    }


def read_graph_marker(client: Any) -> dict[str, Any] | None:
    rows = client.run(
        "MATCH (m:GraphMeta {id: $id}) RETURN properties(m) AS marker",
        id=MARKER_ID,
    )
    if not rows:
        return None
    marker = dict(rows[0]["marker"])
    raw_counts = marker.get("counts")
    if isinstance(raw_counts, str):
        try:
            marker["counts"] = json.loads(raw_counts)
        except json.JSONDecodeError:
            marker["counts"] = {}
    return marker


def write_graph_marker(client: Any, cfg: Config, profile: str | None = None) -> dict[str, Any]:
    """Write the active graph identity and a count snapshot into Neo4j."""
    selected_profile = profile or cfg.profile.name
    client.run(
        "MERGE (m:GraphMeta {id: $id}) "
        "SET m.profile = $profile",
        id=MARKER_ID,
        profile=selected_profile,
    )
    counts = graph_counts(client)
    built_at = datetime.now(timezone.utc).isoformat()
    marker = {
        "profile": selected_profile,
        "built_at": built_at,
        "counts": counts,
        "artefact_hash": graph_artifact_hash(cfg),
        "embed_dim": graph_embed_dim(cfg),
    }
    client.run(
        "MATCH (m:GraphMeta {id: $id}) "
        "SET m.profile = $profile, m.built_at = $built_at, "
        "m.counts = $counts, m.node_count = $node_count, "
        "m.relationship_count = $relationship_count, "
        "m.artefact_hash = $artefact_hash, m.embed_dim = $embed_dim",
        id=MARKER_ID,
        profile=marker["profile"],
        built_at=built_at,
        counts=json.dumps(counts, sort_keys=True),
        node_count=counts["node_count"],
        relationship_count=counts["relationship_count"],
        artefact_hash=marker["artefact_hash"],
        embed_dim=marker["embed_dim"],
    )
    return marker


def dump_metadata_path(cfg: Config, profile: str) -> Path:
    return graph_dumps_dir(cfg) / f"{profile}.meta.json"


def dump_path(cfg: Config, profile: str) -> Path:
    return graph_dumps_dir(cfg) / f"{profile}.dump"


def read_dump_metadata(cfg: Config, profile: str) -> dict[str, Any] | None:
    path = dump_metadata_path(cfg, profile)
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GraphStoreError(f"cannot read graph dump metadata {path}: {exc}") from exc
    return value if isinstance(value, dict) else None


def write_dump_metadata(
    cfg: Config,
    profile: str,
    marker: dict[str, Any],
    *,
    created_at: str | None = None,
) -> dict[str, Any]:
    data = {
        "profile": profile,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "node_count": int(marker["counts"]["node_count"]),
        "relationship_count": int(marker["counts"]["relationship_count"]),
        "counts": marker["counts"],
        "artefact_hash": str(marker["artefact_hash"]),
        "embed_dim": int(marker["embed_dim"]),
    }
    dump_metadata_path(cfg, profile).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return data


def require_loaded_profile(marker: dict[str, Any] | None, profile: str, action: str) -> None:
    loaded = marker.get("profile") if marker else None
    if loaded != profile:
        raise GraphStoreError(
            f"graph {action} refused: requested profile {profile!r}, "
            f"but loaded profile is {loaded!r}; run `python -m entdec --profile "
            f"{profile} graph use` first"
        )


def choose_graph_action(
    loaded_marker: dict[str, Any] | None,
    target_profile: str,
    target_hash: str,
    target_dump_meta: dict[str, Any] | None,
) -> str:
    """Pure decision used by graph use: no-op, restore a matching dump, or build."""
    if (
        loaded_marker
        and loaded_marker.get("profile") == target_profile
        and loaded_marker.get("artefact_hash") == target_hash
    ):
        return "noop"
    if (
        target_dump_meta
        and target_dump_meta.get("profile") == target_profile
        and target_dump_meta.get("artefact_hash") == target_hash
    ):
        return "restore"
    return "build"


def _run(command: list[str], *, timeout_s: float = 3600.0) -> subprocess.CompletedProcess[str]:
    logger.info("step: %s", " ".join(command))
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(COMPOSE_FILE.parent),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.stdout.strip() and not (
        len(command) >= 3 and command[0:2] == ["docker", "inspect"]
    ):
        logger.info("%s", result.stdout.strip())
    if result.returncode:
        raise GraphStoreError(
            f"command failed ({result.returncode}): {' '.join(command)}\n"
            f"{(result.stderr or result.stdout).strip()[-2000:]}"
        )
    logger.info("step complete in %.1fs", time.perf_counter() - started)
    return result


def _container_info() -> tuple[str, str]:
    inspected = _run(
        [
            "docker",
            "inspect",
            CONTAINER_NAME,
            "--format",
            "{{json .}}",
        ],
        timeout_s=30,
    )
    try:
        info = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        raise GraphStoreError("docker inspect returned invalid container JSON") from exc
    if info.get("Config", {}).get("Image") != NEO4J_IMAGE:
        raise GraphStoreError(
            f"refusing graph operation: {CONTAINER_NAME} image is "
            f"{info.get('Config', {}).get('Image')!r}, expected {NEO4J_IMAGE!r}"
        )
    if info.get("Name", "").lstrip("/") != CONTAINER_NAME:
        raise GraphStoreError(f"refusing graph operation on unexpected container {info.get('Name')!r}")

    ports = info.get("NetworkSettings", {}).get("Ports", {})
    expected_ports = {"7474/tcp": "7475", "7687/tcp": "7688"}
    for container_port, host_port in expected_ports.items():
        bindings = ports.get(container_port) or []
        if not any(binding.get("HostPort") == host_port for binding in bindings):
            raise GraphStoreError(
                f"refusing graph operation: {CONTAINER_NAME} does not publish "
                f"{host_port}->{container_port}"
            )
    data_mounts = [
        mount
        for mount in info.get("Mounts", [])
        if mount.get("Destination") == "/data" and mount.get("Type") == "volume"
    ]
    if len(data_mounts) != 1:
        raise GraphStoreError(f"expected exactly one named /data volume on {CONTAINER_NAME}")
    volume = str(data_mounts[0]["Name"])
    if not volume:
        raise GraphStoreError("Neo4j data volume name is empty")
    compose_id = _run(
        ["docker", "compose", "-f", str(COMPOSE_FILE), "ps", "-q", COMPOSE_SERVICE],
        timeout_s=30,
    ).stdout.strip()
    if not compose_id or compose_id != info.get("Id"):
        raise GraphStoreError(
            f"compose service {COMPOSE_SERVICE!r} does not resolve to {CONTAINER_NAME}; "
            "refusing to alter containers"
        )
    logger.info("verified container=%s image=%s data-volume=%s", CONTAINER_NAME, NEO4J_IMAGE, volume)
    return str(info["Id"]), volume


def _stop_container() -> None:
    _run(
        [
            "docker",
            "compose",
            "-f",
            str(COMPOSE_FILE),
            "stop",
            "--timeout",
            "180",
            COMPOSE_SERVICE,
        ],
        timeout_s=180,
    )


def _start_container(container_id: str) -> None:
    _run(["docker", "compose", "-f", str(COMPOSE_FILE), "start", COMPOSE_SERVICE], timeout_s=180)
    inspected = _run(
        ["docker", "inspect", CONTAINER_NAME, "--format", "{{.Id}}"], timeout_s=30
    ).stdout.strip()
    if inspected != container_id:
        raise GraphStoreError(
            f"compose start changed the container id for {CONTAINER_NAME}; expected existing container"
        )


def _admin_dump(volume: str, work_dir: Path) -> None:
    _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{volume}:/data",
            "-v",
            f"{work_dir.resolve()}:/dumps",
            NEO4J_IMAGE,
            "neo4j-admin",
            "database",
            "dump",
            "neo4j",
            "--to-path=/dumps",
            "--overwrite-destination=true",
        ],
        timeout_s=3600,
    )


def _admin_load(volume: str, work_dir: Path) -> None:
    _run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{volume}:/data",
            "-v",
            f"{work_dir.resolve()}:/dumps",
            NEO4J_IMAGE,
            "neo4j-admin",
            "database",
            "load",
            "neo4j",
            "--from-path=/dumps",
            "--overwrite-destination=true",
        ],
        timeout_s=3600,
    )


def _wait_for_graph(cfg: Config, timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    client = Neo4jClient.from_config(cfg)
    try:
        while time.monotonic() < deadline:
            try:
                status = _run(
                    [
                        "docker",
                        "inspect",
                        CONTAINER_NAME,
                        "--format",
                        "{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}",
                    ],
                    timeout_s=20,
                ).stdout.strip()
                if status in {"healthy", "none"} and client.is_ready():
                    indexes = client.run(
                        "SHOW INDEXES YIELD name, state WHERE name = $name RETURN state",
                        name="item_embedding",
                    )
                    if indexes and indexes[0].get("state") == "ONLINE":
                        return
            except Exception as exc:  # noqa: BLE001 - readiness is polled until timeout
                logger.info("waiting for Neo4j health/vector index (%s)", exc)
            time.sleep(5)
    finally:
        client.close()
    raise GraphStoreError(
        f"Neo4j at {cfg.env.neo4j_uri} did not become healthy with vector index ONLINE "
        f"within {timeout_s:.0f}s"
    )


def _marker_or_infer(cfg: Config, client: Any) -> dict[str, Any] | None:
    marker = read_graph_marker(client)
    if marker:
        return marker
    counts = graph_counts(client)
    if counts["node_count"] == 0:
        return None
    graph_labels = counts["nodes"]
    for profile in ("tiny", "default"):
        candidate = load_config(profile)
        expected = _profile_catalogue_counts(candidate)
        expected_relationships = _profile_relationship_counts(candidate)
        if (
            all(graph_labels.get(label, 0) == value for label, value in expected.items())
            and all(
                counts["relationships"].get(rel_type, 0) == value
                for rel_type, value in expected_relationships.items()
            )
        ):
            logger.warning(
                "legacy graph has no GraphMeta marker; node and relationship counts match "
                "profile %s, recording its identity",
                profile,
            )
            return write_graph_marker(client, candidate, profile)
    raise GraphStoreError(
        "Neo4j contains a non-empty graph with no GraphMeta marker and its profile "
        "cannot be identified safely; refusing to overwrite it"
    )


def current_marker(cfg: Config, client: Any) -> dict[str, Any] | None:
    """Return the active graph marker, inferring and recording legacy identity."""
    return _marker_or_infer(cfg, client)


def _profile_catalogue_counts(cfg: Config) -> dict[str, int]:
    from entdec.graph.loader import user_rows

    return {
        "Movie": len(pd.read_parquet(cfg.processed_path("movies.parquet"))),
        "Game": len(pd.read_parquet(cfg.processed_path("games.parquet"))),
        "User": len(user_rows(cfg)),
    }


def _profile_relationship_counts(cfg: Config) -> dict[str, int]:
    """Expected relationship totals used only to identify a legacy unmarked graph."""
    processed = cfg.paths.processed

    def row_count(filename: str) -> int:
        path = processed / filename
        return int(parquet.ParquetFile(path).metadata.num_rows) if path.exists() else 0

    return {
        "HAS_GENRE": row_count("item_genres.parquet"),
        "HAS_THEME": row_count("movie_themes.parquet") + row_count("game_themes.parquet"),
        "RATED": row_count("movie_ratings.parquet"),
        "REVIEWED": row_count("game_reviews.parquet"),
        "SIMILAR_TO": row_count("similar_to.parquet"),
    }


def _marker_is_dumped(cfg: Config, marker: dict[str, Any] | None) -> bool:
    if not marker:
        return True
    profile = str(marker.get("profile", ""))
    metadata = read_dump_metadata(cfg, profile)
    path = dump_path(cfg, profile)
    if not metadata or not path.is_file():
        return False
    try:
        marker_time = datetime.fromisoformat(str(marker["built_at"]))
        dump_time = datetime.fromisoformat(str(metadata["created_at"]))
    except (KeyError, ValueError):
        return False
    return (
        metadata.get("artefact_hash") == marker.get("artefact_hash")
        and dump_time >= marker_time
    )


def preserve_current_graph(
    cfg: Config, client: Any, marker: dict[str, Any] | None, *, force: bool = False
) -> None:
    """Snapshot an active graph if its profile dump does not contain its marker."""
    if marker and not _marker_is_dumped(cfg, marker):
        _dump_loaded(cfg, client, marker, force=force)


def _dump_loaded(
    cfg: Config,
    client: Any,
    marker: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    profile = str(marker["profile"])
    target = dump_path(cfg, profile)
    meta_path = dump_metadata_path(cfg, profile)
    if (target.exists() or meta_path.exists()) and not force:
        raise GraphStoreError(
            f"dump already exists for {profile}; pass --force to overwrite {target.name}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    container_id, volume = _container_info()
    work_dir = target.parent / f".dump-{profile}-{uuid.uuid4().hex}"
    work_dir.mkdir()
    started = time.perf_counter()
    try:
        logger.info("stopping %s for offline graph dump", CONTAINER_NAME)
        _stop_container()
        _admin_dump(volume, work_dir)
        generated = work_dir / "neo4j.dump"
        if not generated.is_file():
            raise GraphStoreError(f"neo4j-admin succeeded but dump is missing: {generated}")
        if target.exists() and force:
            target.unlink()
        generated.replace(target)
        metadata = write_dump_metadata(cfg, profile, marker)
        logger.info(
            "dumped profile=%s size=%.2f GiB elapsed=%.1fs",
            profile,
            target.stat().st_size / (1024**3),
            time.perf_counter() - started,
        )
        return metadata
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _start_container(container_id)
        _wait_for_graph(cfg)


def dump_profile(cfg: Config, profile: str, *, force: bool = False, client: Any = None) -> dict[str, Any]:
    own_client = client is None
    active = client or Neo4jClient.from_config(cfg)
    try:
        if not active.is_ready():
            raise GraphStoreError("cannot dump: Neo4j is not reachable")
        marker = _marker_or_infer(cfg, active)
        require_loaded_profile(marker, profile, "dump")
        return _dump_loaded(cfg, active, marker or {}, force=force)
    finally:
        if own_client:
            active.close()


def restore_profile(
    cfg: Config,
    profile: str,
    *,
    client: Any = None,
    preserve_current: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    target = dump_path(cfg, profile)
    metadata = read_dump_metadata(cfg, profile)
    if not target.is_file() or not metadata:
        raise GraphStoreError(f"no complete dump found for profile {profile!r} at {target}")
    if metadata.get("profile") != profile:
        raise GraphStoreError(f"dump metadata profile mismatch for {target}")
    own_client = client is None
    active = client or Neo4jClient.from_config(cfg)
    try:
        if not active.is_ready():
            raise GraphStoreError("cannot restore: Neo4j is not reachable")
        marker = _marker_or_infer(cfg, active)
        if preserve_current and marker and not _marker_is_dumped(cfg, marker):
            _dump_loaded(cfg, active, marker, force=force)
        container_id, volume = _container_info()
        work_dir = target.parent / f".restore-{profile}-{uuid.uuid4().hex}"
        work_dir.mkdir()
        started = time.perf_counter()
        try:
            shutil.copy2(target, work_dir / "neo4j.dump")
            logger.info("stopping %s for offline restore of %s", CONTAINER_NAME, profile)
            _stop_container()
            _admin_load(volume, work_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            _start_container(container_id)
            _wait_for_graph(cfg)
        restored = read_graph_marker(active)
        if not restored or restored.get("profile") != profile:
            raise GraphStoreError(f"restored graph marker does not identify profile {profile!r}")
        logger.info("restored profile=%s elapsed=%.1fs", profile, time.perf_counter() - started)
        return restored
    finally:
        if own_client:
            active.close()


def build_profile(cfg: Config, client: Any) -> dict[str, Any]:
    from entdec.graph.loader import load_all, verify
    from entdec.graph.schema import apply_schema, reset_database

    logger.info("building graph for profile=%s from processed artefacts", cfg.profile.name)
    reset_database(client)
    dim = graph_embed_dim(cfg)
    apply_schema(client, dim=dim)
    load_all(cfg, client, with_similar=True)
    marker = write_graph_marker(client, cfg)
    result = verify(client)
    if not result["checks"]["ok"]:
        raise GraphStoreError(f"graph verification failed after build: {result['checks']}")
    return marker


def use_profile(cfg: Config, *, client: Any = None, force: bool = False) -> dict[str, Any]:
    """Keep the requested profile loaded, restoring a matching dump or rebuilding."""
    started = time.perf_counter()
    target_profile = cfg.profile.name
    target_hash = graph_artifact_hash(cfg)
    metadata = read_dump_metadata(cfg, target_profile)
    target_dump = dump_path(cfg, target_profile)
    if metadata and target_dump.is_file() and metadata.get("artefact_hash") != target_hash:
        logger.info("profile %s dump is stale for current processed artefacts", target_profile)
        metadata = None

    own_client = client is None
    active = client or Neo4jClient.from_config(cfg)
    try:
        if not active.is_ready() and not active.wait_until_ready(timeout_s=30):
            raise GraphStoreError(f"Neo4j is not reachable at {cfg.env.neo4j_uri}")
        loaded = _marker_or_infer(cfg, active)
        action = choose_graph_action(loaded, target_profile, target_hash, metadata)
        if action == "noop":
            logger.info(
                "graph use: profile=%s and artefact hash already match; no action (%.1fs)",
                target_profile,
                time.perf_counter() - started,
            )
            return loaded or {}

        if loaded and not _marker_is_dumped(cfg, loaded):
            logger.info("preserving currently loaded profile=%s before switching", loaded["profile"])
            _dump_loaded(cfg, active, loaded, force=force)

        if action == "restore":
            logger.info("graph use: restoring matching dump for %s", target_profile)
            restored = restore_profile(
                cfg, target_profile, client=active, preserve_current=False, force=force
            )
            logger.info(
                "graph use completed: action=restore profile=%s elapsed=%.1fs",
                target_profile,
                time.perf_counter() - started,
            )
            return restored

        logger.info("graph use: no matching dump; building profile=%s", target_profile)
        build_started = time.perf_counter()
        marker = build_profile(cfg, active)
        logger.info(
            "graph build completed: profile=%s elapsed=%.1fs",
            target_profile,
            time.perf_counter() - build_started,
        )
        _dump_loaded(cfg, active, marker, force=force)
        logger.info(
            "graph use completed: action=build-and-dump profile=%s elapsed=%.1fs",
            target_profile,
            time.perf_counter() - started,
        )
        return marker
    finally:
        if own_client:
            active.close()


def graph_status(cfg: Config, *, client: Any = None) -> dict[str, Any]:
    own_client = client is None
    active = client or Neo4jClient.from_config(cfg)
    try:
        reachable = active.is_ready()
        marker = None
        counts = None
        if reachable:
            marker = _marker_or_infer(cfg, active)
            counts = graph_counts(active)
    finally:
        if own_client:
            active.close()
    dumps: list[dict[str, Any]] = []
    folder = graph_dumps_dir(cfg)
    for path in sorted(folder.glob("*.dump")):
        metadata = read_dump_metadata(cfg, path.stem)
        dumps.append(
            {
                "profile": path.stem,
                "size_bytes": path.stat().st_size,
                "created_at": metadata.get("created_at") if metadata else None,
            }
        )
    result = {
        "reachable": reachable,
        "loaded_profile": marker.get("profile") if marker else None,
        "counts": counts,
        "dumps": dumps,
    }
    print(f"Neo4j reachable: {'yes' if reachable else 'no'}")
    print(f"Loaded profile: {result['loaded_profile'] or 'unknown'}")
    if counts:
        print(f"Nodes: {counts['node_count']:,}; relationships: {counts['relationship_count']:,}")
    print("Available dumps:")
    if dumps:
        for item in dumps:
            size_gib = item["size_bytes"] / (1024**3)
            print(f"  {item['profile']}: {size_gib:.2f} GiB, {item['created_at'] or 'date unknown'}")
    else:
        print("  none")
    return result


def ensure_graph(
    cfg: Config, *, no_graph: bool = False, force: bool = False
) -> bool:
    """Ensure this profile is loaded, unless graph access is explicitly disabled."""
    if no_graph or cfg.env.entdec_no_graph:
        logger.info("Neo4j disabled; using local Parquet retrieval")
        return False
    try:
        use_profile(cfg, force=force)
    except GraphStoreError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise GraphStoreError(f"unable to switch graph profile: {exc}") from exc
    return True


__all__ = [
    "CONTAINER_NAME",
    "GraphStoreError",
    "build_profile",
    "choose_graph_action",
    "current_marker",
    "dump_profile",
    "ensure_graph",
    "graph_artifact_hash",
    "graph_counts",
    "graph_dumps_dir",
    "graph_status",
    "preserve_current_graph",
    "read_dump_metadata",
    "read_graph_marker",
    "require_loaded_profile",
    "restore_profile",
    "use_profile",
    "write_dump_metadata",
    "write_graph_marker",
]
