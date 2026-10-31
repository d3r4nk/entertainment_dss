"""Neo4j connection handling plus batched write helpers.

All writes go through ``UNWIND $rows AS row`` batches (5k-10k rows) so the loader
stays idempotent and memory friendly. Credentials come from ``.env``; nothing is
ever logged that contains the password.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError as DriverNeo4jError
from neo4j.exceptions import ServiceUnavailable, SessionExpired

from entdec.config import PROJECT_ROOT, Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

DEFAULT_BATCH_SIZE = 5000
EMBEDDING_BATCH_SIZE = 512
COMPOSE_FILE = PROJECT_ROOT / "docker-compose.yml"


class Neo4jError(RuntimeError):
    """Raised when the database cannot be reached or a query fails."""


def _chunks(rows: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


class Neo4jClient:
    """Small wrapper around the official driver."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "entdecdevpass",
        database: str = "neo4j",
    ) -> None:
        self.uri = uri
        self.user = user
        self.database = database
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    # --- lifecycle ------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Config) -> Neo4jClient:
        """Build a client from the application config (``.env``)."""
        return cls(
            uri=cfg.env.neo4j_uri,
            user=cfg.env.neo4j_user,
            password=cfg.env.neo4j_password,
            database=cfg.env.neo4j_database,
        )

    def close(self) -> None:
        """Close the driver and its connection pool."""
        self._driver.close()

    def __enter__(self) -> Neo4jClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- reading / writing ----------------------------------------------------
    def run(self, query: str, **params: Any) -> list[dict[str, Any]]:
        """Run one statement and return all records as dicts."""
        try:
            with self._driver.session(database=self.database) as session:
                result = session.run(query, **params)
                records = [record.data() for record in result]
                result.consume()
                return records
        except (ServiceUnavailable, SessionExpired, DriverNeo4jError) as exc:
            raise Neo4jError(f"query failed: {exc}") from exc

    def run_batches(
        self,
        query: str,
        rows: Sequence[dict[str, Any]],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        label: str = "",
    ) -> int:
        """Run a ``UNWIND $rows`` statement over ``rows`` in batches.

        Returns:
            Number of rows sent.
        """
        if not rows:
            return 0
        sent = 0
        started = time.perf_counter()
        batches = (len(rows) + batch_size - 1) // batch_size
        for batch_number, batch in enumerate(_chunks(rows, batch_size), start=1):
            self.run(query, rows=list(batch))
            sent += len(batch)
            if batches > 20 and (batch_number % 20 == 0 or batch_number == batches):
                logger.info(
                    "loaded %d/%d %s rows (%d/%d batches, %.1fs)",
                    sent,
                    len(rows),
                    label or "graph",
                    batch_number,
                    batches,
                    time.perf_counter() - started,
                )
        logger.info(
            "loaded %d %s rows in %.1fs (%d rows/batch)",
            sent,
            label or "graph",
            time.perf_counter() - started,
            batch_size,
        )
        return sent

    # --- readiness ------------------------------------------------------------
    def is_ready(self) -> bool:
        """True when the database answers a trivial query."""
        try:
            self.run("RETURN 1 AS ok")
            return True
        except Neo4jError:
            return False

    def wait_until_ready(self, timeout_s: float = 120.0, interval_s: float = 2.0) -> bool:
        """Poll until the database answers or ``timeout_s`` elapses."""
        deadline = time.perf_counter() + timeout_s
        attempt = 0
        while time.perf_counter() < deadline:
            attempt += 1
            if self.is_ready():
                logger.info("neo4j is ready (%s, attempt %d)", self.uri, attempt)
                return True
            time.sleep(interval_s)
        logger.error("neo4j not ready after %.0fs (%s)", timeout_s, self.uri)
        return False


# --------------------------------------------------------------------------- #
# docker compose helpers
# --------------------------------------------------------------------------- #
def _compose(args: Sequence[str], timeout_s: float = 180.0) -> subprocess.CompletedProcess[str]:
    command = ["docker", "compose", "-f", str(COMPOSE_FILE), *args]
    logger.info("running: %s", " ".join(command))
    return subprocess.run(  # noqa: S603 - fixed, local, user requested
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def compose_up(wait_s: float = 180.0) -> bool:
    """Start Neo4j with docker compose and wait until Bolt answers."""
    result = _compose(["up", "-d"])
    if result.returncode != 0:
        logger.error("docker compose up failed: %s", (result.stderr or "").strip()[:500])
        return False
    return wait_for_neo4j(wait_s)


def compose_down(volumes: bool = False) -> bool:
    """Stop Neo4j (optionally removing the named volumes)."""
    args = ["down"] + (["-v"] if volumes else [])
    result = _compose(args)
    if result.returncode != 0:
        logger.error("docker compose down failed: %s", (result.stderr or "").strip()[:500])
        return False
    logger.info("neo4j stopped%s", " and volumes removed" if volumes else "")
    return True


def wait_for_neo4j(timeout_s: float = 180.0, uri: str | None = None) -> bool:
    """Wait for a Bolt endpoint to become reachable."""
    import os

    target = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    password = os.environ.get("NEO4J_PASSWORD", "entdecdevpass")
    user = os.environ.get("NEO4J_USER", "neo4j")
    client = Neo4jClient(target, user, password)
    try:
        return client.wait_until_ready(timeout_s=timeout_s)
    finally:
        client.close()


def compose_logs_path() -> Path:
    """Path of the compose file, for reporting."""
    return COMPOSE_FILE


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_BATCH_SIZE",
    "Neo4jClient",
    "Neo4jError",
    "compose_down",
    "compose_logs_path",
    "compose_up",
    "wait_for_neo4j",
]
