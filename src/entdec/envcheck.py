"""Environment inspection (``python -m entdec envcheck``).

Reports the Python runtime, installed third-party packages, external tooling
(Docker, Ollama), the raw dataset files and the active profile. This is a
diagnostic command: it never mutates anything.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from entdec.config import Config
from entdec.logging_setup import get_logger

logger = get_logger(__name__)

_REQUIRED_PACKAGES = (
    "pandas",
    "numpy",
    "pyarrow",
    "scipy",
    "scikit-learn",
    "neo4j",
    "httpx",
    "pydantic",
    "pydantic-settings",
    "rapidfuzz",
    "fastapi",
    "uvicorn",
    "streamlit",
    "pytest",
    "ruff",
    "python-dotenv",
    "PyYAML",
)

_OPTIONAL_PACKAGES = ("implicit",)

# Interpreter versions this project was verified against (venv is built with 3.12).
_SUPPORTED_PYTHON = ((3, 10), (3, 11), (3, 12), (3, 13))

# Expected raw files: (domain, filename, approximate row count from inspection).
_EXPECTED_FILES = (
    ("movie", "movies.csv", 62_424),
    ("movie", "ratings.csv", 25_000_096),
    ("movie", "tags.csv", 1_093_361),
    ("movie", "genome-scores.csv", 15_584_449),
    ("movie", "genome-tags.csv", 1_129),
    ("movie", "links.csv", 62_424),
    ("game", "games.csv", 50_873),
    ("game", "games_metadata.json", 50_872),
    ("game", "users.csv", 14_306_065),
    ("game", "recommendations.csv", 41_154_795),
)


@dataclass
class EnvReport:
    """Structured result of the environment check."""

    python_version: str = ""
    packages: dict[str, str] = field(default_factory=dict)
    optional_packages: dict[str, str] = field(default_factory=dict)
    tools: dict[str, str] = field(default_factory=dict)
    files: dict[str, dict[str, object]] = field(default_factory=dict)
    ollama_models: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing critical is missing."""
        return not self.problems


def _package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _tool_version(command: list[str]) -> str | None:
    if shutil.which(command[0]) is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - fixed, local, read-only commands
            command,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - env dependent
        return None
    output = (result.stdout or result.stderr).strip().splitlines()
    return output[0].strip() if output else ("" if result.returncode == 0 else None)


def list_ollama_models(host: str, timeout: float = 5.0) -> list[str]:
    """Return the model names known to Ollama, or an empty list when unreachable."""
    try:
        response = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return []
    models = payload.get("models") or []
    return sorted(str(model.get("name", "")) for model in models if model.get("name"))


def _file_info(path: Path, expected_rows: int) -> dict[str, object]:
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    return {
        "path": str(path),
        "exists": exists,
        "size_mb": round(size / (1024 * 1024), 1) if exists else 0.0,
        "expected_rows": expected_rows,
    }


def run_env_check(cfg: Config) -> EnvReport:
    """Collect and log the environment report."""
    report = EnvReport(python_version=platform.python_version())

    if sys.version_info[:2] not in _SUPPORTED_PYTHON:
        supported = ", ".join(".".join(map(str, item)) for item in _SUPPORTED_PYTHON)
        report.problems.append(f"Python {supported} required, found {report.python_version}")

    for package in _REQUIRED_PACKAGES:
        version = _package_version(package)
        if version is None:
            report.problems.append(f"missing package: {package}")
        else:
            report.packages[package] = version

    for package in _OPTIONAL_PACKAGES:
        version = _package_version(package)
        if version is not None:
            report.optional_packages[package] = version

    report.tools["git"] = _tool_version(["git", "--version"]) or "not found"
    report.tools["docker"] = _tool_version(["docker", "--version"]) or "not found"
    report.tools["docker compose"] = _tool_version(["docker", "compose", "version"]) or "not found"
    report.tools["ollama"] = _tool_version(["ollama", "--version"]) or "not found"

    report.ollama_models = list_ollama_models(cfg.env.ollama_host)
    for model in (cfg.env.ollama_embed_model, cfg.env.ollama_gen_model):
        if report.ollama_models and model not in report.ollama_models:
            report.problems.append(f"ollama model not pulled: {model}")

    for domain, filename, rows in _EXPECTED_FILES:
        path = cfg.raw_path(domain, filename)
        info = _file_info(path, rows)
        report.files[f"{domain}/{filename}"] = info
        if not info["exists"]:
            report.problems.append(f"missing data file: {path}")

    logger.info("Python: %s", report.python_version)
    logger.info(
        "Profile: %s (max_items=%d, max_users=%d)",
        cfg.profile.name,
        cfg.profile.max_items,
        cfg.profile.max_users,
    )
    logger.info("Packages:")
    for name, version in report.packages.items():
        logger.info("  %-18s %s", name, version)
    if report.optional_packages:
        logger.info("Optional packages: %s", report.optional_packages)
    logger.info("Tools:")
    for name, version in report.tools.items():
        logger.info("  %-14s %s", name, version)
    logger.info("Ollama models: %s", ", ".join(report.ollama_models) or "<unreachable>")
    logger.info("Data files:")
    for name, info in report.files.items():
        status = "ok" if info["exists"] else "MISSING"
        logger.info(
            "  %-32s %-8s %8.1f MB  (~%s rows)",
            name,
            status,
            info["size_mb"],
            f"{info['expected_rows']:,}",
        )

    if report.problems:
        logger.warning("Problems found:")
        for problem in report.problems:
            logger.warning("  - %s", problem)
    else:
        logger.info("Environment check passed.")
    return report


__all__ = ["EnvReport", "list_ollama_models", "run_env_check"]
