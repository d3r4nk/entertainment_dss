"""Configuration loading.

Two sources are merged:

* ``.env``        - secrets and machine-specific values, loaded with
                    :mod:`pydantic_settings` (see :class:`EnvSettings`).
* ``config/settings.yaml`` - non-secret project settings (profiles,
                    thresholds, weights). Loaded as raw YAML and exposed
                    through typed accessor models.

The public entry point is :func:`load_config`, which returns a :class:`Config`
object holding resolved absolute paths, the environment settings, the raw YAML
document and the active sampling profile.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH: Path = PROJECT_ROOT / "config" / "settings.yaml"

ProfileName = Literal["tiny", "default"]


class EnvSettings(BaseSettings):
    """Secrets / machine-specific settings coming from ``.env`` or the process env."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "entdecdevpass"
    neo4j_database: str = "neo4j"
    entdec_no_graph: bool = False

    # Ollama
    ollama_host: str = "http://localhost:11434"
    ollama_embed_model: str = "qwen3-embedding:4b"
    ollama_gen_model: str = "qwen3:4b"

    # run profile
    entdec_profile: ProfileName = "tiny"

    # path overrides
    entdec_data_dir: Path | None = None
    entdec_processed_dir: Path | None = None
    entdec_reports_dir: Path | None = None
    entdec_cache_dir: Path | None = None


class ProfileConfig(BaseModel):
    """One sampling profile from ``settings.yaml``."""

    name: str
    max_items: int = Field(gt=0)
    max_users: int = Field(gt=0)
    min_movie_ratings: int = Field(ge=0)
    min_game_reviews: int = Field(ge=0)
    min_interactions_per_user: int = Field(ge=1)
    min_positive_per_user: int = Field(ge=3)


class Paths(BaseModel):
    """Resolved absolute project paths."""

    root: Path
    data: Path
    movie_data: Path
    game_data: Path
    processed_root: Path
    processed: Path
    reports_root: Path
    reports: Path
    cache: Path
    fixtures: Path

    @field_validator(
        "root",
        "data",
        "processed_root",
        "processed",
        "reports_root",
        "reports",
        "cache",
        mode="after",
    )
    @classmethod
    def _absolute(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    def ensure(self) -> None:
        """Create the writable directories if they are missing."""
        for path in (self.processed_root, self.processed, self.reports, self.cache):
            path.mkdir(parents=True, exist_ok=True)


class Config:
    """Aggregated, resolved configuration for a single profile."""

    def __init__(
        self,
        env: EnvSettings,
        yaml_data: dict[str, Any],
        profile: ProfileConfig,
        paths: Paths,
        settings_path: Path = DEFAULT_SETTINGS_PATH,
    ) -> None:
        self.env = env
        self.yaml = yaml_data
        self.profile = profile
        self.paths = paths
        self.settings_path = settings_path

    # --- generic dotted access -------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a nested YAML value using a dotted key, e.g. ``nlp.embed_dim``."""
        node: Any = self.yaml
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level YAML section as a dict (empty when absent)."""
        value = self.yaml.get(name)
        return dict(value) if isinstance(value, dict) else {}

    # --- properties -----------------------------------------------------------
    @property
    def seed(self) -> int:
        return int(self.get("project.seed", 42))

    @property
    def tuned_weights_path(self) -> Path:
        """Path of the generated per-profile recommender and chat weights."""
        return PROJECT_ROOT / "config" / "tuned_weights.yaml"

    @property
    def profiles(self) -> dict[str, ProfileConfig]:
        raw = self.section("profiles")
        return {name: ProfileConfig(name=name, **values) for name, values in raw.items()}

    @property
    def hybrid_weights(self) -> dict[str, float]:
        """Flat weight map for the profile, skipping per-domain sub-dicts."""
        weights = self.section("hybrid_weights").get(self.profile.name, {})
        return {
            str(k): float(v)
            for k, v in weights.items()
            if isinstance(v, (int, float))
        }

    @property
    def top_k_report(self) -> list[int]:
        return [int(k) for k in self.get("recsys.top_k_report", [10, 20])]

    def path(self, *parts: str) -> Path:
        """Resolve a path below the project root."""
        return self.paths.root.joinpath(*parts)

    def processed_path(self, name: str) -> Path:
        """Path of a file inside the active profile's processed directory."""
        return self.paths.processed / name

    def report_path(self, name: str) -> Path:
        """Path of a file inside the active profile's reports directory."""
        return self.paths.reports / name

    @property
    def theme_map_path(self) -> Path:
        """Shared editable theme map, independent of the active profile."""
        return self.paths.processed_root / "theme_map.csv"

    def raw_path(self, domain: str, name: str) -> Path:
        """Path of a raw dataset file, e.g. ``raw_path("movie", "ratings.csv")``."""
        base = self.paths.movie_data if domain == "movie" else self.paths.game_data
        direct = base / name
        if direct.exists():
            return direct
        # MovieLens archives are sometimes nested one level (ml-25m/...).
        for candidate in sorted(base.glob(f"*/{name}")):
            return candidate
        return direct

    # --- mutation (used by the weight tuner) ----------------------------------
    def set_hybrid_weights(self, weights: dict[str, float], persist: bool = True) -> None:
        """Update the flat hybrid weights in memory (and optionally on disk).

        Per-domain tuned weights are stored in ``config/tuned_weights.yaml`` by
        ``entdec.recsys.evaluate.save_hybrid_weights`` instead, so that the
        hand-edited comments in ``settings.yaml`` survive.
        """
        section = self.yaml.setdefault("hybrid_weights", {})
        section[self.profile.name] = {k: round(float(v), 4) for k, v in weights.items()}
        if persist:
            self.save_yaml()

    def save_yaml(self) -> None:
        """Write the in-memory YAML document back (used by tests / fallback)."""
        self.settings_path.write_text(
            yaml.safe_dump(self.yaml, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    def load_tuned_weights(self, domain: str | None = None) -> dict[str, float] | dict[str, dict[str, float]]:
        """Read the generated ``config/tuned_weights.yaml``.

        Returns the per-domain weight block for ``domain``
        (``{model: weight}``) or, without ``domain``, the whole block for
        the active profile (``{domain: {model: weight}}``). Empty dict when the
        file does not cover this profile.
        """
        data = load_yaml(self.tuned_weights_path)
        node = data.get(self.profile.name, {})
        if not isinstance(node, dict):
            return {}
        if domain is None:
            return {
                str(d): {str(m): float(w) for m, w in weights.items()}
                for d, weights in node.items()
                if d != "chat" and isinstance(weights, dict)
            }
        weights = node.get(domain, {})
        return {str(m): float(w) for m, w in weights.items()} if isinstance(weights, dict) else {}

    def load_tuned_chat_weights(self, domain: str) -> dict[str, float]:
        """Read validation-tuned, anonymous-chat model weights for one domain."""
        data = load_yaml(self.tuned_weights_path)
        profile = data.get(self.profile.name, {})
        chat = profile.get("chat", {}) if isinstance(profile, dict) else {}
        weights = chat.get(domain, {}) if isinstance(chat, dict) else {}
        return {str(name): float(value) for name, value in weights.items()} if isinstance(weights, dict) else {}

    def copy(self) -> Config:
        """Deep copy of this config (safe for mutation)."""
        return Config(
            env=self.env.model_copy(deep=True),
            yaml_data=copy.deepcopy(self.yaml),
            profile=self.profile.model_copy(deep=True),
            paths=self.paths.model_copy(deep=True),
            settings_path=self.settings_path,
        )

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"Config(profile={self.profile.name!r}, root={self.paths.root!s})"


def load_yaml(path: Path | None = None) -> dict[str, Any]:
    """Read a YAML document, returning an empty dict when the file is missing."""
    target = path or DEFAULT_SETTINGS_PATH
    if not target.exists():
        return {}
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def build_paths(env: EnvSettings, profile: str | None = None) -> Paths:
    """Resolve project paths, isolating processed data and reports by profile.

    Environment directory overrides specify the shared parent directory;
    profile-specific output is stored in a child directory.
    """
    profile_name = profile or env.entdec_profile
    data = env.entdec_data_dir or PROJECT_ROOT / "data"
    processed_root = env.entdec_processed_dir or data / "processed"
    reports_root = env.entdec_reports_dir or PROJECT_ROOT / "reports"
    return Paths(
        root=PROJECT_ROOT,
        data=data,
        movie_data=data / "movie_data",
        game_data=data / "game_data",
        processed_root=processed_root,
        processed=processed_root / profile_name,
        reports_root=reports_root,
        reports=reports_root / profile_name,
        cache=env.entdec_cache_dir or PROJECT_ROOT / ".cache",
        fixtures=PROJECT_ROOT / "tests" / "fixtures",
    )


def load_config(
    profile: str | None = None,
    settings_path: Path | None = None,
) -> Config:
    """Build the runtime configuration.

    Args:
        profile: ``tiny`` or ``default``. Defaults to ``ENTDEC_PROFILE``.
        settings_path: Alternate ``settings.yaml`` location (used by tests).

    Returns:
        A resolved :class:`Config`.
    """
    settings_path = settings_path or DEFAULT_SETTINGS_PATH
    env = EnvSettings()
    yaml_data = load_yaml(settings_path)
    name = profile or env.entdec_profile or "tiny"
    raw_profiles = yaml_data.get("profiles", {})
    if name not in raw_profiles:
        raise KeyError(f"Unknown profile {name!r}; available: {sorted(raw_profiles) or 'none'}")
    profile_cfg = ProfileConfig(name=name, **raw_profiles[name])
    return Config(
        env=env,
        yaml_data=yaml_data,
        profile=profile_cfg,
        paths=build_paths(env, name),
        settings_path=settings_path,
    )


@lru_cache(maxsize=8)
def _cached_config(profile: str) -> Config:
    return load_config(profile)


def get_config(profile: str | None = None, use_cache: bool = True) -> Config:
    """Return a cached configuration for ``profile`` (falling back to the env default)."""
    resolved = profile or EnvSettings().entdec_profile or "tiny"
    return _cached_config(resolved) if use_cache else load_config(resolved)


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_SETTINGS_PATH",
    "Config",
    "EnvSettings",
    "Paths",
    "ProfileConfig",
    "ProfileName",
    "build_paths",
    "get_config",
    "load_config",
    "load_yaml",
]
