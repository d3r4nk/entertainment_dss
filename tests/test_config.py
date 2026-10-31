"""Tests for :mod:`entdec.config`."""

from __future__ import annotations

from pathlib import Path

import pytest

from entdec.config import EnvSettings, build_paths, load_config, load_yaml


def test_loads_both_profiles() -> None:
    data = load_yaml()
    assert {"tiny", "default"} <= set(data["profiles"])


def test_default_profile_values() -> None:
    cfg = load_config("default")
    assert cfg.profile.max_items == 10_000
    assert cfg.profile.max_users == 20_000
    assert cfg.profile.min_movie_ratings == 500
    assert cfg.profile.min_game_reviews == 500
    assert cfg.profile.min_interactions_per_user == 20


def test_tiny_profile_is_smaller() -> None:
    tiny = load_config("tiny")
    default = load_config("default")
    assert tiny.profile.max_items < default.profile.max_items
    assert tiny.profile.max_users < default.profile.max_users
    assert tiny.paths.processed == tiny.paths.processed_root / "tiny"
    assert default.paths.processed == default.paths.processed_root / "default"
    assert tiny.paths.processed != default.paths.processed
    assert tiny.paths.reports == tiny.paths.reports_root / "tiny"
    assert default.paths.reports == default.paths.reports_root / "default"
    assert tiny.paths.reports != default.paths.reports
    assert tiny.theme_map_path == default.theme_map_path


def test_unknown_profile_raises() -> None:
    with pytest.raises(KeyError):
        load_config("nope")


def test_dotted_get_and_sections() -> None:
    cfg = load_config("tiny")
    assert cfg.get("nlp.embed_dim") == 1024
    assert cfg.get("does.not.exist", "fallback") == "fallback"
    assert "temperature" in cfg.section("llm")
    assert cfg.section("missing") == {}


def test_env_overrides_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTDEC_PROFILE", "default")
    env = EnvSettings()
    assert env.entdec_profile == "default"
    cfg = load_config()
    assert cfg.profile.name == "default"


def test_env_can_disable_neo4j(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENTDEC_NO_GRAPH", "true")
    assert EnvSettings().entdec_no_graph is True


def test_path_overrides_from_env(tmp_path: Path) -> None:
    env = EnvSettings(entdec_data_dir=tmp_path / "d", entdec_cache_dir=tmp_path / "c")
    paths = build_paths(env)
    assert paths.data == (tmp_path / "d").resolve()
    assert paths.movie_data == (tmp_path / "d" / "movie_data").resolve()
    assert paths.processed == (tmp_path / "d" / "processed" / "tiny").resolve()
    assert paths.reports == (load_config("tiny").paths.reports_root / "tiny").resolve()
    assert paths.cache == (tmp_path / "c").resolve()


def test_processed_and_reports_overrides_are_profile_parents(tmp_path: Path) -> None:
    env = EnvSettings(
        entdec_processed_dir=tmp_path / "processed",
        entdec_reports_dir=tmp_path / "reports",
    )
    tiny = build_paths(env, "tiny")
    default = build_paths(env, "default")
    assert tiny.processed == (tmp_path / "processed" / "tiny").resolve()
    assert default.processed == (tmp_path / "processed" / "default").resolve()
    assert tiny.reports == (tmp_path / "reports" / "tiny").resolve()
    assert default.reports == (tmp_path / "reports" / "default").resolve()
    assert tiny.processed_root == default.processed_root
    assert tiny.reports_root == default.reports_root


def test_raw_path_finds_nested_archive(tmp_path: Path) -> None:
    nested = tmp_path / "data" / "movie_data" / "ml-25m"
    nested.mkdir(parents=True)
    (nested / "movies.csv").write_text("movieId,title,genres\n", encoding="utf-8")
    env = EnvSettings(entdec_data_dir=tmp_path / "data")
    cfg = load_config("tiny")
    cfg.env = env
    cfg.paths = build_paths(env, cfg.profile.name)
    assert cfg.raw_path("movie", "movies.csv").parent.name == "ml-25m"


def test_hybrid_weights_per_profile() -> None:
    tiny = load_config("tiny").hybrid_weights
    assert set(tiny) >= {"popularity", "itemknn", "mf", "content", "kg"}
    assert abs(sum(tiny.values()) - 1.0) < 0.01


def test_set_hybrid_weights_persist_false(tmp_path: Path) -> None:
    cfg = load_config("tiny")
    before = tmp_path / "settings.yaml"
    before.write_text("profiles: {}\n", encoding="utf-8")
    cfg.set_hybrid_weights({"content": 2.0, "kg": 1.0}, persist=False)
    assert cfg.hybrid_weights == {"content": 2.0, "kg": 1.0}
    assert before.read_text(encoding="utf-8") == "profiles: {}\n"


def test_copy_is_independent() -> None:
    cfg = load_config("tiny")
    clone = cfg.copy()
    clone.set_hybrid_weights({"content": 5.0}, persist=False)
    assert cfg.hybrid_weights != clone.hybrid_weights


def test_top_k_report() -> None:
    assert load_config("tiny").top_k_report == [10, 20]


def test_llm_defaults_are_spec_compliant() -> None:
    cfg = load_config("tiny")
    assert cfg.get("llm.think") is False
    assert cfg.get("llm.num_ctx") == 4096
    assert cfg.get("llm.temperature") == 0.2
    assert cfg.get("llm.intent_temperature") == 0.0
    assert cfg.get("llm.default_language") == "vi"
