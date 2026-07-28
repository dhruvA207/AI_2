"""Tests for configuration loading and merge precedence."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc.config import Config, _coerce, _deep_merge, _env_overrides
from arc.errors import ConfigError


def write(directory: Path, name: str, body: str) -> None:
    """Write a YAML file into a config directory."""
    (directory / name).write_text(body, encoding="utf-8")


def test_default_yaml_merges_at_root(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "runtime:\n  dry_run: true\n")
    config = Config.load(directory=config_dir, use_env=False)
    assert config.get("runtime.dry_run") is True


def test_other_files_merge_under_their_own_name(config_dir: Path) -> None:
    """policy.yaml must be reachable as policy.*, so a key's path is guessable."""
    write(config_dir, "policy.yaml", "allow_shell: true\n")
    config = Config.load(directory=config_dir, use_env=False)
    assert config.get("policy.allow_shell") is True


def test_local_override_beats_committed_config(config_dir: Path, arc_home_tmp: Path) -> None:
    write(config_dir, "default.yaml", "runtime:\n  session_name: committed\n")
    (arc_home_tmp / "config.yaml").write_text("runtime:\n  session_name: local\n", encoding="utf-8")
    config = Config.load(directory=config_dir, use_env=False)
    assert config.get("runtime.session_name") == "local"


def test_env_beats_everything(
    config_dir: Path, arc_home_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(config_dir, "default.yaml", "runtime:\n  session_name: committed\n")
    (arc_home_tmp / "config.yaml").write_text("runtime:\n  session_name: local\n", encoding="utf-8")
    monkeypatch.setenv("ARC_RUNTIME__SESSION_NAME", "from-env")
    config = Config.load(directory=config_dir)
    assert config.get("runtime.session_name") == "from-env"


def test_env_double_underscore_nests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_AGENT__MAX_STEPS", "12")
    assert _env_overrides() == {"agent": {"max_steps": 12}}


def test_env_single_underscore_stays_in_key_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Single underscores are common inside key names and must survive."""
    monkeypatch.setenv("ARC_RUNTIME__DRY_RUN", "true")
    assert _env_overrides() == {"runtime": {"dry_run": True}}


def test_path_steering_vars_are_not_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARC_HOME", "/tmp/nope")
    monkeypatch.setenv("ARC_CONFIG_DIR", "/tmp/nope")
    assert _env_overrides() == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), ("12", 12), ("1.5", 1.5), ("hello", "hello")],
)
def test_coerce_produces_real_types(raw: str, expected: object) -> None:
    """Without this, every override arrives as a string downstream code must convert."""
    assert _coerce(raw) == expected


def test_deep_merge_recurses_into_dicts() -> None:
    base = {"a": {"x": 1, "y": 2}}
    overlay = {"a": {"y": 3}}
    assert _deep_merge(base, overlay) == {"a": {"x": 1, "y": 3}}


def test_deep_merge_replaces_lists_rather_than_appending() -> None:
    """Appending would make it impossible to remove a default entry."""
    assert _deep_merge({"a": [1, 2]}, {"a": [3]}) == {"a": [3]}


def test_deep_merge_does_not_mutate_base() -> None:
    base = {"a": {"x": 1}}
    _deep_merge(base, {"a": {"x": 2}})
    assert base == {"a": {"x": 1}}


def test_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config directory not found"):
        Config.load(directory=tmp_path / "absent")


def test_non_mapping_yaml_raises(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "- just\n- a\n- list\n")
    with pytest.raises(ConfigError, match="must contain a mapping"):
        Config.load(directory=config_dir, use_env=False)


def test_invalid_yaml_raises(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "a: [unclosed\n")
    with pytest.raises(ConfigError, match="invalid YAML"):
        Config.load(directory=config_dir, use_env=False)


def test_empty_file_is_tolerated(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "")
    assert Config.load(directory=config_dir, use_env=False).as_dict() == {}


def test_get_returns_default_for_missing_path(config_dir: Path) -> None:
    config = Config.load(directory=config_dir, use_env=False)
    assert config.get("nothing.here", "fallback") == "fallback"


def test_require_raises_on_missing(config_dir: Path) -> None:
    config = Config.load(directory=config_dir, use_env=False)
    with pytest.raises(ConfigError, match="required config key missing"):
        config.require("nothing.here")


def test_typed_rejects_bool_for_int(config_dir: Path) -> None:
    """bool subclasses int, so an isinstance check alone would let `true` pass as 1."""
    write(config_dir, "default.yaml", "n: true\n")
    config = Config.load(directory=config_dir, use_env=False)
    with pytest.raises(ConfigError, match="got bool"):
        config.typed("n", int, 0)


def test_typed_accepts_bool_for_bool(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "flag: true\n")
    config = Config.load(directory=config_dir, use_env=False)
    assert config.typed("flag", bool, False) is True


def test_typed_reports_the_offending_key(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "n: not-a-number\n")
    config = Config.load(directory=config_dir, use_env=False)
    with pytest.raises(ConfigError, match="config key n expected int"):
        config.typed("n", int, 0)


def test_section_rejects_non_mapping(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "thing: 5\n")
    config = Config.load(directory=config_dir, use_env=False)
    with pytest.raises(ConfigError, match="is not a mapping"):
        config.section("thing")


def test_contains(config_dir: Path) -> None:
    write(config_dir, "default.yaml", "a:\n  b: 1\n")
    config = Config.load(directory=config_dir, use_env=False)
    assert "a.b" in config
    assert "a.c" not in config


def test_real_config_directory_loads() -> None:
    """The committed config/ must actually parse — a broken default is a broken install."""
    config = Config.load(use_env=False)
    assert config.get("runtime.dry_run") is False
    assert config.get("hardware.os_reserve_gb") == 4.0
