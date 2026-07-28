"""Tests for the filesystem layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc.paths import (
    ARC_HOME_ENV,
    arc_home,
    audit_dir,
    config_dir,
    ensure_runtime_dirs,
    hardware_file,
    log_dir,
    models_dir,
    project_root,
    run_dir,
)


def test_arc_home_follows_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "elsewhere"
    monkeypatch.setenv(ARC_HOME_ENV, str(target))
    assert arc_home() == target.resolve()


def test_arc_home_read_per_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The env var must not be cached at import time, or fixtures could not redirect it."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    monkeypatch.setenv(ARC_HOME_ENV, str(first))
    assert arc_home() == first.resolve()
    monkeypatch.setenv(ARC_HOME_ENV, str(second))
    assert arc_home() == second.resolve()


def test_derived_paths_live_under_home(arc_home_tmp: Path) -> None:
    for path in (audit_dir(), log_dir(), run_dir(), models_dir(), hardware_file()):
        assert path.is_relative_to(arc_home_tmp)


def test_ensure_runtime_dirs_is_idempotent(arc_home_tmp: Path) -> None:
    ensure_runtime_dirs()
    ensure_runtime_dirs()  # Must not raise on the second call.
    for path in (audit_dir(), log_dir(), run_dir(), models_dir()):
        assert path.is_dir()


def test_config_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "cfg"
    monkeypatch.setenv("ARC_CONFIG_DIR", str(target))
    assert config_dir() == target.resolve()


def test_project_root_is_independent_of_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Derived from __file__, so `arc` behaves the same wherever it is invoked."""
    before = project_root()
    monkeypatch.chdir(tmp_path)
    assert project_root() == before
    assert (before / "config").is_dir()
