"""Shared fixtures.

The single most important thing here is that no test may touch the real ``~/.arc``.
``arc/paths.py`` reads ``ARC_HOME`` on every call rather than caching it at import
time specifically so this redirection works, and the fixture is autouse so a new test
file cannot forget to ask for it.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from arc.paths import ARC_HOME_ENV


@pytest.fixture(autouse=True)
def arc_home_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point the runtime data root at a temp directory for every test."""
    home = tmp_path / "arc-home"
    home.mkdir()
    monkeypatch.setenv(ARC_HOME_ENV, str(home))
    yield home


@pytest.fixture(autouse=True)
def no_ambient_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip inherited ``ARC_*`` variables.

    Without this, running the suite in a shell that happens to export ``ARC_LOGGING__LEVEL``
    would change the merged config under the tests' feet and produce failures that
    reproduce on one machine and not another.
    """
    import os

    for name in list(os.environ):
        if name.startswith("ARC_") and name != ARC_HOME_ENV:
            monkeypatch.delenv(name, raising=False)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """An empty directory to write test YAML into."""
    directory = tmp_path / "config"
    directory.mkdir()
    return directory
