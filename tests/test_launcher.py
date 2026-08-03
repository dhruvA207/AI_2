"""Tests for the `ARC` terminal launcher.

The script itself is shell, so what is checked here is the contract around it: that it
exists, that it is runnable, and that bare `ARC` opens the app rather than printing
argparse usage — the failure that would make typing `ARC` feel broken.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parent.parent / "bin" / "ARC"


def test_the_launcher_exists_and_is_executable() -> None:
    assert LAUNCHER.is_file(), "bin/ARC is missing"
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR, "bin/ARC is not executable"


def test_bare_arc_opens_the_app() -> None:
    """No arguments must mean "launch", not "show me the usage message"."""
    body = LAUNCHER.read_text(encoding="utf-8")
    assert "-m arc ui" in body


def test_arguments_pass_straight_through() -> None:
    """`ARC doctor` should be `arc doctor`, so this is not a second, lesser CLI."""
    assert '-m arc "$@"' in LAUNCHER.read_text(encoding="utf-8")


def test_it_resolves_its_own_repo_through_a_symlink() -> None:
    """It is invoked via ~/.local/bin/ARC, so a relative path would find nothing."""
    body = LAUNCHER.read_text(encoding="utf-8")
    assert "BASH_SOURCE" in body
    assert "readlink" in body


@pytest.mark.skipif(sys.platform != "darwin", reason="the launcher is macOS-only")
def test_it_runs_and_reports_the_version(tmp_path: Path) -> None:
    """End to end, from an unrelated directory: the self-location has to hold."""
    result = subprocess.run(
        [str(LAUNCHER), "version"],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        timeout=120,
        env={**os.environ, "ARC_HOME": str(tmp_path / "home")},
    )
    assert result.returncode == 0, result.stderr
    assert "arc" in result.stdout.lower()
