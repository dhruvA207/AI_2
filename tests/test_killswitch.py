"""Tests for the kill switch.

The registry is tested rather than the killing: a test that actually SIGKILLs process
trees would be killing processes on the machine running the suite. What matters here
is that the bookkeeping is correct, because ``kill_all`` acts on exactly what
``registered()`` reports.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from arc.audit.killswitch import KillSwitch, ProcessEntry
from arc.paths import run_dir


def dead_pid() -> int:
    """Return a PID that is guaranteed to have exited."""
    proc = subprocess.Popen(["/usr/bin/true"])
    proc.wait()
    return proc.pid


def write_pid_file(directory: Path, name: str, pid: int) -> Path:
    """Write a PID file by hand, since register() can only register ourselves."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{pid}.pid"
    path.write_text(
        json.dumps({"name": name, "pid": pid, "started_at": "2026-07-28T00:00:00+00:00"}),
        encoding="utf-8",
    )
    return path


def test_register_writes_a_pid_file(arc_home_tmp: Path) -> None:
    switch = KillSwitch()
    path = switch.register("agent")
    assert path.is_file()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["name"] == "agent"


def test_pid_in_filename_prevents_clobbering(arc_home_tmp: Path) -> None:
    """Two ARC processes (agent plus a training supervisor) must both be registered."""
    switch = KillSwitch()
    switch.register("agent")
    write_pid_file(run_dir(), "agent", 999001)
    assert len(switch.registered()) == 2


def test_unregister_removes_the_file(arc_home_tmp: Path) -> None:
    switch = KillSwitch()
    switch.register("agent")
    switch.unregister("agent")
    assert switch.registered() == []


def test_unregister_is_safe_when_already_gone(arc_home_tmp: Path) -> None:
    KillSwitch().unregister("never-registered")  # Must not raise.


def test_registered_on_missing_directory(arc_home_tmp: Path) -> None:
    assert KillSwitch().registered() == []


def test_registered_skips_unreadable_files(arc_home_tmp: Path) -> None:
    """A corrupt PID file must not stop the kill switch from finding the others."""
    switch = KillSwitch()
    switch.register("good")
    (run_dir() / "broken-1.pid").write_text("{ not json", encoding="utf-8")
    (run_dir() / "missing-keys-2.pid").write_text("{}", encoding="utf-8")

    entries = switch.registered()
    assert len(entries) == 1
    assert entries[0].name == "good"


def test_live_process_reports_alive(arc_home_tmp: Path) -> None:
    entry = ProcessEntry("self", os.getpid(), "now", Path("/tmp/x"))
    assert entry.alive is True


def test_dead_process_reports_not_alive(arc_home_tmp: Path) -> None:
    entry = ProcessEntry("gone", dead_pid(), "now", Path("/tmp/x"))
    assert entry.alive is False


def test_reap_stale_removes_dead_entries(arc_home_tmp: Path) -> None:
    switch = KillSwitch()
    switch.register("alive")
    write_pid_file(run_dir(), "dead", dead_pid())

    assert switch.reap_stale() == 1
    assert [e.name for e in switch.registered()] == ["alive"]


def test_reap_stale_keeps_live_entries(arc_home_tmp: Path) -> None:
    switch = KillSwitch()
    switch.register("alive")
    assert switch.reap_stale() == 0
    assert len(switch.registered()) == 1


def test_kill_all_skips_self_by_default(arc_home_tmp: Path) -> None:
    """Killing the process doing the killing would abort the job halfway through."""
    switch = KillSwitch()
    switch.register("me")
    assert switch.kill_all() == []
    assert os.getpid() == os.getpid()  # Still here.


def test_kill_all_reaps_dead_entries(arc_home_tmp: Path) -> None:
    switch = KillSwitch()
    write_pid_file(run_dir(), "dead", dead_pid())
    assert switch.kill_all() == []
    assert switch.registered() == []


def test_kill_all_on_empty_registry(arc_home_tmp: Path) -> None:
    assert KillSwitch().kill_all() == []


def test_arc_kill_entry_point_with_nothing_registered(
    arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from arc.audit.killswitch import main

    assert main([]) == 0
    assert "no ARC processes registered" in capsys.readouterr().out


def test_arc_kill_dry_run_lists_without_killing(
    arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    switch = KillSwitch()
    switch.register("agent")
    from arc.audit.killswitch import main

    assert main(["--dry-run"]) == 0
    assert "[dry-run]" in capsys.readouterr().out
    assert len(switch.registered()) == 1  # still there


def test_arc_kill_json_output(arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from arc.audit.killswitch import main

    assert main(["--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"killed": [], "reaped_stale": 0}


def test_arc_kill_reaps_stale_entries(
    arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_pid_file(run_dir(), "dead", dead_pid())
    from arc.audit.killswitch import main

    assert main([]) == 0
    assert "stale PID file" in capsys.readouterr().out


def test_arc_kill_needs_no_config(arc_home_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point: stopping ARC must not depend on ARC being healthy, so this
    path loads no YAML, probes no hardware, and writes no logs."""
    monkeypatch.setenv("ARC_CONFIG_DIR", "/definitely/not/here")
    from arc.audit.killswitch import main

    assert main([]) == 0
