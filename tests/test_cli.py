"""Tests for the command-line entry point.

``main()`` returns an exit code rather than calling ``sys.exit``, which is what makes
these run in-process instead of needing a subprocess per assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc import __version__
from arc.__main__ import Check, _exit_code, _render, main
from arc.paths import audit_dir, hardware_file


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_version_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "version"]) == 0
    assert json.loads(capsys.readouterr().out) == {"version": __version__}


def test_no_subcommand_is_an_error() -> None:
    """argparse exits 2 for usage errors; the required=True on subparsers enforces it."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2


def test_doctor_succeeds_on_this_machine(
    arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "ARC doctor" in out
    assert "recommended model" in out
    assert "0 failed" in out


def test_doctor_json_is_machine_readable(
    arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", "doctor"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    names = {c["name"] for c in payload["checks"]}
    assert {"arc", "python", "platform", "config", "machine"} <= names


def test_doctor_reports_a_missing_config_dir(
    arc_home_tmp: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A broken config is diagnosed, not a reason to refuse to start."""
    assert main(["--config-dir", str(tmp_path / "absent"), "--json", "doctor"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    config = next(c for c in payload["checks"] if c["name"] == "config")
    assert config["status"] == "fail"


def test_doctor_writes_hardware_json(arc_home_tmp: Path) -> None:
    main(["doctor"])
    assert hardware_file().is_file()


def test_probe_writes_hardware_json(arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["probe"]) == 0
    assert hardware_file().is_file()
    assert "wrote" in capsys.readouterr().out


def test_probe_json_has_both_sections(
    arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--json", "probe"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"hardware", "recommendation"}
    assert payload["hardware"]["schema_version"] == 2


def test_probe_refresh_flag_accepted(arc_home_tmp: Path) -> None:
    assert main(["probe", "--refresh"]) == 0


def test_kill_with_nothing_registered(
    arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["kill"]) == 0
    assert "no ARC processes registered" in capsys.readouterr().out


def test_kill_dry_run_does_not_kill(arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--dry-run", "kill"]) == 0
    assert "[dry-run]" in capsys.readouterr().out


def test_kill_json(arc_home_tmp: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--json", "kill"]) == 0
    assert json.loads(capsys.readouterr().out) == {"killed": [], "reaped_stale": 0}


def test_invocations_are_audited(arc_home_tmp: Path) -> None:
    """Brief §0.3: the log answers "what did I run", not only "what did the agent do"."""
    main(["version"])
    logs = list(audit_dir().glob("*.jsonl"))
    assert len(logs) == 1
    records = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "cli.version"
    assert records[-1]["status"] == "ok"


def test_dry_run_is_recorded_as_such(arc_home_tmp: Path) -> None:
    main(["--dry-run", "kill"])
    logs = list(audit_dir().glob("*.jsonl"))
    records = [json.loads(line) for line in logs[0].read_text(encoding="utf-8").splitlines()]
    assert records[-1]["status"] == "dry_run"
    assert records[-1]["dry_run"] is True


def test_exit_code_ignores_warnings() -> None:
    """A fanless machine is a caveat, not a broken install."""
    checks = [Check("a", "ok", ""), Check("b", "warn", "")]
    assert _exit_code(checks) == 0


def test_exit_code_fails_on_any_failure() -> None:
    checks = [Check("a", "ok", ""), Check("b", "fail", "")]
    assert _exit_code(checks) == 1


def test_render_aligns_names(capsys: pytest.CaptureFixture[str]) -> None:
    _render([Check("short", "ok", "x"), Check("much-longer", "ok", "y")], as_json=False)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].index("x") == lines[1].index("y")
