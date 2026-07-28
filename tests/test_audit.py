"""Tests for the append-only action log."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from arc.audit.logger import _TRUNCATION_SUFFIX, AuditLogger, _truncate
from arc.errors import AuditError
from arc.paths import audit_dir


def test_record_then_read_roundtrip(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    logger.record("tool.call", tool="shell", args={"cmd": "ls"})
    records = logger.read_recent()
    assert len(records) == 1
    assert records[0]["event"] == "tool.call"
    assert records[0]["args"] == {"cmd": "ls"}


def test_log_is_append_only(arc_home_tmp: Path) -> None:
    """Nothing may rewrite history: a log with silent gaps invites false confidence."""
    logger = AuditLogger()
    for i in range(5):
        logger.record(f"event.{i}")
    assert [r["event"] for r in logger.read_recent()] == [f"event.{i}" for i in range(5)]


def test_sequence_numbers_are_monotonic(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    for _ in range(10):
        logger.record("x")
    assert [r["seq"] for r in logger.read_recent()] == list(range(1, 11))


def test_sequence_numbers_unique_under_threads(arc_home_tmp: Path) -> None:
    """Phase 4 will call this from parallel tool dispatch, so the counter must lock."""
    logger = AuditLogger()

    def worker() -> None:
        for _ in range(20):
            logger.record("concurrent")

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    seqs = [r["seq"] for r in logger.read_recent(limit=1000)]
    assert len(seqs) == 100
    assert sorted(seqs) == list(range(1, 101))


def test_session_id_ties_records_together(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    logger.record("a")
    logger.record("b")
    ids = {r["session_id"] for r in logger.read_recent()}
    assert ids == {logger.session_id}


def test_every_line_is_valid_json(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    logger.record("weird", args={"path": Path("/tmp/x"), "n": 1})
    raw = logger.path_for_today().read_text(encoding="utf-8")
    for line in raw.splitlines():
        json.loads(line)


def test_unserializable_values_degrade_to_repr(arc_home_tmp: Path) -> None:
    """default=str, so a strange argument cannot lose the record it appears in."""
    logger = AuditLogger()
    logger.record("obj", args={"p": Path("/tmp/thing")})
    assert "/tmp/thing" in logger.read_recent()[0]["args"]["p"]


def test_truncate_clips_long_strings() -> None:
    out = _truncate("x" * 100, 10)
    assert out == "x" * 10 + _TRUNCATION_SUFFIX


def test_truncate_recurses_into_dicts() -> None:
    """One enormous field must not force its siblings to be discarded."""
    out = _truncate({"big": "x" * 100, "small": "ok"}, 10)
    assert out["small"] == "ok"
    assert out["big"].endswith(_TRUNCATION_SUFFIX)


def test_truncate_caps_long_lists() -> None:
    out = _truncate(list(range(500)), 10)
    assert len(out) == 101
    assert out[-1] == _TRUNCATION_SUFFIX


def test_truncate_leaves_short_values_alone() -> None:
    assert _truncate("short", 100) == "short"
    assert _truncate(42, 100) == 42


def test_oversized_arguments_are_clipped_in_the_log(arc_home_tmp: Path) -> None:
    logger = AuditLogger(max_field_chars=20)
    logger.record("screenshot", args={"data": "b" * 5000})
    assert logger.read_recent()[0]["args"]["data"].endswith(_TRUNCATION_SUFFIX)


def test_track_records_success(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    with logger.track("tool.run", tool="shell"):
        pass
    record = logger.read_recent()[0]
    assert record["status"] == "ok"
    assert record["duration_ms"] is not None


def test_track_records_failure_and_reraises(arc_home_tmp: Path) -> None:
    """The failing case is the one you most want recorded."""
    logger = AuditLogger()
    with pytest.raises(ValueError, match="boom"), logger.track("tool.run", tool="shell"):
        raise ValueError("boom")

    record = logger.read_recent()[0]
    assert record["status"] == "error"
    assert "ValueError: boom" in record["error"]


def test_track_marks_dry_run(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    with logger.track("tool.run", dry_run=True):
        pass
    assert logger.read_recent()[0]["status"] == "dry_run"


def test_read_recent_skips_malformed_lines(arc_home_tmp: Path) -> None:
    """A truncated final line from a hard kill must not block reading the rest."""
    logger = AuditLogger()
    logger.record("good")
    with logger.path_for_today().open("a", encoding="utf-8") as handle:
        handle.write("{ this is not json\n")
    logger.record("also-good")

    events = [r["event"] for r in logger.read_recent()]
    assert events == ["good", "also-good"]


def test_read_recent_on_missing_file(arc_home_tmp: Path) -> None:
    assert AuditLogger().read_recent() == []


def test_read_recent_respects_limit(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    for i in range(10):
        logger.record(f"e{i}")
    assert len(logger.read_recent(limit=3)) == 3


def test_directory_is_created_on_demand(arc_home_tmp: Path) -> None:
    assert not audit_dir().exists()
    AuditLogger().record("first")
    assert audit_dir().is_dir()


def test_write_failure_raises(arc_home_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Failing to record is fatal, not ignorable — that is the point of the log."""
    logger = AuditLogger()

    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "mkdir", boom)
    with pytest.raises(AuditError, match="could not append"):
        logger.record("doomed")


def test_context_manager_brackets_the_session(arc_home_tmp: Path) -> None:
    with AuditLogger() as logger:
        logger.record("work")
    events = [r["event"] for r in logger.read_recent()]
    assert events == ["session.start", "work", "session.end"]


def test_context_manager_records_exception_and_propagates(arc_home_tmp: Path) -> None:
    logger = AuditLogger()
    with pytest.raises(RuntimeError, match="fail"), logger:
        raise RuntimeError("fail")

    end = logger.read_recent()[-1]
    assert end["event"] == "session.end"
    assert end["status"] == "error"
    assert "RuntimeError: fail" in end["error"]
