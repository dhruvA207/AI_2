"""Tests for task journaling and crash recovery.

The scenario these protect: an agent eight steps into a task, having created files and
run commands, when the process dies. Getting recovery subtly wrong means re-running
mutating steps — doing the work twice — which is worse than not recovering at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc.agent import journal
from arc.errors import ArcError


class FakeObservation:
    def __init__(self, ok: bool = True, output: str = "done", dry_run: bool = False) -> None:
        self.ok = ok
        self.output = output
        self.dry_run = dry_run


def test_journal_records_a_completed_run(arc_home_tmp: Path) -> None:
    entry = journal.Journal("do a thing")
    entry.step(1, "read_file", {"path": "/tmp/x"}, FakeObservation())
    entry.finish("all done")

    loaded = journal.load(entry.id)
    assert loaded.task == "do a thing"
    assert loaded.status == "completed"
    assert loaded.answer == "all done"
    assert loaded.tools_used == ["read_file"]


def test_a_journal_with_no_finish_is_interrupted(arc_home_tmp: Path) -> None:
    """This is the crash case: the process died before writing an outcome."""
    entry = journal.Journal("do a thing")
    entry.step(1, "read_file", {}, FakeObservation())

    assert journal.load(entry.id).status == "interrupted"


def test_failures_are_recorded(arc_home_tmp: Path) -> None:
    entry = journal.Journal("doomed")
    entry.fail("RuntimeError: boom")
    assert journal.load(entry.id).status == "failed"


def test_truncated_final_line_is_survivable(arc_home_tmp: Path) -> None:
    """A journal from a killed process very often ends mid-write, and that is exactly
    the journal you most want to read."""
    entry = journal.Journal("do a thing")
    entry.step(1, "read_file", {}, FakeObservation())
    with entry.record.path.open("a", encoding="utf-8") as handle:
        handle.write('{"event": "step", "number": 2, "tool": "wri')

    loaded = journal.load(entry.id)
    assert len(loaded.steps) == 1
    assert loaded.status == "interrupted"


def test_missing_journal_raises(arc_home_tmp: Path) -> None:
    with pytest.raises(ArcError, match="no journal"):
        journal.load("nonexistent")


def test_mutating_steps_are_distinguished(arc_home_tmp: Path) -> None:
    """Re-running a read is free; re-running a delete is not."""
    entry = journal.Journal("mixed")
    entry.step(1, "read_file", {}, FakeObservation())
    entry.step(2, "write_file", {}, FakeObservation(dry_run=True))

    loaded = journal.load(entry.id)
    assert len(loaded.steps) == 2
    assert len(loaded.mutating_steps) == 1


def test_resume_summary_tells_the_model_what_happened(arc_home_tmp: Path) -> None:
    entry = journal.Journal("do a thing")
    entry.step(1, "read_file", {"path": "/tmp/a"}, FakeObservation(output="alpha"))
    entry.step(2, "write_file", {"path": "/tmp/b"}, FakeObservation(ok=False, output="denied"))

    summary = journal.load(entry.id).summarize_for_model()
    assert "read_file" in summary
    assert "alpha" in summary
    assert "FAILED" in summary
    assert "Do not repeat" in summary


def test_resume_summary_when_nothing_happened(arc_home_tmp: Path) -> None:
    entry = journal.Journal("do a thing")
    assert "No steps completed" in journal.load(entry.id).summarize_for_model()


def test_marking_resumed_clears_the_interrupted_flag(arc_home_tmp: Path) -> None:
    """Regression: an interrupted task stayed interrupted forever even after being
    resumed and finished, so `arc task list` filled with entries that looked like
    unfinished work and buried the one that was."""
    entry = journal.Journal("do a thing")
    entry.step(1, "read_file", {}, FakeObservation())
    assert journal.load(entry.id).status == "interrupted"

    journal.mark_resumed(entry.id, "newtask123")
    loaded = journal.load(entry.id)
    assert loaded.status == "resumed"
    assert loaded.resumed_by == "newtask123"


def test_marking_resumed_does_not_duplicate_the_start_line(arc_home_tmp: Path) -> None:
    """Going through Journal() would write a second "start", making the task's own
    history ambiguous."""
    entry = journal.Journal("do a thing")
    journal.mark_resumed(entry.id, "other")

    lines = [json.loads(line) for line in entry.record.path.read_text().splitlines()]
    assert sum(1 for line in lines if line["event"] == "start") == 1


def test_marking_an_absent_journal_is_a_noop(arc_home_tmp: Path) -> None:
    journal.mark_resumed("nonexistent", "other")


def test_interrupted_lists_only_unfinished_tasks(arc_home_tmp: Path) -> None:
    done = journal.Journal("finished")
    done.finish("ok")
    stuck = journal.Journal("crashed")
    stuck.step(1, "read_file", {}, FakeObservation())

    ids = [record.id for record in journal.interrupted()]
    assert stuck.id in ids
    assert done.id not in ids


def test_recent_is_newest_first(arc_home_tmp: Path) -> None:
    import time

    first = journal.Journal("older")
    first.finish("a")
    time.sleep(0.02)
    second = journal.Journal("newer")
    second.finish("b")

    assert journal.recent(5)[0].id == second.id


def test_journal_write_failure_does_not_kill_the_task(
    arc_home_tmp: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A journal that cannot be written is a lost recovery option; one that raises
    takes the whole task with it, which is strictly worse."""
    entry = journal.Journal("do a thing")

    def boom(*_a: object, **_k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", boom)
    entry.step(1, "read_file", {}, FakeObservation())  # must not raise
