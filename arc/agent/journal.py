"""Task journaling — surviving a crash mid-task.

§7 asks for crash recovery and resumable tasks. The failure this addresses is concrete:
an agent eight steps into a task, having created files and run commands, when the
process dies. Without a record, the next run starts from nothing and repeats work that
already happened — which for mutating tools means doing it *twice*.

So every step is appended to ``~/.arc/tasks/<id>.jsonl`` as it completes. Append-only
JSONL for the same reason the audit log is: a partial write costs the last line rather
than the file, and a crash mid-write cannot corrupt what came before.

Resumption is deliberately **not automatic**. The journal records what happened and
``arc task resume`` replays it into the model's context so it can continue with full
knowledge of what it already did. Silently re-running a task that was halfway through
deleting things is precisely the behaviour a recovery mechanism should not have.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from arc.errors import ArcError
from arc.log import get_logger
from arc.paths import arc_home

_log = get_logger(__name__)


def task_dir() -> Path:
    """Where task journals live."""
    target = arc_home() / "tasks"
    target.mkdir(parents=True, exist_ok=True)
    return target


@dataclass
class TaskRecord:
    """A journalled task and everything known about its progress."""

    id: str
    task: str
    started_at: float
    status: str = "running"  # running | completed | failed | abandoned
    steps: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    finished_at: float = 0.0
    #: Set when this task was continued by a later one.
    resumed_by: str = ""

    @property
    def path(self) -> Path:
        """Where this task's journal is written."""
        return task_dir() / f"{self.id}.jsonl"

    @property
    def tools_used(self) -> list[str]:
        """Tools called so far, in order."""
        return [s["tool"] for s in self.steps if s.get("tool")]

    @property
    def mutating_steps(self) -> list[dict[str, Any]]:
        """Steps that changed something.

        The ones that matter on resume: re-running a read is free, re-running a delete
        is not.
        """
        return [s for s in self.steps if s.get("mutating")]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": len(self.steps),
            "tools_used": self.tools_used,
            "answer": self.answer[:500],
            "resumed_by": self.resumed_by,
        }

    def summarize_for_model(self) -> str:
        """Render prior progress so a resumed run knows what it already did."""
        if not self.steps:
            return "No steps completed before the interruption."

        lines = [f"You were working on this task and completed {len(self.steps)} step(s):"]
        for step in self.steps:
            outcome = "succeeded" if step.get("ok") else "FAILED"
            lines.append(
                f"  {step['number']}. {step.get('tool')}({_brief(step.get('arguments'))}) "
                f"— {outcome}: {str(step.get('output', ''))[:160]}"
            )
        lines.append(
            "\nContinue from here. Do not repeat steps that already succeeded, "
            "especially ones that changed files or ran commands."
        )
        return "\n".join(lines)


def _brief(arguments: Any) -> str:
    """Render arguments compactly for a resume summary."""
    if not isinstance(arguments, dict):
        return ""
    parts = []
    for key, value in arguments.items():
        text = repr(value)
        parts.append(f"{key}={text[:40]}")
    return ", ".join(parts)


class Journal:
    """Appends a task's progress to disk as it happens."""

    def __init__(self, task: str, *, task_id: str | None = None) -> None:
        self.record = TaskRecord(
            id=task_id or uuid.uuid4().hex[:12], task=task, started_at=time.time()
        )
        self._write({"event": "start", "task": task, "at": self.record.started_at})

    @property
    def id(self) -> str:
        """This task's identifier."""
        return self.record.id

    def _write(self, entry: dict[str, Any]) -> None:
        """Append one line. Failures are logged, never raised.

        A journal that cannot be written is a lost recovery option; a journal that
        *raises* takes the whole task with it, which is strictly worse.
        """
        try:
            with self.record.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, default=str) + "\n")
                handle.flush()
        except OSError as exc:
            _log.warning("could not write task journal: %s", exc)

    def step(self, number: int, tool: str | None, arguments: Any, observation: Any) -> None:
        """Record a completed step."""
        entry = {
            "event": "step",
            "number": number,
            "tool": tool,
            "arguments": arguments,
            "ok": bool(getattr(observation, "ok", True)),
            "output": str(getattr(observation, "output", ""))[:2000],
            "mutating": bool(getattr(observation, "dry_run", False) is False and tool),
            "at": time.time(),
        }
        self.record.steps.append(entry)
        self._write(entry)

    def finish(self, answer: str, *, status: str = "completed") -> None:
        """Record the outcome."""
        self.record.status = status
        self.record.answer = answer
        self.record.finished_at = time.time()
        self._write(
            {"event": "finish", "status": status, "answer": answer[:4000], "at": time.time()}
        )

    def fail(self, error: str) -> None:
        """Record that the task ended badly."""
        self.record.status = "failed"
        self.record.finished_at = time.time()
        self._write({"event": "failed", "error": error, "at": time.time()})


def mark_resumed(original_id: str, continued_by: str) -> None:
    """Record that an interrupted task was picked up by a later one.

    Appends directly rather than going through ``Journal``, whose constructor writes a
    fresh "start" line — doing that to an existing journal would give it two starts and
    make its own history ambiguous.

    Without this an interrupted task stays interrupted forever even after it was
    resumed and finished, so `arc task list` fills with entries that look like
    unfinished work, and the one that genuinely is gets lost among them.
    """
    path = task_dir() / f"{original_id}.jsonl"
    if not path.is_file():
        return
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"event": "resumed", "by": continued_by, "at": time.time()}) + "\n"
            )
    except OSError as exc:  # pragma: no cover - disk failure
        _log.warning("could not mark task %s resumed: %s", original_id, exc)


def load(task_id: str) -> TaskRecord:
    """Read a journal back from disk.

    Malformed lines are skipped rather than fatal: the last line of a journal from a
    crashed process is very often truncated, and that is exactly the journal you most
    want to read.
    """
    path = task_dir() / f"{task_id}.jsonl"
    if not path.is_file():
        raise ArcError(f"no journal for task {task_id!r}")

    record = TaskRecord(id=task_id, task="", started_at=0.0)
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = entry.get("event")
        if event == "start":
            record.task = str(entry.get("task", ""))
            record.started_at = float(entry.get("at", 0.0))
        elif event == "step":
            record.steps.append(entry)
        elif event == "finish":
            record.status = str(entry.get("status", "completed"))
            record.answer = str(entry.get("answer", ""))
            record.finished_at = float(entry.get("at", 0.0))
        elif event == "failed":
            record.status = "failed"
            record.finished_at = float(entry.get("at", 0.0))
        elif event == "resumed":
            record.status = "resumed"
            record.resumed_by = str(entry.get("by", ""))
            record.finished_at = float(entry.get("at", 0.0))

    # A journal with no finish line is a task whose process died mid-run — which is
    # precisely the case resumption exists for.
    if record.status == "running" and record.finished_at == 0.0:
        record.status = "interrupted"
    return record


def recent(limit: int = 20) -> list[TaskRecord]:
    """Return recent tasks, newest first."""
    found: list[TaskRecord] = []
    for path in sorted(task_dir().glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            found.append(load(path.stem))
        except ArcError:
            continue
        if len(found) >= limit:
            break
    return found


def interrupted() -> list[TaskRecord]:
    """Return tasks that were never finished — the resumable ones."""
    return [record for record in recent(50) if record.status == "interrupted"]
