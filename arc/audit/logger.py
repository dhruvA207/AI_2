"""Append-only action log.

Every tool call ARC makes lands here as one JSON object per line, in
``~/.arc/audit/YYYY-MM-DD.jsonl``. Append-only and never rotated or truncated: an
action log with silent gaps is worse than none, because it invites false confidence
while debugging.

Failures to write are raised, not swallowed. If we cannot record what the agent is
doing, that is worth interrupting for.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from arc.errors import AuditError
from arc.paths import audit_dir

Status = Literal["ok", "error", "dry_run", "denied"]

#: Long arguments (a file's whole contents, a base64 screenshot) would make the log
#: unreadable and enormous. Truncate the *serialized* form and mark that we did, so a
#: reader can tell the difference between a short value and a clipped one.
_DEFAULT_MAX_FIELD_CHARS = 4000

_TRUNCATION_SUFFIX = "...[truncated]"


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One recorded action."""

    ts: str
    session_id: str
    seq: int
    event: str
    status: Status
    tool: str | None = None
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    duration_ms: float | None = None
    dry_run: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


def _truncate(value: Any, limit: int) -> Any:
    """Shorten oversized values, preserving structure where possible.

    Recurses into dicts and lists so that one enormous field does not force us to
    discard its siblings.
    """
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return value[:limit] + _TRUNCATION_SUFFIX
    if isinstance(value, dict):
        return {k: _truncate(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 100:
            return [_truncate(v, limit) for v in value[:100]] + [_TRUNCATION_SUFFIX]
        return [_truncate(v, limit) for v in value]
    return value


class AuditLogger:
    """Thread-safe append-only writer for the action log."""

    def __init__(
        self,
        *,
        directory: Path | None = None,
        session_id: str | None = None,
        fsync: bool = False,
        max_field_chars: int = _DEFAULT_MAX_FIELD_CHARS,
    ) -> None:
        """Create a logger.

        ``fsync`` defaults to False: flushing to disk on every tool call costs a few
        milliseconds each, which is real overhead in a tight agent loop. The data is
        still handed to the OS immediately, so it survives a process crash — only a
        kernel panic or power loss can lose the tail. Turn it on in
        ``config/policy.yaml`` if that trade is not worth it to you.
        """
        self._dir = directory if directory is not None else audit_dir()
        self._session_id = session_id or uuid.uuid4().hex[:12]
        self._fsync = fsync
        self._max_field_chars = max_field_chars
        self._lock = threading.Lock()
        self._seq = 0

    @property
    def session_id(self) -> str:
        """Identifier tying every record from this process together."""
        return self._session_id

    def path_for_today(self) -> Path:
        """Return the log file for the current UTC day.

        Computed per write rather than cached so a long-running process rolls over at
        midnight instead of writing tomorrow's events into yesterday's file.
        """
        return self._dir / f"{datetime.now(UTC):%Y-%m-%d}.jsonl"

    def record(
        self,
        event: str,
        *,
        status: Status = "ok",
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        result: Any = None,
        error: str | None = None,
        duration_ms: float | None = None,
        dry_run: bool = False,
        meta: dict[str, Any] | None = None,
    ) -> AuditRecord:
        """Append one record and return it."""
        with self._lock:
            self._seq += 1
            seq = self._seq

        limit = self._max_field_chars
        record = AuditRecord(
            ts=datetime.now(UTC).isoformat(),
            session_id=self._session_id,
            seq=seq,
            event=event,
            status=status,
            tool=tool,
            args=_truncate(args or {}, limit),
            result=_truncate(result, limit),
            error=error,
            duration_ms=duration_ms,
            dry_run=dry_run,
            meta=meta or {},
        )
        self._write(record)
        return record

    def _write(self, record: AuditRecord) -> None:
        """Append a single line, creating the directory on demand."""
        line = json.dumps(record.to_dict(), default=str)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            with self._lock, self.path_for_today().open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
        except OSError as exc:
            raise AuditError(f"could not append to audit log: {exc}") from exc

    @contextmanager
    def track(
        self,
        event: str,
        *,
        tool: str | None = None,
        args: dict[str, Any] | None = None,
        dry_run: bool = False,
    ) -> Any:
        """Time a block and record its outcome, including on failure.

        Wrapping a tool call in this is what guarantees the log has an entry even when
        the call raises — the case you most want recorded.
        """
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            self.record(
                event,
                status="error",
                tool=tool,
                args=args,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
                dry_run=dry_run,
            )
            raise
        self.record(
            event,
            status="dry_run" if dry_run else "ok",
            tool=tool,
            args=args,
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            dry_run=dry_run,
        )

    def read_recent(self, limit: int = 20) -> list[dict[str, Any]]:
        """Return the most recent records from today's file, oldest first.

        Malformed lines are skipped rather than raising: a truncated final line from a
        hard kill should not stop you reading the rest of the log.
        """
        path = self.path_for_today()
        if not path.is_file():
            return []
        try:
            with path.open(encoding="utf-8") as handle:
                lines = handle.readlines()
        except OSError as exc:
            raise AuditError(f"could not read audit log: {exc}") from exc

        records: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records

    def __enter__(self) -> AuditLogger:
        self.record("session.start", meta={"pid": os.getpid()})
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> Literal[False]:
        self.record(
            "session.end",
            status="error" if exc_type else "ok",
            error=f"{exc_type.__name__}: {exc}" if exc_type else None,
        )
        return False
