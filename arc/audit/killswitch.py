"""The kill switch.

An agent with mouse and keyboard control that enters a loop can make the machine
unusable, and at that point you may not be able to interact with a terminal reliably.
So stopping ARC must not depend on ARC being healthy: every process writes a PID file
to ``~/.arc/run/`` on startup, and ``arc kill`` reads those files from a *separate*
process and SIGKILLs the trees. Nothing in the kill path touches the agent's own
state, event loop, or memory.

SIGKILL rather than SIGTERM is deliberate. A graceful shutdown asks a process to
cooperate; a wedged process cannot. Phase 6 adds the global hotkey on top of the same
machinery, which needs OS-level input hooks that do not exist yet.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

from arc.log import get_logger
from arc.paths import run_dir
from arc.platform import get_platform

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ProcessEntry:
    """A registered ARC process."""

    name: str
    pid: int
    started_at: str
    path: Path

    @property
    def alive(self) -> bool:
        """Whether the process still exists.

        ``kill(pid, 0)`` performs the permission and existence checks without
        delivering a signal — the standard liveness probe.
        """
        try:
            os.kill(self.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # It exists, we just do not own it. Treat as alive so we do not silently
            # reap a PID file for a running process.
            return True
        return True


class KillSwitch:
    """Registers this process and terminates registered ones."""

    def __init__(self, *, directory: Path | None = None) -> None:
        self._dir = directory if directory is not None else run_dir()

    def register(self, name: str = "arc") -> Path:
        """Write a PID file for the current process.

        The filename includes the PID so two ARC processes (say the agent and a
        training supervisor) do not clobber each other's registration.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{name}-{os.getpid()}.pid"
        payload = {
            "name": name,
            "pid": os.getpid(),
            "started_at": datetime.now(UTC).isoformat(),
            "argv": sys.argv,
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path

    def unregister(self, name: str = "arc") -> None:
        """Remove this process's PID file. Safe to call when it is already gone."""
        (self._dir / f"{name}-{os.getpid()}.pid").unlink(missing_ok=True)

    def registered(self) -> list[ProcessEntry]:
        """Return every registered process, skipping unreadable files."""
        if not self._dir.is_dir():
            return []

        entries: list[ProcessEntry] = []
        for path in sorted(self._dir.glob("*.pid")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries.append(
                    ProcessEntry(
                        name=str(data["name"]),
                        pid=int(data["pid"]),
                        started_at=str(data.get("started_at", "unknown")),
                        path=path,
                    )
                )
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
        return entries

    def reap_stale(self) -> int:
        """Delete PID files whose processes are gone. Returns how many were removed.

        Called before killing so the report reflects what is actually running rather
        than counting corpses from a previous crash.
        """
        removed = 0
        for entry in self.registered():
            if not entry.alive:
                entry.path.unlink(missing_ok=True)
                removed += 1
        return removed

    def kill_all(self, *, include_self: bool = False) -> list[int]:
        """SIGKILL every registered process tree. Returns the PIDs signalled.

        ``include_self`` defaults to False so that running ``arc kill`` from inside a
        registered process does not kill the very process doing the killing before it
        finishes the job.
        """
        own = os.getpid()
        platform = get_platform()
        killed: list[int] = []

        for entry in self.registered():
            if entry.pid == own and not include_self:
                continue
            if not entry.alive:
                entry.path.unlink(missing_ok=True)
                continue
            try:
                killed.extend(platform.kill_process_tree(entry.pid))
            except OSError as exc:
                _log.warning("could not kill pid %d: %s", entry.pid, exc)
                continue
            entry.path.unlink(missing_ok=True)

        return killed

    def install_signal_handlers(self, on_shutdown: Callable[[], None] | None = None) -> None:
        """Clean up the PID file when the process is asked to exit.

        Only handles SIGINT and SIGTERM — SIGKILL cannot be caught by design, which is
        precisely why the kill switch uses it and why ``reap_stale`` exists to clean up
        afterwards.
        """

        def _handler(signum: int, _frame: FrameType | None) -> None:
            _log.info("received signal %d, shutting down", signum)
            self.unregister()
            if on_shutdown is not None:
                on_shutdown()
            raise SystemExit(128 + signum)

        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the standalone ``arc-kill`` command.

    Deliberately separate from ``arc kill``. The kill switch exists for the case where
    ARC has wedged the machine, and at that moment the last thing you want is for
    stopping it to depend on loading YAML, probing hardware, or the subcommand
    dispatch working. This path imports nothing beyond the standard library and this
    module, does no configuration, and writes no logs.

    ``arc kill`` remains as a convenience alias for when things are fine.
    """
    parser = argparse.ArgumentParser(
        prog="arc-kill",
        description="SIGKILL every registered ARC process tree. Works even when ARC does not.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be killed, kill nothing"
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)

    switch = KillSwitch()
    reaped = switch.reap_stale()
    entries = switch.registered()

    if args.dry_run:
        payload = {
            "dry_run": True,
            "would_kill": [{"name": e.name, "pid": e.pid} for e in entries],
            "reaped_stale": reaped,
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            listing = ", ".join(f"{e.name}:{e.pid}" for e in entries) or "none"
            print(f"[dry-run] would kill {len(entries)} process(es): {listing}")
        return 0

    killed = switch.kill_all()

    if args.json:
        print(json.dumps({"killed": killed, "reaped_stale": reaped}, indent=2))
        return 0

    if not killed and not reaped:
        print("no ARC processes registered")
    else:
        if killed:
            print(f"killed {len(killed)} process(es): {', '.join(str(p) for p in killed)}")
        if reaped:
            print(f"cleaned up {reaped} stale PID file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
