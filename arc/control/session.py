"""Control sessions — ARC takes the mouse and keyboard, visibly and reversibly.

Three guarantees, in priority order:

1. **You can always take it back.** Moving the mouse yourself ends the session
   immediately. So does Escape, so does ``arc control release``, so does ``arc-kill``.
   None of these depend on the agent being responsive, because the moment you most
   need to reclaim control is the moment the agent is misbehaving.
2. **You can always tell.** A blue glow surrounds every display for exactly as long as
   ARC holds control. It appears before the first synthetic event and disappears after
   the last.
3. **It cannot get stuck on.** The overlay is a child process and the session is a
   context manager, so an exception, a crash, or a kill all release control and clear
   the glow.

Takeover detection compares the real pointer position against where ARC last put it.
Polling rather than an event tap: a tap needs its own Accessibility grant and a run
loop, and the failure mode of polling — noticing a takeover 100 ms late — is far kinder
than the failure mode of a tap that silently stops delivering events.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from arc.audit import AuditLogger
from arc.errors import ControlError
from arc.log import get_logger

_log = get_logger(__name__)

#: How often to check whether the user has grabbed the pointer.
POLL_INTERVAL = 0.1

#: Pointer movement, in points, that counts as the user taking over. Small enough to
#: catch a deliberate nudge, large enough to ignore sensor jitter on a trackpad.
TAKEOVER_THRESHOLD = 12.0


@dataclass
class ControlState:
    """What a control session is currently doing."""

    active: bool = False
    reason: str = ""
    started_at: float = 0.0
    #: Where ARC last deliberately placed the pointer, for takeover comparison.
    expected_pointer: tuple[float, float] | None = None
    released_by: str = ""
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "active": self.active,
            "reason": self.reason,
            "held_seconds": round(time.monotonic() - self.started_at, 1) if self.active else 0.0,
            "released_by": self.released_by,
        }


class ControlSession:
    """Holds mouse and keyboard control, with a visible indicator and instant takeover."""

    def __init__(
        self,
        *,
        reason: str = "agent task",
        audit: AuditLogger | None = None,
        show_overlay: bool = True,
        watch_for_takeover: bool = True,
    ) -> None:
        self._reason = reason
        self._audit = audit
        self._show_overlay = show_overlay
        self._watch = watch_for_takeover
        self._overlay: subprocess.Popen[bytes] | None = None
        self._watcher: threading.Thread | None = None
        self._stop = threading.Event()
        self.state = ControlState()

    # ── Lifecycle ───────────────────────────────────────────────────────────────

    def __enter__(self) -> ControlSession:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release("session ended")

    def acquire(self) -> None:
        """Take control: show the glow, start watching for a takeover."""
        if self.state.active:
            return

        self.state.active = True
        self.state.started_at = time.monotonic()
        self.state.reason = self._reason
        self.state.released_by = ""
        self._stop.clear()

        if self._show_overlay:
            self._start_overlay()
        if self._watch:
            self._start_watcher()

        _log.info("control acquired", extra={"reason": self._reason})
        if self._audit is not None:
            self._audit.record("control.acquire", tool="control", args={"reason": self._reason})

    def release(self, by: str = "arc") -> None:
        """Give control back and clear the glow. Safe to call repeatedly."""
        if not self.state.active:
            return

        held = time.monotonic() - self.state.started_at
        self.state.active = False
        self.state.released_by = by
        self._stop.set()

        self._stop_overlay()

        if self._watcher is not None and self._watcher is not threading.current_thread():
            self._watcher.join(timeout=1.0)
        self._watcher = None

        _log.info("control released", extra={"by": by, "held_seconds": round(held, 1)})
        if self._audit is not None:
            self._audit.record(
                "control.release",
                tool="control",
                args={"released_by": by},
                result={"held_seconds": round(held, 1)},
            )

    def check(self) -> None:
        """Raise if control has been lost. Called before every synthetic event.

        This is what makes takeover *immediate* rather than eventual: the agent cannot
        deliver another click after you have taken the mouse, because every send goes
        through here first.
        """
        if not self.state.active:
            raise ControlError(
                f"control was released by {self.state.released_by or 'the user'}; "
                "not sending further input"
            )

    # ── The glow ────────────────────────────────────────────────────────────────

    def _start_overlay(self) -> None:
        """Launch the indicator in its own process."""
        try:
            self._overlay = subprocess.Popen(
                [sys.executable, "-m", "arc.control.overlay"],
                # The pipe is the shutdown channel, not decoration: closing it is what
                # makes the glow vanish. See overlay._exit_when_parent_goes.
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            # A missing indicator must not silently mean *no* indicator while control
            # proceeds. Refuse to take control you cannot signal.
            raise ControlError(f"could not show the control indicator: {exc}") from exc

    def _stop_overlay(self) -> None:
        """Terminate the indicator, escalating if it does not go quietly."""
        if self._overlay is None:
            return
        try:
            # Closing stdin is the fast path — the overlay exits within milliseconds.
            # terminate() alone does not work: AppKit's run loop never lets Python
            # service the signal.
            if self._overlay.stdin is not None:
                self._overlay.stdin.close()
            self._overlay.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            self._overlay.kill()
        except OSError:
            pass
        self._overlay = None

    # ── Takeover ────────────────────────────────────────────────────────────────

    def _start_watcher(self) -> None:
        """Watch the pointer for the user reclaiming control."""
        self._watcher = threading.Thread(
            target=self._watch_pointer, name="arc-control-watch", daemon=True
        )
        self._watcher.start()

    def _watch_pointer(self) -> None:
        """Poll the pointer; release the moment it moves somewhere ARC did not put it."""
        while not self._stop.wait(POLL_INTERVAL):
            expected = self.state.expected_pointer
            if expected is None:
                continue

            current = pointer_position()
            if current is None:
                continue

            drift = max(abs(current[0] - expected[0]), abs(current[1] - expected[1]))
            if drift > TAKEOVER_THRESHOLD:
                _log.info("user took over the pointer", extra={"drift": round(drift, 1)})
                self.release("user moved the mouse")
                return

    def note_pointer(self, position: tuple[float, float]) -> None:
        """Record where ARC just put the pointer, so its own motion is not a takeover."""
        self.state.expected_pointer = position


def pointer_position() -> tuple[float, float] | None:
    """Return the current pointer position, or None if it cannot be read."""
    try:
        import Quartz
    except ImportError:  # pragma: no cover - non-macOS
        return None

    try:
        event = Quartz.CGEventCreate(None)
        point = Quartz.CGEventGetLocation(event)
        return (float(point.x), float(point.y))
    except Exception:  # pragma: no cover - defensive
        return None


#: The process-wide session. Input tools consult this before every event, so control
#: state is one thing rather than a parameter threaded through every call site.
_current: ControlSession | None = None


def current() -> ControlSession | None:
    """Return the active control session, if any."""
    return _current


def start(
    *, reason: str = "agent task", audit: AuditLogger | None = None, overlay: bool = True
) -> ControlSession:
    """Begin a control session and make it the current one."""
    global _current
    if _current is not None and _current.state.active:
        return _current

    session = ControlSession(reason=reason, audit=audit, show_overlay=overlay)
    session.acquire()
    _current = session
    return session


def stop(by: str = "arc") -> None:
    """End the current control session, if there is one."""
    global _current
    if _current is not None:
        _current.release(by)
    _current = None


def require() -> ControlSession:
    """Return the active session, or raise.

    Input tools call this rather than acting unconditionally, so a synthetic event can
    never be delivered without a visible indicator being up.
    """
    session = _current
    if session is None or not session.state.active:
        raise ControlError(
            "ARC does not currently have input control. "
            "Start a session first so the indicator is visible."
        )
    session.check()
    return session
