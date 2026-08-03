"""Move and click the cursor for gesture control.

Deliberately independent of :mod:`arc.control.input`, and therefore of ARC's control
session. Camera gestures are a *feature you switch on*, not ARC taking over the machine:
your own hand is driving the cursor the whole time. Routing it through the control
session would raise the blue glow, register a kill-switch entry, and — worst of all —
end the session the moment you touched the physical mouse, none of which makes sense for
a mode you asked to be in and can leave by asking.

That does mean the two paths are similar by nature. They are not shared on purpose: the
control-session path exists to make *ARC's* synthetic input visible and interruptible,
and borrowing it here would either weaken that guarantee or impose it where it does not
belong.

Needs Accessibility permission, the same grant window control needs.
"""

from __future__ import annotations

from typing import Any

from arc.errors import PlatformError


def _quartz() -> Any:
    try:
        import Quartz
    except ImportError as exc:  # pragma: no cover - non-macOS
        raise PlatformError(
            "cursor control needs pyobjc. Install with: pip install 'arc[camera]'"
        ) from exc
    return Quartz


def position() -> tuple[float, float] | None:
    """Where the cursor is now, or None if it cannot be read."""
    quartz = _quartz()
    try:
        event = quartz.CGEventCreate(None)
        point = quartz.CGEventGetLocation(event)
        return (float(point.x), float(point.y))
    except Exception:  # pragma: no cover - defensive
        return None


def move_to(x: float, y: float) -> None:
    """Put the cursor at a screen coordinate.

    No animation between points, unlike the control-session path: the hand is already
    moving continuously and the caller has smoothed it, so interpolating again would
    only add lag to something being steered in real time.
    """
    quartz = _quartz()
    event = quartz.CGEventCreateMouseEvent(None, quartz.kCGEventMouseMoved, (x, y), 0)
    quartz.CGEventPost(quartz.kCGHIDEventTap, event)


def left_click() -> None:
    """Click wherever the cursor already is."""
    quartz = _quartz()
    where = position()
    if where is None:
        raise PlatformError("could not read the cursor position")

    for kind in (quartz.kCGEventLeftMouseDown, quartz.kCGEventLeftMouseUp):
        event = quartz.CGEventCreateMouseEvent(None, kind, where, 0)
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
