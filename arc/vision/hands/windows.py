"""Move and resize the frontmost window through the Accessibility API.

Direct AX reads and writes rather than AppleScript: this runs every frame, and a
subprocess per frame would cap the whole loop at a few updates a second.

The target is deliberately the frontmost window that is *not* ours — otherwise the
first fist grabs the gesture preview window and the user watches it fly off the screen
instead of the window they meant.
"""

from __future__ import annotations

import os
from typing import Any

from arc.errors import PlatformError

#: Window owners that are never a drag target: system chrome, and our own preview.
_SKIP_OWNERS = frozenset(
    {"Window Server", "Dock", "Spotlight", "Notification Center", "Python", "ARC"}
)

_SELF_PID = os.getpid()


def _services() -> tuple[Any, Any]:
    """Return the Quartz and ApplicationServices modules, or raise something useful."""
    try:
        import ApplicationServices
        import Quartz
    except ImportError as exc:  # pragma: no cover - non-macOS
        raise PlatformError(
            "window control needs pyobjc. Install with: pip install 'arc[camera]'"
        ) from exc
    return Quartz, ApplicationServices


def screen_size() -> tuple[int, int]:
    """Size of the main display, for placing the preview window."""
    quartz, _ = _services()
    bounds = quartz.CGDisplayBounds(quartz.CGMainDisplayID())
    return int(bounds.size.width), int(bounds.size.height)


def desktop_bounds() -> tuple[int, int, int, int]:
    """``(x, y, w, h)`` covering every display, in the coordinates AX writes.

    Mapping hand travel against the main display alone means a window cannot be thrown
    to another monitor in one motion — it stalls at the edge and has to be ratcheted
    across.
    """
    quartz, _ = _services()
    error, display_ids, count = quartz.CGGetActiveDisplayList(16, None, None)
    if error != 0 or not count:
        width, height = screen_size()
        return 0, 0, width, height

    x0 = y0 = float("inf")
    x1 = y1 = float("-inf")
    for display_id in display_ids[:count]:
        bounds = quartz.CGDisplayBounds(display_id)
        x0 = min(x0, bounds.origin.x)
        y0 = min(y0, bounds.origin.y)
        x1 = max(x1, bounds.origin.x + bounds.size.width)
        y1 = max(y1, bounds.origin.y + bounds.size.height)
    return int(x0), int(y0), int(x1 - x0), int(y1 - y0)


def _frontmost_other_app() -> tuple[int | None, str | None]:
    """PID and name of the app owning the frontmost normal window that is not ours."""
    quartz, _ = _services()
    windows = quartz.CGWindowListCopyWindowInfo(
        quartz.kCGWindowListOptionOnScreenOnly | quartz.kCGWindowListExcludeDesktopElements,
        quartz.kCGNullWindowID,
    )
    for window in windows or []:  # front-to-back order
        if window.get("kCGWindowLayer", 0) != 0:  # menu bar, dock, our own overlay
            continue
        pid = window.get("kCGWindowOwnerPID")
        if pid in (None, _SELF_PID):
            continue
        name = window.get("kCGWindowOwnerName", "")
        if name in _SKIP_OWNERS:
            continue
        return pid, name
    return None, None


class WindowController:
    """Holds one target window and pushes new geometry to it."""

    def __init__(self) -> None:
        self._window: Any = None
        self.name: str | None = None

    def acquire(self) -> bool:
        """Grab the frontmost window that is not ours. False when there is none."""
        _, services = _services()
        pid, name = _frontmost_other_app()
        if pid is None:
            self._window = None
            return False

        app = services.AXUIElementCreateApplication(pid)
        error, window = services.AXUIElementCopyAttributeValue(
            app, services.kAXFocusedWindowAttribute, None
        )
        if error != 0 or window is None:
            # An app can be frontmost with nothing focused; its first window will do.
            error, windows = services.AXUIElementCopyAttributeValue(
                app, services.kAXWindowsAttribute, None
            )
            window = windows[0] if (error == 0 and windows) else None

        self._window = window
        self.name = name
        return window is not None

    def frame(self) -> tuple[float, float, float, float] | None:
        """``(x, y, w, h)`` of the target window, or None if it went away."""
        if not self._window:
            return None
        _, services = _services()
        pos_error, position = services.AXUIElementCopyAttributeValue(
            self._window, services.kAXPositionAttribute, None
        )
        size_error, size = services.AXUIElementCopyAttributeValue(
            self._window, services.kAXSizeAttribute, None
        )
        if pos_error != 0 or size_error != 0 or position is None or size is None:
            return None
        _, point = services.AXValueGetValue(position, services.kAXValueCGPointType, None)
        _, extent = services.AXValueGetValue(size, services.kAXValueCGSizeType, None)
        return (point.x, point.y, extent.width, extent.height)

    def set_position(self, x: float, y: float) -> None:
        if not self._window:
            return
        quartz, services = _services()
        value = services.AXValueCreate(
            services.kAXValueCGPointType, quartz.CGPoint(float(x), float(y))
        )
        services.AXUIElementSetAttributeValue(self._window, services.kAXPositionAttribute, value)

    def set_size(self, width: float, height: float) -> None:
        if not self._window:
            return
        quartz, services = _services()
        value = services.AXValueCreate(
            services.kAXValueCGSizeType, quartz.CGSize(float(width), float(height))
        )
        services.AXUIElementSetAttributeValue(self._window, services.kAXSizeAttribute, value)

    def release(self) -> None:
        self._window = None
        self.name = None
