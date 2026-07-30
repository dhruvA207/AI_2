"""The control indicator — a soft blue glow around the screen while ARC has control.

When ARC is driving the mouse and keyboard, that must be **unmistakable**. An agent
moving the pointer with no visible signal is alarming and, worse, ambiguous: you cannot
tell a stuck agent from a stuck machine.

Three properties make this safe rather than decorative:

- **Click-through.** ``ignoresMouseEvents`` is set on every window. The glow can never
  intercept a click, so it cannot make the machine less usable than it was — which
  would defeat the purpose of an indicator meant to reassure.
- **Every display.** One window per screen, joining all Spaces. A glow on the laptop
  while ARC clicks on the external monitor would be worse than none.
- **Its own process.** Run as ``python -m arc.control.overlay``. AppKit demands the
  main thread for UI, and sharing that with the agent loop would mean a busy agent
  freezes the indicator exactly when you most want to trust it. A separate process also
  means the glow dies with the process tree when ``arc-kill`` fires.

Drawn as concentric rounded rectangles with alpha falling off inward, which gives a
soft bloom rather than a hard border.
"""

from __future__ import annotations

import sys
from typing import Any

#: ARC's accent. A soft blue that reads clearly on light and dark desktops without
#: looking like a system error, which red or amber would.
ACCENT = (0.29, 0.62, 1.0)  # #4A9EFF

#: How far the bloom reaches inward, in points.
GLOW_WIDTH = 22.0

#: Alpha at the outermost ring. Kept well below 1.0 so the glow reads as ambient light
#: rather than a painted frame.
PEAK_ALPHA = 0.55

#: Corner rounding, roughly matching macOS display corners.
CORNER_RADIUS = 14.0


def _make_glow_view(appkit: Any) -> Any:
    """Build the NSView subclass that paints the bloom.

    Defined inside a function rather than at module scope so this file imports cleanly
    on Windows and Linux, where AppKit does not exist. Subclassing a Cocoa class at
    import time would make the module unimportable there, and the platform factory
    needs to be able to *reason* about backends it cannot instantiate.
    """

    class GlowView(appkit.NSView):  # type: ignore[misc]
        """Paints concentric rounded rectangles with alpha falling off inward."""

        def drawRect_(self, _rect: Any) -> None:
            bounds = self.bounds()
            appkit.NSColor.clearColor().set()
            appkit.NSRectFill(bounds)

            steps = 18
            for index in range(steps):
                # 0 at the outer edge, 1 at the inner limit of the glow.
                progress = index / steps
                inset = progress * GLOW_WIDTH
                # Quadratic falloff reads as a softer bloom than a linear ramp.
                alpha = PEAK_ALPHA * (1.0 - progress) ** 2

                rect = appkit.NSInsetRect(bounds, inset, inset)
                if rect.size.width <= 0 or rect.size.height <= 0:
                    break

                path = appkit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                    rect, CORNER_RADIUS, CORNER_RADIUS
                )
                path.setLineWidth_(2.0)
                appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    ACCENT[0], ACCENT[1], ACCENT[2], alpha
                ).set()
                path.stroke()

        def isOpaque(self) -> bool:
            return False

    return GlowView


def _build_window(screen: Any, appkit: Any, view_class: Any) -> Any:
    """Create one click-through overlay window covering a screen."""
    frame = screen.frame()

    window = appkit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_screen_(
        frame,
        appkit.NSWindowStyleMaskBorderless,
        appkit.NSBackingStoreBuffered,
        False,
        screen,
    )

    window.setOpaque_(False)
    window.setBackgroundColor_(appkit.NSColor.clearColor())
    window.setHasShadow_(False)
    # The one non-negotiable property: the indicator must never eat a click.
    window.setIgnoresMouseEvents_(True)
    # Above normal windows and full-screen apps, but below the system's own alerts.
    window.setLevel_(appkit.NSScreenSaverWindowLevel)
    window.setCollectionBehavior_(
        appkit.NSWindowCollectionBehaviorCanJoinAllSpaces
        | appkit.NSWindowCollectionBehaviorStationary
        | appkit.NSWindowCollectionBehaviorFullScreenAuxiliary
        | appkit.NSWindowCollectionBehaviorIgnoresCycle
    )

    view = view_class.alloc().initWithFrame_(
        appkit.NSMakeRect(0, 0, frame.size.width, frame.size.height)
    )
    window.setContentView_(view)
    window.orderFrontRegardless()
    return window


def run() -> int:
    """Show the glow on every screen until the process is terminated.

    Blocks. Intended to be run as a child process by ``arc.control.session``.
    """
    try:
        import AppKit
    except ImportError:
        print("the control indicator requires macOS (AppKit)", file=sys.stderr)
        return 1

    application = AppKit.NSApplication.sharedApplication()
    # Accessory, not regular: no Dock icon, no menu bar, never steals focus.
    application.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    view_class = _make_glow_view(AppKit)
    windows = [_build_window(screen, AppKit, view_class) for screen in AppKit.NSScreen.screens()]
    if not windows:
        print("no screens found", file=sys.stderr)
        return 1

    _exit_when_parent_goes(sys.stdin)

    application.run()
    return 0


def _exit_when_parent_goes(stream: Any) -> None:
    """Exit as soon as the parent closes our stdin, or writes anything to it.

    Signals do not work here. ``NSApplication.run()`` blocks inside Objective-C and
    CPython only runs signal handlers between bytecodes, so SIGTERM is simply never
    delivered — measured, not assumed: the process ignored it and had to be SIGKILLed,
    which left the glow on screen for seconds after control was released. An NSTimer
    yielding to the interpreter did not fix it either.

    A pipe does. The parent holds the write end; the moment it closes or dies — for any
    reason, including SIGKILL — this read returns and the indicator disappears. That
    also makes the glow strictly incapable of outliving the session that owns it.

    ``os._exit`` rather than ``sys.exit``: this runs on a worker thread, where
    SystemExit would only end the thread and leave the run loop spinning.
    """
    import contextlib
    import os
    import threading

    def wait() -> None:
        with contextlib.suppress(Exception):
            stream.readline()
        os._exit(0)

    threading.Thread(target=wait, name="arc-overlay-parent-watch", daemon=True).start()


if __name__ == "__main__":
    sys.exit(run())
