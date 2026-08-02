"""The control indicator — a blue glow around the screen while ARC has control.

When ARC is driving the mouse and keyboard, that must be **unmistakable**. An agent
moving the pointer with no visible signal is alarming and, worse, ambiguous: you cannot
tell a stuck agent from a stuck machine.

Four properties make this safe rather than decorative:

- **Click-through.** ``ignoresMouseEvents`` is set on every window. The glow can never
  intercept a click, so it cannot make the machine less usable than it was — which
  would defeat the purpose of an indicator meant to reassure.
- **Every display.** One window per screen, joining all Spaces. A glow on the laptop
  while ARC clicks on the external monitor would be worse than none.
- **A way out that is always on screen.** Alongside the glow, a small panel shows the
  abort phrase. Type ``arc-kill`` anywhere, with nothing focused and without touching
  the mouse, and every ARC process is killed. See :func:`_install_key_watcher`.
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

from arc.control.events import ARC_EVENT_TAG

#: ARC's accent. A soft blue that reads clearly on light and dark desktops without
#: looking like a system error, which red or amber would.
ACCENT = (0.29, 0.62, 1.0)  # #4A9EFF

#: How far the bloom reaches inward, in points. Wide enough to register in peripheral
#: vision — the indicator is worthless if you have to look for it.
GLOW_WIDTH = 34.0

#: Alpha at the outermost ring. High enough to hold its own against a blue desktop,
#: which an earlier, fainter value did not.
PEAK_ALPHA = 0.9

#: Corner rounding, roughly matching macOS display corners.
CORNER_RADIUS = 14.0

#: Typing this anywhere aborts ARC. Chosen to match the ``arc-kill`` command, so there
#: is one thing to remember rather than two.
KILL_PHRASE = "arc-kill"

#: Kill panel geometry, in points.
BOX_WIDTH = 316.0
BOX_HEIGHT = 74.0
BOX_MARGIN_BOTTOM = 64.0

#: Abandon a half-typed phrase after this many seconds, so stray keystrokes minutes
#: apart cannot accumulate into an accidental abort.
TYPING_RESET_SECONDS = 6.0


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

            steps = 30
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
                # Bands overlap deliberately. Hairlines with gaps between them read as
                # a set of thin rings rather than as one continuous glow.
                path.setLineWidth_(3.0)
                appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                    ACCENT[0], ACCENT[1], ACCENT[2], alpha
                ).set()
                path.stroke()

        def isOpaque(self) -> bool:
            return False

    return GlowView


def _make_kill_box_view(appkit: Any, state: dict[str, Any]) -> Any:
    """Build the view for the abort panel.

    Progress lives in ``state`` rather than on the instance because attribute storage
    on a PyObjC subclass is fiddly, and a plain dict closed over here is both simpler
    and easier to drive from the key watcher.
    """

    class KillBoxView(appkit.NSView):  # type: ignore[misc]
        """A translucent panel showing the abort phrase and how much of it is typed."""

        def drawRect_(self, _rect: Any) -> None:
            bounds = self.bounds()
            appkit.NSColor.clearColor().set()
            appkit.NSRectFill(bounds)

            panel = appkit.NSInsetRect(bounds, 2.0, 2.0)
            path = appkit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(panel, 12.0, 12.0)

            # Dark and translucent rather than opaque: the panel has to be legible on
            # any desktop without hiding whatever it happens to cover.
            appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                0.04, 0.06, 0.10, 0.72
            ).setFill()
            path.fill()

            # The same accent as the glow, so the two read as one system.
            appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
                ACCENT[0], ACCENT[1], ACCENT[2], 0.95
            ).setStroke()
            path.setLineWidth_(2.0)
            path.stroke()

            typed = str(state.get("typed", ""))
            if state.get("stopping"):
                self._draw_centred("stopping ARC…", 26.0, 15.0, _accent(appkit, 1.0))
                return

            self._draw_centred("ARC is controlling this Mac", 46.0, 11.5, _dim(appkit))
            self._draw_phrase(typed)

        def _draw_phrase(self, typed: str) -> None:
            """Render the phrase, with what is typed so far lit up."""
            done = KILL_PHRASE[: len(typed)]
            rest = KILL_PHRASE[len(typed) :]
            font = _mono(appkit, 19.0)

            lit = _attributed(appkit, done, font, _accent(appkit, 1.0))
            pending = _attributed(appkit, rest, font, _dim(appkit))
            total = lit.size().width + pending.size().width

            x = (self.bounds().size.width - total) / 2.0
            y = 16.0
            lit.drawAtPoint_((x, y))
            pending.drawAtPoint_((x + lit.size().width, y))

        def _draw_centred(self, text: str, y: float, size: float, colour: Any) -> None:
            drawn = _attributed(appkit, text, _mono(appkit, size), colour)
            x = (self.bounds().size.width - drawn.size().width) / 2.0
            drawn.drawAtPoint_((x, y))

        def isOpaque(self) -> bool:
            return False

    return KillBoxView


def _mono(appkit: Any, size: float) -> Any:
    """A monospaced font, so the phrase does not reflow as it lights up."""
    try:
        return appkit.NSFont.monospacedSystemFontOfSize_weight_(size, 0.0)
    except AttributeError:  # pragma: no cover - very old macOS
        return appkit.NSFont.systemFontOfSize_(size)


def _accent(appkit: Any, alpha: float) -> Any:
    return appkit.NSColor.colorWithCalibratedRed_green_blue_alpha_(
        ACCENT[0], ACCENT[1], ACCENT[2], alpha
    )


def _dim(appkit: Any) -> Any:
    return appkit.NSColor.colorWithCalibratedWhite_alpha_(0.72, 0.55)


def _attributed(appkit: Any, text: str, font: Any, colour: Any) -> Any:
    return appkit.NSAttributedString.alloc().initWithString_attributes_(
        text,
        {
            appkit.NSFontAttributeName: font,
            appkit.NSForegroundColorAttributeName: colour,
        },
    )


def _build_window(frame: Any, appkit: Any, view_class: Any) -> Any:
    """Create one click-through overlay window at a frame in global screen space.

    Deliberately passes no ``screen:`` argument. That initialiser interprets the
    content rect relative to the *given screen's* lower-left corner, not globally, so
    handing it a global origin offsets a secondary display's window by its own origin
    twice — which put the external monitor's glow entirely off the side of the screen
    and left that display with no indicator at all.
    """
    window = appkit.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        frame,
        appkit.NSWindowStyleMaskBorderless,
        appkit.NSBackingStoreBuffered,
        False,
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
    # setFrame after the fact, so the window lands exactly where asked regardless of
    # any constraining AppKit applies during initialisation.
    window.setFrame_display_(frame, True)
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

    # Reading the CGEvent behind an NSEvent hands back an unbridged pointer, which
    # PyObjC warns about on every keystroke. The value is used read-only, for one
    # integer field, so the warning is noise on a hot path.
    _silence_pointer_warnings()

    application = AppKit.NSApplication.sharedApplication()
    # Accessory, not regular: no Dock icon, no menu bar, never steals focus.
    application.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)

    screens = list(AppKit.NSScreen.screens())
    if not screens:
        print("no screens found", file=sys.stderr)
        return 1

    glow_class = _make_glow_view(AppKit)
    windows = [_build_window(screen.frame(), AppKit, glow_class) for screen in screens]

    state: dict[str, Any] = {"typed": "", "stopping": False, "last_key": 0.0}
    box_view, box_window = _build_kill_box(AppKit, screens[0], state)
    windows.append(box_window)

    _install_key_watcher(AppKit, state, box_view)
    _exit_when_parent_goes(sys.stdin)

    application.run()
    return 0


def _build_kill_box(appkit: Any, screen: Any, state: dict[str, Any]) -> tuple[Any, Any]:
    """Place the abort panel at the bottom centre of the primary display."""
    frame = screen.frame()
    rect = appkit.NSMakeRect(
        frame.origin.x + (frame.size.width - BOX_WIDTH) / 2.0,
        frame.origin.y + BOX_MARGIN_BOTTOM,
        BOX_WIDTH,
        BOX_HEIGHT,
    )
    window = _build_window(rect, appkit, _make_kill_box_view(appkit, state))
    return window.contentView(), window


def _install_key_watcher(appkit: Any, state: dict[str, Any], view: Any) -> None:
    """Watch the keyboard globally for the abort phrase.

    A *global* monitor observes without consuming, and needs no focus. Both matter. A
    panel you had to click into would be useless here twice over: reaching it means
    moving the mouse, which already ends the session, and focusing it would steal the
    keyboard from whatever ARC is driving. This way the phrase can be typed at any
    moment, into whatever happens to be frontmost.

    The keystrokes do still reach that frontmost application — this observes, it does
    not intercept — so aborting mid-task may leave the letters in a text field. That is
    a deliberate trade: a watcher that swallowed keys would need an event tap that can
    fail closed and wedge the keyboard, which is a far worse failure than stray text.

    Requires Accessibility, which any control session already requires.
    """
    import time

    # Modifiers that mean the keystroke was a shortcut, not typing. Without this,
    # Cmd-A counts as the "a" of the phrase.
    shortcut_mask = (
        appkit.NSEventModifierFlagCommand
        | appkit.NSEventModifierFlagControl
        | appkit.NSEventModifierFlagOption
    )

    def on_key(event: Any) -> None:
        if state.get("stopping"):
            return
        if event.modifierFlags() & shortcut_mask:
            return
        if _is_arcs_own(event):
            return

        characters = str(event.charactersIgnoringModifiers() or "")
        if consume(state, characters, time.monotonic()):
            view.setNeedsDisplay_(True)
            _fire_kill_switch()
            return

        view.setNeedsDisplay_(True)

    # Retained on the module: a dropped monitor stops delivering silently.
    global _KEY_MONITOR
    _KEY_MONITOR = appkit.NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(
        appkit.NSEventMaskKeyDown, on_key
    )


def _silence_pointer_warnings() -> None:
    """Drop PyObjC's unbridged-pointer warning, which would fire on every keystroke."""
    import warnings

    try:
        import objc

        warnings.filterwarnings("ignore", category=objc.ObjCPointerWarning)
    except (ImportError, AttributeError):  # pragma: no cover - defensive
        pass


def _is_arcs_own(event: Any) -> bool:
    """Whether this keystroke was synthesised by ARC rather than typed by a person.

    Without this an agent that types the phrase into a document aborts its own
    session. See :mod:`arc.control.events`.
    """
    try:
        native = event.CGEvent()
        if native is None:
            return False
        import Quartz

        tag = Quartz.CGEventGetIntegerValueField(native, Quartz.kCGEventSourceUserData)
        return bool(tag == ARC_EVENT_TAG)
    except Exception:  # pragma: no cover - defensive
        return False


def consume(state: dict[str, Any], characters: str, now: float) -> bool:
    """Fold keystrokes into ``state``; return True when the phrase has been completed.

    Kept free of AppKit so the matching rules — the part with the interesting edge
    cases — can be tested without a window server.
    """
    if state.get("stopping"):
        return False

    if now - float(state.get("last_key", 0.0)) > TYPING_RESET_SECONDS:
        state["typed"] = ""
    state["last_key"] = now

    for character in characters.lower():
        state["typed"] = _advance(str(state.get("typed", "")), character)

    if state.get("typed") == KILL_PHRASE:
        state["stopping"] = True
        return True
    return False


#: Keeps the global key monitor alive for the life of the process.
_KEY_MONITOR: Any = None


def _advance(typed: str, character: str) -> str:
    """Return the new matched prefix of :data:`KILL_PHRASE` after ``character``.

    Restarting on a mismatch — rather than clearing outright — means a fumbled attempt
    like "arcarc-kill" still gets there.
    """
    candidate = typed + character
    if KILL_PHRASE.startswith(candidate):
        return candidate
    if KILL_PHRASE.startswith(character):
        return character
    return ""


def _fire_kill_switch() -> None:
    """Run the real kill switch, then exit.

    Shells out to ``arc.audit.killswitch`` as a separate process rather than calling it
    in-process, for the same reason ``arc-kill`` exists as its own command: the moment
    you need it is the moment ARC may be wedged, and the abort path should share as
    little as possible with whatever is failing.
    """
    import contextlib
    import os
    import pathlib
    import subprocess
    import threading

    # The installed console script when it is there, the module otherwise, so this
    # works from a checkout as well as from an install.
    script = pathlib.Path(sys.executable).with_name("arc-kill")
    command = [str(script)] if script.is_file() else [sys.executable, "-m", "arc.audit.killswitch"]

    def stop() -> None:
        with contextlib.suppress(Exception):
            subprocess.run(command, capture_output=True, timeout=10.0, check=False)
        # Long enough for "stopping ARC…" to register, short enough to feel immediate.
        # Also covers the standalone case, where nothing else will close our stdin.
        os._exit(0)

    threading.Thread(target=stop, name="arc-overlay-kill", daemon=True).start()


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
