"""Synthetic mouse and keyboard input.

Every function here calls ``session.require()`` first. That is the enforcement point
for the two promises in ``session.py``: no synthetic event is ever delivered without
the blue glow being up, and none is delivered after the user has taken the pointer
back. Putting the check in the input layer rather than asking callers to remember it
means a new tool cannot accidentally bypass it.

Uses Quartz event taps (``CGEventPost``) rather than AppleScript. AppleScript's
``System Events`` works, but each call spawns a process and takes ~100 ms, which makes
dragging and typing visibly stuttery. Quartz needs the same Accessibility grant and is
roughly a thousand times faster.
"""

from __future__ import annotations

import time
from typing import Any

from arc.control import session as control_session
from arc.errors import ControlError
from arc.log import get_logger

_log = get_logger(__name__)

#: Pause after each synthetic event. Some applications drop events delivered faster
#: than they can process them, and a click that silently does nothing is worse for the
#: agent than one that is slightly slow.
EVENT_DELAY = 0.02

#: Steps used to animate a pointer move. Teleporting the cursor makes hover states and
#: drag targets misfire in many applications, and it is also disconcerting to watch.
MOVE_STEPS = 18

MOUSE_BUTTONS = ("left", "right", "middle")

#: macOS virtual key codes for keys with no printable character.
_SPECIAL_KEYS = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "escape": 53,
    "esc": 53,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "home": 115,
    "end": 119,
    "pageup": 116,
    "pagedown": 121,
    "forwarddelete": 117,
    "f1": 122,
    "f2": 120,
    "f3": 99,
    "f4": 118,
    "f5": 96,
    "f6": 97,
}

_MODIFIERS = {"cmd", "command", "shift", "alt", "option", "ctrl", "control", "fn"}


def _quartz() -> Any:
    """Return the Quartz module, or raise something actionable."""
    try:
        import Quartz
    except ImportError as exc:  # pragma: no cover - non-macOS
        raise ControlError(
            "input control needs pyobjc-framework-Quartz. Install with: pip install 'arc[control]'"
        ) from exc
    return Quartz


def _modifier_flags(quartz: Any, modifiers: list[str]) -> int:
    """Build a CGEventFlags mask from modifier names."""
    flags = 0
    for name in modifiers:
        lowered = name.lower()
        if lowered in ("cmd", "command"):
            flags |= quartz.kCGEventFlagMaskCommand
        elif lowered == "shift":
            flags |= quartz.kCGEventFlagMaskShift
        elif lowered in ("alt", "option"):
            flags |= quartz.kCGEventFlagMaskAlternate
        elif lowered in ("ctrl", "control"):
            flags |= quartz.kCGEventFlagMaskControl
        elif lowered == "fn":
            flags |= quartz.kCGEventFlagMaskSecondaryFn
    return flags


def pointer() -> tuple[float, float]:
    """Return the current pointer position."""
    position = control_session.pointer_position()
    if position is None:
        raise ControlError("could not read the pointer position")
    return position


def move_to(x: float, y: float, *, smooth: bool = True) -> None:
    """Move the pointer, animating so hover and drag targets behave."""
    session = control_session.require()
    quartz = _quartz()

    start = control_session.pointer_position() or (x, y)
    steps = MOVE_STEPS if smooth else 1

    for index in range(1, steps + 1):
        session.check()  # Re-checked every step: a takeover mid-move stops it dead.
        progress = index / steps
        # Ease-out, which looks intentional rather than mechanical.
        eased = 1 - (1 - progress) ** 3
        current_x = start[0] + (x - start[0]) * eased
        current_y = start[1] + (y - start[1]) * eased

        event = quartz.CGEventCreateMouseEvent(
            None, quartz.kCGEventMouseMoved, (current_x, current_y), 0
        )
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        # Tell the session where we put it, so its own motion is not read as a takeover.
        session.note_pointer((current_x, current_y))
        if steps > 1:
            time.sleep(0.008)

    session.note_pointer((x, y))
    time.sleep(EVENT_DELAY)


def click(
    x: float | None = None,
    y: float | None = None,
    *,
    button: str = "left",
    count: int = 1,
) -> None:
    """Click at a position, or where the pointer already is."""
    session = control_session.require()
    quartz = _quartz()

    if button not in MOUSE_BUTTONS:
        raise ControlError(f"unknown mouse button {button!r}; expected one of {MOUSE_BUTTONS}")

    if x is not None and y is not None:
        move_to(x, y)

    position = control_session.pointer_position()
    if position is None:
        raise ControlError("could not read the pointer position")

    down, up, code = {
        "left": (quartz.kCGEventLeftMouseDown, quartz.kCGEventLeftMouseUp, 0),
        "right": (quartz.kCGEventRightMouseDown, quartz.kCGEventRightMouseUp, 1),
        "middle": (quartz.kCGEventOtherMouseDown, quartz.kCGEventOtherMouseUp, 2),
    }[button]

    for index in range(count):
        session.check()
        for kind in (down, up):
            event = quartz.CGEventCreateMouseEvent(None, kind, position, code)
            # Click count must be set for double-clicks to register as such rather
            # than as two unrelated single clicks.
            quartz.CGEventSetIntegerValueField(event, quartz.kCGMouseEventClickState, index + 1)
            quartz.CGEventPost(quartz.kCGHIDEventTap, event)
            time.sleep(0.01)
        time.sleep(EVENT_DELAY)


def drag(from_x: float, from_y: float, to_x: float, to_y: float) -> None:
    """Press at one point, move, and release at another."""
    session = control_session.require()
    quartz = _quartz()

    move_to(from_x, from_y)
    session.check()

    down = quartz.CGEventCreateMouseEvent(None, quartz.kCGEventLeftMouseDown, (from_x, from_y), 0)
    quartz.CGEventPost(quartz.kCGHIDEventTap, down)
    time.sleep(0.05)

    for index in range(1, MOVE_STEPS + 1):
        session.check()
        progress = index / MOVE_STEPS
        current = (
            from_x + (to_x - from_x) * progress,
            from_y + (to_y - from_y) * progress,
        )
        event = quartz.CGEventCreateMouseEvent(None, quartz.kCGEventLeftMouseDragged, current, 0)
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        session.note_pointer(current)
        time.sleep(0.012)

    up = quartz.CGEventCreateMouseEvent(None, quartz.kCGEventLeftMouseUp, (to_x, to_y), 0)
    quartz.CGEventPost(quartz.kCGHIDEventTap, up)
    session.note_pointer((to_x, to_y))
    time.sleep(EVENT_DELAY)


def scroll(amount: int, *, horizontal: bool = False) -> None:
    """Scroll by ``amount`` lines. Positive is up, or right when horizontal."""
    session = control_session.require()
    quartz = _quartz()
    session.check()

    event = quartz.CGEventCreateScrollWheelEvent(
        None,
        quartz.kCGScrollEventUnitLine,
        2,
        0 if horizontal else amount,
        amount if horizontal else 0,
    )
    quartz.CGEventPost(quartz.kCGHIDEventTap, event)
    time.sleep(EVENT_DELAY)


def type_text(text: str) -> None:
    """Type a string.

    Sends each character as a unicode string on the event rather than mapping it to a
    virtual key code. Key codes are keyboard-layout dependent, so a code-based
    implementation types gibberish on a Dvorak or non-US layout; unicode is
    layout-independent.
    """
    session = control_session.require()
    quartz = _quartz()

    for character in text:
        session.check()
        for pressed in (True, False):
            event = quartz.CGEventCreateKeyboardEvent(None, 0, pressed)
            quartz.CGEventKeyboardSetUnicodeString(event, len(character), character)
            quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        # Fast enough to feel like typing, slow enough that applications keep up.
        time.sleep(0.008)


def press(key: str, *modifiers: str) -> None:
    """Press a named key, optionally with modifiers. E.g. ``press("c", "cmd")``."""
    session = control_session.require()
    quartz = _quartz()
    session.check()

    lowered = key.lower()
    if lowered in _SPECIAL_KEYS:
        code = _SPECIAL_KEYS[lowered]
    elif len(key) == 1:
        code = _character_code(key)
    else:
        raise ControlError(
            f"unknown key {key!r}; expected a single character or one of "
            f"{', '.join(sorted(_SPECIAL_KEYS))}"
        )

    flags = _modifier_flags(quartz, list(modifiers))

    for pressed in (True, False):
        event = quartz.CGEventCreateKeyboardEvent(None, code, pressed)
        if flags:
            quartz.CGEventSetFlags(event, flags)
        quartz.CGEventPost(quartz.kCGHIDEventTap, event)
        time.sleep(0.01)
    time.sleep(EVENT_DELAY)


#: US-layout key codes, used only for shortcut keys where the *code* matters (Cmd-C
#: must send the physical C key, not the character "c"). Text typing goes through the
#: layout-independent unicode path above.
_US_CODES = {
    "a": 0,
    "s": 1,
    "d": 2,
    "f": 3,
    "h": 4,
    "g": 5,
    "z": 6,
    "x": 7,
    "c": 8,
    "v": 9,
    "b": 11,
    "q": 12,
    "w": 13,
    "e": 14,
    "r": 15,
    "y": 16,
    "t": 17,
    "1": 18,
    "2": 19,
    "3": 20,
    "4": 21,
    "6": 22,
    "5": 23,
    "9": 25,
    "7": 26,
    "8": 28,
    "0": 29,
    "o": 31,
    "u": 32,
    "i": 34,
    "p": 35,
    "l": 37,
    "j": 38,
    "k": 40,
    "n": 45,
    "m": 46,
}


def _character_code(character: str) -> int:
    """Return the US-layout virtual key code for a character."""
    code = _US_CODES.get(character.lower())
    if code is None:
        raise ControlError(f"no key code for {character!r}")
    return code
