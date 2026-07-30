"""Mouse and keyboard control, with a visible indicator and instant takeover.

ARC never delivers a synthetic input event without a blue glow around every display
saying so, and never continues after the user has reclaimed the pointer. Both
properties are enforced in ``session.py`` rather than left to callers to remember.
"""

from arc.control.session import (
    ControlSession,
    ControlState,
    current,
    pointer_position,
    require,
    start,
    stop,
)

__all__ = [
    "ControlSession",
    "ControlState",
    "current",
    "pointer_position",
    "require",
    "start",
    "stop",
]
