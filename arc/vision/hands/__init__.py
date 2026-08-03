"""Hand-gesture control of windows and the cursor, from one or two cameras.

  fist (one hand)             -> move the frontmost window
  pinch with BOTH hands       -> resize it
  two fingers together        -> move the cursor
  two fingers spread (peace)  -> left click

Imports nothing heavy at module level: ``arc`` has to keep importing on a machine
without MediaPipe or OpenCV, and the tool layer reports a clear error when the extra is
missing rather than letting an ImportError surface from three frames down.
"""

from __future__ import annotations

from arc.vision.hands.fusion import Fuser, Gate, score
from arc.vision.hands.gestures import recognize, two_hand_gap
from arc.vision.hands.pointer import PointerController, map_to_screen

__all__ = [
    "Fuser",
    "Gate",
    "PointerController",
    "map_to_screen",
    "recognize",
    "score",
    "two_hand_gap",
]
