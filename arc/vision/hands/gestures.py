"""Recognise a gesture from one hand's 21 landmarks.

Pure geometry — no camera, no MediaPipe graph — so the rules that decide whether a
hand is a fist can be tested directly rather than by waving at a webcam.

MediaPipe hand landmark indices used here::

    0  wrist
    4  thumb tip        3 thumb IP
    8  index tip        6 index PIP        5 index MCP
    12 middle tip      10 middle PIP       9 middle MCP
    16 ring tip        14 ring PIP
    20 pinky tip       18 pinky PIP

Everything is normalised by hand size (wrist → middle MCP), so the same thresholds
hold whether the hand is at the keyboard or across the desk.

**Order matters**, and every step of it is load-bearing:

* A closed fist also puts the thumb tip near the index tip, so a naive pinch test
  fires on every fist. Fist is decided first.
* A pinch curls the spare fingers, so a loose fist test fires on every pinch. The
  fist test therefore stays strict about how many fingers are curled.
* A pinch usually leaves the middle finger extended, so the two-finger cursor pose
  claims it unless pinch is checked first — which would route a resize to the mouse.

The resulting precedence is fist, pinch, two fingers, open, point.
"""

from __future__ import annotations

import math
from typing import Any

#: Tip-to-tip thumb/index distance, in hand widths, below which the hand is pinching.
#: Raised from the 0.30 this was ported with: at 0.30 the tips had to visibly touch,
#: which is a hard pose to hold steady at arm's length.
PINCH_DISTANCE = 0.36

#: A finger counts as extended when its tip is this much farther from the wrist than
#: its PIP joint, and curled when it is this much closer.
EXTENDED_RATIO = 1.15
CURLED_RATIO = 0.88

#: How many fingers must be curled for the geometric fist test. Kept at all four:
#: relaxing it to three makes a pinch — which curls the spare fingers — read as a
#: fist, and the two drive completely different actions.
FIST_CURLED_FINGERS = 4

#: MediaPipe's own classifications, used to corroborate the geometry. The bundled
#: model already computes these and the original port discarded them.
CANNED_FIST = "Closed_Fist"
CANNED_OPEN = "Open_Palm"
CANNED_TWO = "Victory"

_FINGERS = ((8, 6), (12, 10), (16, 14), (20, 18))  # (tip, pip)


def _distance(a: Any, b: Any) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def recognize(landmarks: list[Any], canned: str | None = None) -> dict[str, Any]:
    """Classify one hand.

    Args:
        landmarks: 21 ``(x, y, z)`` normalised landmarks.
        canned: MediaPipe's own gesture label for this hand, when available. Used
            only to *confirm* a pose the geometry is unsure about, never to override
            geometry that already decided — the canned labels are trained on a
            head-on view and get noisy from the side camera's angle.

    Returns:
        A dict with the discrete ``gesture`` plus the metrics the callers need:
        pinch strength and midpoint for panning, the two-finger spread and midpoint
        for the cursor.
    """
    hand_scale = _distance(landmarks[0], landmarks[9]) + 1e-6  # wrist → middle MCP
    wrist = landmarks[0]
    thumb_tip, index_tip, middle_tip = landmarks[4], landmarks[8], landmarks[12]

    pinch_distance = _distance(thumb_tip, index_tip) / hand_scale
    pinching = pinch_distance < PINCH_DISTANCE
    pinch_strength = max(0.0, min(1.0, 1.0 - pinch_distance / 0.6))

    extended = [
        _distance(landmarks[t], wrist) > _distance(landmarks[p], wrist) * EXTENDED_RATIO
        for t, p in _FINGERS
    ]
    curled = [
        _distance(landmarks[t], wrist) < _distance(landmarks[p], wrist) * CURLED_RATIO
        for t, p in _FINGERS
    ]
    n_extended, n_curled = sum(extended), sum(curled)

    # Index + middle up, ring + pinky down. The thumb is ignored because it sits
    # curled in both the peace sign and the fingers-together sign; the gap between
    # the two fingertips is what tells those apart, and the caller applies
    # hysteresis to it rather than this deciding the split here.
    two_finger = extended[0] and extended[1] and not extended[2] and not extended[3]
    two_spread = _distance(index_tip, middle_tip) / hand_scale if two_finger else None
    two_mid = (
        ((index_tip[0] + middle_tip[0]) / 2.0, (index_tip[1] + middle_tip[1]) / 2.0)
        if two_finger
        else None
    )

    gesture = _classify(
        n_curled=n_curled,
        n_extended=n_extended,
        extended=extended,
        two_finger=two_finger,
        pinching=pinching,
        canned=canned,
    )

    return {
        "gesture": gesture,
        "pinch": gesture == "pinch",
        "pinch_distance": round(pinch_distance, 3),
        "pinch_strength": round(pinch_strength, 3),
        "pinch_point": (
            (thumb_tip[0] + index_tip[0]) / 2.0,
            (thumb_tip[1] + index_tip[1]) / 2.0,
        ),
        "fingers_extended": n_extended,
        "fingers_curled": n_curled,
        "index_tip": index_tip,
        "two_spread": round(two_spread, 3) if two_spread is not None else None,
        "two_mid": two_mid,
    }


def _classify(
    *,
    n_curled: int,
    n_extended: int,
    extended: list[bool],
    two_finger: bool,
    pinching: bool,
    canned: str | None,
) -> str:
    """Pick one label. Split out so the precedence is readable in one screen."""
    # Fist first, and it may be settled by either the geometry or MediaPipe's own
    # classifier. Requiring both was what made a fist so hard to land that the
    # gesture looked broken; requiring either keeps the pose specific — nothing
    # else in this vocabulary curls every finger — while letting a clench that is
    # slightly off-angle for one test be caught by the other.
    if n_curled >= FIST_CURLED_FINGERS or canned == CANNED_FIST:
        return "fist"
    # Pinch before the two-finger pose. Pinching rarely curls the middle finger, so
    # a pinch reads as index-and-middle-extended and the two-finger test claims it
    # first — sending a resize to the cursor. Thumb-and-index contact is the more
    # specific signal, and the cursor poses keep the thumb tucked well clear of it,
    # so nothing is taken the other way.
    if pinching:
        return "pinch"
    if two_finger:
        return "two"
    if n_extended >= 4 or canned == CANNED_OPEN:
        return "open"
    if extended[0] and not any(extended[1:]):
        return "point"
    return "other"


def two_hand_gap(hand_a: dict[str, Any], hand_b: dict[str, Any]) -> float:
    """Distance between two hands' pinch points — the raw signal behind resize/zoom."""
    return _distance(hand_a["pinch_point"], hand_b["pinch_point"])
