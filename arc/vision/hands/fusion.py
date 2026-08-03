"""Merge what two cameras say about the same hand.

The desk this is built for: a monitor straight ahead with a webcam on top seeing the
hands head-on (the **front** view), and the laptop off to one side seeing the same
hands roughly edge-on (the **side** view). The ~90-degree separation buys two things:

* **Speed.** A gesture both cameras call the same thing is almost never a false
  positive, so it can commit in a couple of frames instead of holding for a dozen.
* **Depth.** One camera cannot tell how far away a hand is. Apparent hand size across
  both views can.

Everything here is camera-free so the merge rules are testable. :mod:`arc.vision.hands.cameras`
supplies the per-view readings.

Why confidence works the way it does
------------------------------------

The version this was ported from scored a hand the two cameras *disagreed* about at
0.55 and then dropped anything below 0.70. That silently deleted the two gestures the
side camera is worst at judging: a fist and a pinch seen edge-on are heavily
foreshortened, so the side view dissented almost every frame and neither gesture ever
committed. Cursor control kept working only because it read one camera's raw hands and
never went through this path at all — which is exactly the shape of the bug report.

So agreement now *accelerates* a gesture rather than gating it. The front camera is the
authority on what the hand is doing, because it is the one the hands are pointed at,
and a reading it is confident about survives the side camera's dissent; the side camera
still slows that reading down via :class:`Gate`, and still owns depth. A hand only the
side camera can see stays below the bar, which is the one case where the geometry
really is unreliable.
"""

from __future__ import annotations

import math
from typing import Any

FRONT = "front"
SIDE = "side"

#: Confidence when both views agree — the strongest evidence available.
CONF_AGREED = 1.0
#: Confidence when the front view sees the hand and the side view dissents or is
#: absent. Above :data:`DEFAULT_MIN_CONF` on purpose: see the module docstring.
CONF_FRONT = 0.8
#: Confidence when only the side view sees the hand. Below the bar — an edge-on
#: reading with nothing to check it against is the case worth discarding.
CONF_SIDE_ONLY = 0.5

DEFAULT_MIN_CONF = 0.7


class Gate:
    """Hold a candidate gesture until it has been seen enough consecutive frames.

    This is where cross-camera agreement earns its keep: an agreed gesture commits
    almost immediately, a contested one has to be held. Being slower to trust is the
    right response to disagreement — dropping the gesture outright was not.
    """

    def __init__(self, agree: int = 2, solo: int = 5) -> None:
        self.agree = agree
        self.solo = solo
        self.committed = "other"
        self._candidate: str | None = None
        self._count = 0

    def update(self, candidate: str | None, agreed: bool) -> str:
        needed = self.agree if agreed else self.solo
        if candidate != self._candidate:
            self._candidate, self._count = candidate, 1
        else:
            self._count += 1
        if self._count >= needed and candidate is not None:
            self.committed = candidate
        return self.committed


class Depth:
    """Depth from apparent hand *size* rather than image position.

    Reading depth off the side camera's horizontal axis conflates two motions:
    reaching forward and sliding sideways both move the hand along it, so lateral
    movement reads as depth. Apparent size does not have that problem — distance goes
    as roughly 1/size, so log(span) rises smoothly as the hand approaches and is
    unchanged by sideways motion.

    Each camera has its own constant offset (different field of view and distance), so
    the side view is bias-corrected against the front before averaging. That keeps the
    estimate continuous when one view drops out instead of jumping.

    Larger ``z`` means the hand is closer.
    """

    def __init__(self, ema: float = 0.15, dead: float = 0.01) -> None:
        self.ema = ema
        self.dead = dead
        self.z: float | None = None
        self.dz = 0.0
        self._raw: float | None = None
        self.bias: float | None = None

    def update(self, span_front: float | None, span_side: float | None) -> float | None:
        z_front = math.log(max(span_front, 1e-4)) if span_front else None
        z_side = math.log(max(span_side, 1e-4)) if span_side else None

        if z_front is not None and z_side is not None:
            offset = z_side - z_front
            self.bias = offset if self.bias is None else self.bias + (offset - self.bias) * 0.05
            raw = (z_front + (z_side - self.bias)) / 2.0
        elif z_front is not None:
            raw = z_front
        elif z_side is not None:
            raw = z_side - (self.bias or 0.0)
        else:
            self.dz = 0.0
            return self.z

        self._raw = raw if self._raw is None else self._raw + (raw - self._raw) * self.ema
        if self.z is None:
            self.z, self.dz = self._raw, 0.0
        elif abs(self._raw - self.z) > self.dead:  # dead zone kills the flicker
            self.dz = self._raw - self.z
            self.z = self._raw
        else:
            self.dz = 0.0
        return self.z


def score(
    front: dict[str, Any] | None,
    side: dict[str, Any] | None,
    *,
    dual: bool = True,
) -> tuple[str | None, float, bool]:
    """Return ``(gesture, confidence, agreed)`` for one hand seen by either view.

    Args:
        front: The head-on view's reading, or None if it did not see this hand.
        side: The edge-on view's reading, or None.
        dual: Whether two cameras are actually running. This is about the *setup*, not
            about which views saw the hand this frame.

    ``dual`` is load-bearing. A side-only reading is discounted because a head-on
    camera was watching and did not corroborate it, which usually means a spurious
    detection. With one camera running there is nothing suspicious about it being the
    only source — it is simply the evidence available, and which role the config
    happens to label it is irrelevant. Scoring it low anyway meant unplugging the
    webcam silently disabled fist and pinch while the cursor kept working, because the
    cursor reads a view's raw hands and never comes through here.
    """
    front_gesture = front["g"]["gesture"] if front else None
    side_gesture = side["g"]["gesture"] if side else None

    if not dual:
        seen = front if front is not None else side
        gesture = front_gesture if front is not None else side_gesture
        return (gesture, CONF_FRONT, False) if seen is not None else (None, 0.0, False)

    if front is not None and side is not None:
        agreed = front_gesture == side_gesture
        return front_gesture, (CONF_AGREED if agreed else CONF_FRONT), agreed
    if front is not None:
        return front_gesture, CONF_FRONT, False
    return side_gesture, CONF_SIDE_ONLY, False


class Fuser:
    """Per-hand gate and depth state, applied to each frame's readings."""

    def __init__(
        self,
        *,
        min_conf: float = DEFAULT_MIN_CONF,
        confirm_agree: int = 2,
        confirm_solo: int = 5,
        depth_ema: float = 0.15,
        depth_dead: float = 0.01,
        dual: bool = True,
    ) -> None:
        #: Whether two cameras are running. See :func:`score` — with one, its reading
        #: is the authority whichever role the config gave it.
        self.dual = dual
        self.min_conf = min_conf
        self.confirm_agree = confirm_agree
        self.confirm_solo = confirm_solo
        self.depth_ema = depth_ema
        self.depth_dead = depth_dead
        self._gates: dict[str, Gate] = {}
        self._depths: dict[str, Depth] = {}

    def _gate(self, label: str) -> Gate:
        if label not in self._gates:
            self._gates[label] = Gate(self.confirm_agree, self.confirm_solo)
        return self._gates[label]

    def _depth(self, label: str) -> Depth:
        if label not in self._depths:
            self._depths[label] = Depth(self.depth_ema, self.depth_dead)
        return self._depths[label]

    def fuse(
        self, front: dict[str, dict[str, Any]], side: dict[str, dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge per-label readings from both views into one list of hands.

        Args:
            front: Front-view hands keyed by handedness label.
            side: Side-view hands keyed by handedness label.

        Returns:
            Fused hands, most confident first.
        """
        fused = []
        for label in set(front) | set(side):
            f, s = front.get(label), side.get(label)
            candidate, confidence, agreed = score(f, s, dual=self.dual)
            gesture = self._gate(label).update(candidate, agreed)

            base = f or s
            if base is None:  # pragma: no cover - a label exists only if a view saw it
                continue
            depth = self._depth(label)
            z = depth.update(f["span"] if f else None, s["span"] if s else None)

            fused.append(
                {
                    "label": label,
                    "gesture": gesture,
                    "raw_front": f["g"]["gesture"] if f else None,
                    "raw_side": s["g"]["gesture"] if s else None,
                    "conf": confidence,
                    "agreed": agreed,
                    "seen_by": [n for n, v in ((FRONT, f), (SIDE, s)) if v is not None],
                    "x": base["center"][0],
                    "y": base["center"][1],
                    "z": z,
                    "dz": depth.dz,
                    "angle": base["angle"],
                    "span": base["span"],
                    "pinch_point": base["g"]["pinch_point"],
                    "g": base["g"],
                }
            )
        fused.sort(key=lambda h: -h["conf"])
        return fused

    def confident(self, hands: list[dict[str, Any]], gesture: str) -> list[dict[str, Any]]:
        """Hands showing ``gesture`` with enough confidence to act on."""
        return [h for h in hands if h["gesture"] == gesture and h["conf"] >= self.min_conf]
