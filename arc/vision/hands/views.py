"""One camera plus its recogniser — the per-view readings the fuser merges.

Split from :mod:`arc.vision.hands.fusion` so that module stays camera-free and
testable: everything that needs a device or a MediaPipe graph lives here.
"""

from __future__ import annotations

import math
from typing import Any

from arc.log import get_logger
from arc.vision.hands.cameras import CameraStream, resolve
from arc.vision.hands.gestures import recognize
from arc.vision.hands.tracker import HandTracker

_log = get_logger(__name__)

#: Mirroring the frame swaps which hand is which, so the label has to swap back or
#: the two views disagree about every hand and nothing ever fuses.
_SWAP = {"Left": "Right", "Right": "Left"}


def palm_center(landmarks: list[Any]) -> tuple[float, float]:
    """Wrist plus all four knuckles — steadier than any single landmark."""
    points = [landmarks[i] for i in (0, 5, 9, 13, 17)]
    return (
        sum(p[0] for p in points) / 5.0,
        sum(p[1] for p in points) / 5.0,
    )


def palm_angle(landmarks: list[Any]) -> float:
    """The across-the-knuckles line, which rotates cleanly even with a closed fist."""
    a, b = landmarks[5], landmarks[17]
    return math.atan2(b[1] - a[1], b[0] - a[0])


def palm_span(landmarks: list[Any]) -> float:
    """Apparent hand size. Grows as the hand comes toward the camera; feeds depth."""
    return math.hypot(landmarks[0][0] - landmarks[9][0], landmarks[0][1] - landmarks[9][1])


class CameraView:
    """A camera, its recogniser, and this frame's hands."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        max_hands: int = 2,
        mirror: bool = True,
        track_width: int = 960,
    ) -> None:
        self.role = str(config.get("role", "front"))
        self.label = str(config.get("label", self.role))
        self.mirror = mirror
        self.track_width = track_width
        self.ok = False
        self.frame: Any = None
        self.hands: list[dict[str, Any]] = []
        self.source: CameraStream | None = None
        self.tracker: HandTracker | None = None

        index, name = resolve(config)
        self.device = name
        if index is None:
            _log.info("camera not connected", extra={"role": self.role, "want": name})
            return

        try:
            self.source = CameraStream(
                index,
                self.role,
                int(config.get("width", 1280)),
                int(config.get("height", 720)),
            ).start()
        except Exception as exc:
            _log.warning("camera failed to open: %s", exc, extra={"role": self.role})
            self.source = None
            return

        self.tracker = HandTracker(max_hands=max_hands)
        self.ok = True
        _log.info("camera opened", extra={"role": self.role, "device": name, "index": index})

    def poll(self) -> list[dict[str, Any]]:
        """Read the freshest frame and return this view's hands."""
        if not self.ok or self.source is None or self.tracker is None:
            return []

        import cv2

        frame = self.source.read()
        if frame is None:
            self.hands = []
            return []
        if self.mirror:
            frame = cv2.flip(frame, 1)
        self.frame = frame

        # Track on a downscaled copy: MediaPipe's accuracy is unchanged well below
        # 1280px and the cost is not, which matters with two cameras on a fanless Mac.
        small = frame
        if self.track_width and frame.shape[1] > self.track_width:
            scale = self.track_width / frame.shape[1]
            small = cv2.resize(frame, (self.track_width, int(frame.shape[0] * scale)))

        hands = []
        for detected in self.tracker.process(small):
            landmarks = detected["landmarks"]
            label = detected["label"]
            if self.mirror:
                label = _SWAP.get(label, label)
            hands.append(
                {
                    "role": self.role,
                    "label": label,
                    "landmarks": landmarks,
                    "g": recognize(landmarks, detected.get("canned")),
                    "center": palm_center(landmarks),
                    "angle": palm_angle(landmarks),
                    "span": palm_span(landmarks),
                }
            )
        self.hands = hands
        return hands

    def close(self) -> None:
        if self.source is not None:
            self.source.stop()
            self.source = None
        if self.tracker is not None:
            self.tracker.close()
            self.tracker = None
        self.ok = False
