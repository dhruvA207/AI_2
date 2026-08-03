"""MediaPipe hand landmarks, one recogniser per camera.

A recogniser carries per-stream tracking state, so two cameras sharing one would have
each frame invalidate the other's history. One graph per view.

Runs in VIDEO mode, which needs strictly increasing timestamps — hence the counter
rather than passing the wall clock straight through.

For each hand this returns the 21 landmarks, the handedness label, and MediaPipe's own
gesture classification. That last one is free (the bundled model computes it either
way) and :mod:`arc.vision.hands.gestures` uses it to corroborate a fist, which is the
pose the geometry alone is worst at.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from arc.errors import PlatformError
from arc.log import get_logger
from arc.paths import arc_home

_log = get_logger(__name__)

#: Where the bundled gesture model is expected to live.
MODEL_NAME = "gesture_recognizer.task"


def model_path() -> Path:
    """Locate the gesture model, or say exactly how to get it."""
    path = arc_home() / "models" / MODEL_NAME
    if path.is_file():
        return path
    raise PlatformError(
        f"the hand gesture model is missing from {path}. Download it with:\n"
        f"  mkdir -p {path.parent} && curl -L -o {path} \\\n"
        "    https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task"
    )


class HandTracker:
    """Wraps one MediaPipe gesture recogniser."""

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        max_hands: int = 2,
        detection_confidence: float = 0.5,
        tracking_confidence: float = 0.5,
    ) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - optional extra
            raise PlatformError(
                "camera gestures need mediapipe. Install with: pip install 'arc[camera]'"
            ) from exc

        self._vision = vision
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model or model_path())),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._timestamp = 0

    def process(self, frame_bgr: Any) -> list[dict[str, Any]]:
        """Detect hands in one BGR frame."""
        import cv2
        import mediapipe as mp

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # VIDEO mode rejects a timestamp that does not advance.
        self._timestamp = max(self._timestamp + 1, int(time.time() * 1000))
        result = self._recognizer.recognize_for_video(image, self._timestamp)

        hands = []
        for index, landmarks in enumerate(result.hand_landmarks or []):
            label = "?"
            if result.handedness and index < len(result.handedness) and result.handedness[index]:
                label = result.handedness[index][0].category_name
            canned = None
            if result.gestures and index < len(result.gestures) and result.gestures[index]:
                canned = result.gestures[index][0].category_name
            hands.append(
                {
                    "label": label,
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in landmarks],
                    "canned": canned,
                }
            )
        return hands

    def draw_on(self, frame_bgr: Any, hands: list[dict[str, Any]]) -> None:
        """Draw the skeleton, so the preview shows what is actually being tracked."""
        import cv2

        height, width = frame_bgr.shape[:2]
        connections = self._vision.HandLandmarksConnections.HAND_CONNECTIONS
        for hand in hands:
            points = [(int(x * width), int(y * height)) for (x, y, _z) in hand["landmarks"]]
            for connection in connections:
                cv2.line(
                    frame_bgr, points[connection.start], points[connection.end], (0, 230, 120), 2
                )
            for point in points:
                cv2.circle(frame_bgr, point, 4, (0, 180, 255), -1)

    def close(self) -> None:
        self._recognizer.close()
