"""Finding cameras and reading them without stalling the loop.

Two concerns live here because they are the same concern in practice: *which* device
to open, and how to read it without the main loop paying for it.

Cameras are resolved by **name**, never by a hard-coded index. AVFoundation indices
shift whenever anything is plugged in, unplugged, or macOS decides a nearby iPhone is
now a webcam — which is how the phone camera ends up grabbed instead of the one on the
monitor. Continuity Camera devices are excluded unless asked for by name.

Reading happens on a thread per camera because ``VideoCapture.read()`` blocks. With two
cameras a single loop would halve the frame rate and let the feeds drift out of sync,
so each stream keeps only its newest frame and the loop takes whatever is freshest.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from arc.errors import PlatformError
from arc.log import get_logger

_log = get_logger(__name__)

#: Continuity Camera devices. Never selected, by any route — not by name match and not
#: by an explicit index. A phone is not part of this setup: it drifts in and out of
#: range, it usually lands at index 0 and shifts every other camera along, and having it
#: silently become a tracking camera is the failure this whole name-matching scheme
#: exists to prevent.
_CONTINUITY = ("iphone", "ipad", "continuity")

#: How long to wait for a camera's first frame before calling it dead.
FIRST_FRAME_TIMEOUT = 3.0


def _cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional extra
        raise PlatformError(
            "camera gestures need opencv and mediapipe. Install with: pip install 'arc[camera]'"
        ) from exc
    return cv2


def list_devices() -> list[str]:
    """Camera names in AVFoundation order — the order OpenCV indexes them in."""
    try:
        import AVFoundation
    except ImportError:  # pragma: no cover - non-macOS
        return []
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    return [str(device.localizedName()) for device in devices or []]


def is_continuity(name: str) -> bool:
    """Whether a device name is an iPhone or iPad acting as a camera."""
    lowered = name.lower()
    return any(marker in lowered for marker in _CONTINUITY)


def find_device(match: str) -> int | None:
    """Index of the first camera whose name contains ``match``, case-insensitively.

    Continuity Camera devices are skipped even when the match string names one, so
    there is no configuration that can select a phone.
    """
    if not match:
        return None
    wanted = match.lower()
    for index, name in enumerate(list_devices()):
        if is_continuity(name):
            continue
        if wanted in name.lower():
            return index
    return None


def device_name(index: int) -> str:
    names = list_devices()
    return names[index] if 0 <= index < len(names) else f"index {index}"


def resolve(config: dict[str, Any]) -> tuple[int | None, str]:
    """``(index, human name)`` for a camera config entry.

    An explicit ``index`` wins; otherwise the ``match`` string is looked up by name.
    Either way a Continuity Camera is refused — an index can land on a phone by
    accident, since plugging anything in renumbers the list.
    """
    if config.get("index") is not None:
        index = int(config["index"])
        name = device_name(index)
        if is_continuity(name):
            return None, f"<refusing Continuity Camera at index {index}: {name}>"
        return index, name

    index = find_device(str(config.get("match", "")))
    if index is None:
        return None, f"<{config.get('match')} not connected>"
    return index, device_name(index)


class CameraStream:
    """One camera, read continuously on its own thread."""

    def __init__(
        self, index: int, name: str = "camera", width: int = 1280, height: int = 720
    ) -> None:
        self.index = index
        self.name = name
        self.width = width
        self.height = height
        self._capture: Any = None
        self._frame: Any = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> CameraStream:
        cv2 = _cv2()
        self._capture = cv2.VideoCapture(self.index, cv2.CAP_AVFOUNDATION)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self._capture.isOpened():
            raise PlatformError(
                f"camera {self.name!r} (index {self.index}) would not open. Another "
                "application may already own it, or Camera access is not granted in "
                "System Settings > Privacy & Security > Camera."
            )

        self._running = True
        self._thread = threading.Thread(
            target=self._loop, name=f"arc-camera-{self.name}", daemon=True
        )
        self._thread.start()

        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT
        while time.monotonic() < deadline:
            if self.read() is not None:
                return self
            time.sleep(0.02)
        raise PlatformError(f"camera {self.name!r} opened but delivered no frames")

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._capture.read()
            if ok and frame is not None:
                with self._lock:
                    self._frame = frame
            else:
                time.sleep(0.005)

    def read(self) -> Any:
        """The newest frame, or None before the first one arrives."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
