"""Screen capture.

Uses macOS's built-in ``screencapture``, which needs no dependency and no extra
permission beyond the Screen Recording grant the OS already enforces.

**Downscaling is not optional.** A Retina screenshot is 2940x1912 and costs roughly
6,000 tokens through a vision model — enough to crowd out the conversation it was meant
to inform. §4.3 calls this out specifically, so captures are downscaled before they go
anywhere near a model, and the scale factor is recorded so coordinates can be mapped
back to real screen space.
"""

from __future__ import annotations

import base64
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc.errors import PlatformError
from arc.log import get_logger
from arc.paths import arc_home

_log = get_logger(__name__)

#: Longest edge, in pixels, after downscaling. ~1400 keeps UI text legible to a vision
#: model while cutting a Retina capture to roughly a quarter of its token cost.
MAX_EDGE = 1400

CAPTURE_TIMEOUT = 15.0


@dataclass(frozen=True, slots=True)
class Screenshot:
    """A captured image and what is needed to interpret it."""

    path: Path
    width: int
    height: int
    #: Original dimensions, before downscaling.
    source_width: int
    source_height: int
    display: int = 0
    captured_at: float = 0.0

    @property
    def scale(self) -> float:
        """Factor to multiply image coordinates by to get screen coordinates.

        Load-bearing: a vision model reports a button at (700, 400) in the *downscaled*
        image, and clicking there without scaling lands somewhere else entirely.
        """
        return self.source_width / self.width if self.width else 1.0

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Map a coordinate in this image back to screen space."""
        factor = self.scale
        return (x * factor, y * factor)

    def as_base64(self) -> str:
        """Return the image encoded for a vision model."""
        return base64.b64encode(self.path.read_bytes()).decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "source": [self.source_width, self.source_height],
            "scale": round(self.scale, 3),
            "display": self.display,
        }


def capture_dir() -> Path:
    """Where screenshots are written."""
    target = arc_home() / "screenshots"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _run(command: list[str]) -> None:
    """Run a capture command, raising something actionable on failure."""
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=CAPTURE_TIMEOUT, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PlatformError(f"screen capture failed: {exc}") from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        # The overwhelmingly common cause, and one the user must fix interactively.
        if "not authorized" in stderr.lower() or "permission" in stderr.lower():
            raise PlatformError(
                "screen capture is not permitted. Grant Screen Recording to your "
                "terminal in System Settings > Privacy & Security > Screen Recording, "
                "then restart the terminal."
            )
        raise PlatformError(f"screen capture failed: {stderr or 'unknown error'}")


def _dimensions(path: Path) -> tuple[int, int]:
    """Read an image's pixel dimensions via ``sips``."""
    try:
        result = subprocess.run(
            ["/usr/bin/sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return (0, 0)

    width = height = 0
    for line in result.stdout.splitlines():
        if "pixelWidth:" in line:
            width = int(line.split(":")[-1].strip())
        elif "pixelHeight:" in line:
            height = int(line.split(":")[-1].strip())
    return (width, height)


def _downscale(path: Path, max_edge: int) -> None:
    """Shrink an image in place so its longest edge is at most ``max_edge``."""
    subprocess.run(
        ["/usr/bin/sips", "--resampleHeightWidthMax", str(max_edge), str(path)],
        capture_output=True,
        text=True,
        timeout=20.0,
        check=False,
    )


def capture(
    *,
    display: int = 0,
    region: tuple[int, int, int, int] | None = None,
    max_edge: int = MAX_EDGE,
    name: str | None = None,
) -> Screenshot:
    """Capture a screen or a region of one.

    Args:
        display: Which display, 0-based.
        region: ``(x, y, width, height)`` to crop to, in screen points.
        max_edge: Downscale so the longest edge is at most this many pixels.
        name: Filename stem; defaults to a timestamp.
    """
    target = capture_dir() / f"{name or f'screen-{int(time.time() * 1000)}'}.png"

    command = ["/usr/sbin/screencapture", "-x"]  # -x: no shutter sound
    if region is not None:
        x, y, width, height = region
        command += ["-R", f"{x},{y},{width},{height}"]
    else:
        command += ["-D", str(display + 1)]  # screencapture displays are 1-based
    command.append(str(target))

    _run(command)
    if not target.is_file():
        raise PlatformError("screen capture produced no file")

    source_width, source_height = _dimensions(target)
    if max_edge and max(source_width, source_height) > max_edge:
        _downscale(target, max_edge)
    width, height = _dimensions(target)

    _log.info(
        "captured screen",
        extra={"display": display, "size": f"{width}x{height}", "path": str(target)},
    )
    return Screenshot(
        path=target,
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
        display=display,
        captured_at=time.time(),
    )


def displays() -> list[dict[str, Any]]:
    """Return the attached displays and their geometry."""
    try:
        import AppKit
    except ImportError:  # pragma: no cover - non-macOS
        return []

    found = []
    for index, screen in enumerate(AppKit.NSScreen.screens()):
        frame = screen.frame()
        found.append(
            {
                "index": index,
                "width": int(frame.size.width),
                "height": int(frame.size.height),
                "x": int(frame.origin.x),
                "y": int(frame.origin.y),
                "primary": index == 0,
            }
        )
    return found
