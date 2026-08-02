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

#: Ceiling for the active display list query. Far more than any real desk.
MAX_DISPLAYS = 16


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
    #: Top-left corner of this capture in global screen space. Non-zero for any display
    #: but the primary one, and for a region capture. Without it, a coordinate read off
    #: a secondary display maps to the same spot on the *primary* one.
    origin_x: float = 0.0
    origin_y: float = 0.0
    #: Size of the captured area in screen *points*, which is the unit clicks are
    #: delivered in. On a Retina display this is half the pixel size. Zero means
    #: unknown, in which case points are assumed to equal pixels.
    screen_width: float = 0.0
    screen_height: float = 0.0

    @property
    def scale(self) -> float:
        """Factor from image pixels to source-image pixels — how much was downscaled."""
        return self.source_width / self.width if self.width else 1.0

    @property
    def point_scale(self) -> float:
        """Factor from image pixels to screen points. The one that matters for clicking.

        Distinct from :attr:`scale` because a Retina capture has two independent
        conversions stacked on it: the downscale this class applied, and the display's
        own backing scale factor. ``screencapture`` yields *pixels*, so a 1470-point
        display produces a 2940-pixel image; multiplying an image coordinate by
        :attr:`scale` alone lands twice as far right and twice as far down as intended.
        """
        if not self.width or not self.screen_width:
            return self.scale
        return self.screen_width / self.width

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        """Map a coordinate in this image back to global screen space.

        Two corrections, both load-bearing. The scale undoes the downscaling *and* the
        display's backing scale factor, landing in points. The origin shift places the
        result on the display it actually came from: a capture of a secondary display
        starts at (0, 0) in its own image, but that pixel may be 1470 points to the
        right in the space clicks are delivered in.
        """
        factor = self.point_scale
        return (self.origin_x + x * factor, self.origin_y + y * factor)

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
        # A region is already stated in global points, so it is its own origin and size.
        origin = (float(x), float(y))
        point_size = (float(width), float(height))
    else:
        command += ["-D", str(display + 1)]  # screencapture displays are 1-based
        origin, point_size = _geometry_of(display)
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
        origin_x=origin[0],
        origin_y=origin[1],
        screen_width=point_size[0],
        screen_height=point_size[1],
    )


def _geometry_of(display: int) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return a display's ``(origin, size)`` in global screen points.

    Falls back to zeroes rather than raising: a capture is still a usable image without
    geometry, and :func:`capture` is the read-only path that must keep working even
    when the window server cannot be queried. A zero size makes ``point_scale`` fall
    back to the pixel scale, which is correct on a non-Retina display and no worse than
    the previous behaviour anywhere else.
    """
    bounds = _display_bounds()
    if 0 <= display < len(bounds):
        x, y, width, height = bounds[display]
        return ((x, y), (width, height))
    return ((0.0, 0.0), (0.0, 0.0))


def displays() -> list[dict[str, Any]]:
    """Return the attached displays and their geometry, in global screen space.

    Uses ``CGDisplayBounds`` rather than ``NSScreen.frame()`` on purpose. NSScreen
    measures from the bottom-left of the primary display with y running upward, so a
    taller secondary monitor reports a *negative* origin; clicks, the accessibility
    tree, and ``CGEventPost`` all use top-left with y running down. Reporting the
    AppKit numbers meant every coordinate on a secondary display was quietly wrong by
    the difference between the two conventions.

    The ordering is also the one ``screencapture -D`` uses, so an index here selects
    the same display in :func:`capture`.
    """
    bounds = _display_bounds()
    return [
        {
            "index": index,
            "width": int(width),
            "height": int(height),
            "x": int(x),
            "y": int(y),
            "primary": (x, y) == (0.0, 0.0),
        }
        for index, (x, y, width, height) in enumerate(bounds)
    ]


def _display_bounds() -> list[tuple[float, float, float, float]]:
    """Return ``(x, y, width, height)`` per display, top-left origin, capture order."""
    try:
        import Quartz
    except ImportError:  # pragma: no cover - non-macOS
        return []

    try:
        error, display_ids, _count = Quartz.CGGetActiveDisplayList(MAX_DISPLAYS, None, None)
        if error != 0 or not display_ids:
            return []
        found = []
        for display_id in display_ids:
            rect = Quartz.CGDisplayBounds(display_id)
            found.append(
                (
                    float(rect.origin.x),
                    float(rect.origin.y),
                    float(rect.size.width),
                    float(rect.size.height),
                )
            )
        return found
    except Exception:  # pragma: no cover - defensive
        return []
