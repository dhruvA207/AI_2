"""Screen tools for the agent.

Read-only, so they still run under ``--dry-run``: looking at the screen changes
nothing. Taking *control* of it is a separate category, in ``input_control.py``.
"""

from __future__ import annotations

from arc.errors import PlatformError, ToolError
from arc.log import get_logger
from arc.tools.registry import tool
from arc.vision import capture as capture_screen
from arc.vision import displays as list_displays

_log = get_logger(__name__)


@tool(category="screen")
def screenshot(display: int = 0) -> str:
    """Capture a screen and return where the image was saved.

    Args:
        display: Which display to capture, 0-based.
    """
    try:
        shot = capture_screen(display=display)
    except PlatformError as exc:
        raise ToolError(str(exc)) from exc

    return (
        f"captured display {display}: {shot.width}x{shot.height} "
        f"(from {shot.source_width}x{shot.source_height}, scale {shot.scale:.2f})\n"
        f"saved to {shot.path}"
    )


@tool(category="screen")
def screen_layout() -> str:
    """List the attached displays and their sizes and positions."""
    found = list_displays()
    if not found:
        return "no displays detected"
    return "\n".join(
        f"display {d['index']}: {d['width']}x{d['height']} at ({d['x']}, {d['y']})"
        + (" [primary]" if d["primary"] else "")
        for d in found
    )
