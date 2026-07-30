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


@tool(category="screen")
def read_screen_elements(query: str = "") -> str:
    """List the interactive elements on screen, from the accessibility tree.

    Prefer this over a screenshot: it reports exactly what each control is and where,
    as the application itself declares it, so clicking is reliable rather than a guess
    at pixel coordinates.

    Args:
        query: Only return elements whose label contains this text. Omit for all.
    """
    from arc.vision import accessibility as ax

    try:
        tree = ax.read_tree()
    except PlatformError as exc:
        raise ToolError(str(exc)) from exc

    if query:
        matches = ax.find(tree, query)
        if not matches:
            return f"no interactive element matching {query!r}"
        return "\n".join(element.describe() for element in matches[:25])

    return ax.summarize(tree)


@tool(category="screen")
def read_screen_text(display: int = 0) -> str:
    """Read the text on screen with OCR.

    Use when read_screen_elements comes up empty — some applications draw their own
    interface and expose no accessibility tree at all.

    Args:
        display: Which display to read, 0-based.
    """
    from arc.vision import ocr

    try:
        shot = capture_screen(display=display)
        text = ocr.read_text(shot.path)
    except PlatformError as exc:
        raise ToolError(str(exc)) from exc

    if not text.strip():
        return "no text recognised on screen"
    return text[:6000]


@tool(category="screen")
def find_on_screen(text: str, display: int = 0) -> str:
    """Find where text appears on screen, returning coordinates that can be clicked.

    Tries the accessibility tree first and falls back to OCR, per the cheapest-method
    ladder: the tree is exact where it exists, OCR works everywhere.

    Args:
        text: Text to look for.
        display: Which display to search when falling back to OCR.
    """
    from arc.vision import accessibility as ax
    from arc.vision import ocr

    try:
        matches = ax.find(ax.read_tree(), text)
        if matches:
            return "found via accessibility tree:\n" + "\n".join(
                element.describe() for element in matches[:10]
            )
    except PlatformError:
        pass  # No tree available; OCR below is the fallback.

    try:
        shot = capture_screen(display=display)
        regions = ocr.find_text(shot.path, text)
    except PlatformError as exc:
        raise ToolError(str(exc)) from exc

    if not regions:
        return f"{text!r} not found on screen"

    lines = [f"found via OCR ({len(regions)} match(es)):"]
    for region in regions[:10]:
        # OCR coordinates are in the downscaled image; map back to screen space or the
        # click lands somewhere else entirely.
        x, y = shot.to_screen(*region.center)
        lines.append(f"  {region.text[:60]} @ ({x:.0f}, {y:.0f})")
    return "\n".join(lines)
