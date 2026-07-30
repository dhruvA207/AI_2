"""Mouse and keyboard tools for the agent.

All mutating, so ``--dry-run`` skips them: a dry run must never actually click on
anything. Each one also requires an active control session, which means the blue glow
is up and the user can take over at any moment (see ``arc/control/session.py``).
"""

from __future__ import annotations

from arc.control import input as ctl
from arc.control import session as control_session
from arc.errors import ControlError, ToolError
from arc.log import get_logger
from arc.tools.registry import tool

_log = get_logger(__name__)


def _guard(action: str) -> None:
    """Turn a control failure into a tool observation the model can act on."""
    try:
        control_session.require()
    except ControlError as exc:
        raise ToolError(f"cannot {action}: {exc}") from exc


@tool(category="input", mutating=True)
def mouse_move(x: float, y: float) -> str:
    """Move the mouse pointer to a screen coordinate.

    Args:
        x: Horizontal screen coordinate.
        y: Vertical screen coordinate.
    """
    _guard("move the mouse")
    try:
        ctl.move_to(x, y)
    except ControlError as exc:
        raise ToolError(str(exc)) from exc
    return f"moved pointer to ({x:.0f}, {y:.0f})"


@tool(category="input", mutating=True)
def mouse_click(
    x: float | None = None, y: float | None = None, button: str = "left", count: int = 1
) -> str:
    """Click the mouse, optionally moving to a coordinate first.

    Args:
        x: Horizontal coordinate. Omit to click where the pointer already is.
        y: Vertical coordinate.
        button: One of left, right, middle.
        count: Number of clicks; 2 for a double-click.
    """
    _guard("click")
    try:
        ctl.click(x, y, button=button, count=count)
    except ControlError as exc:
        raise ToolError(str(exc)) from exc
    where = f" at ({x:.0f}, {y:.0f})" if x is not None and y is not None else ""
    return f"{button} clicked{where} x{count}"


@tool(category="input", mutating=True)
def mouse_drag(from_x: float, from_y: float, to_x: float, to_y: float) -> str:
    """Drag from one screen coordinate to another.

    Args:
        from_x: Starting horizontal coordinate.
        from_y: Starting vertical coordinate.
        to_x: Ending horizontal coordinate.
        to_y: Ending vertical coordinate.
    """
    _guard("drag")
    try:
        ctl.drag(from_x, from_y, to_x, to_y)
    except ControlError as exc:
        raise ToolError(str(exc)) from exc
    return f"dragged ({from_x:.0f}, {from_y:.0f}) to ({to_x:.0f}, {to_y:.0f})"


@tool(category="input", mutating=True)
def mouse_scroll(amount: int, horizontal: bool = False) -> str:
    """Scroll the wheel. Positive scrolls up, or right when horizontal.

    Args:
        amount: Lines to scroll.
        horizontal: Scroll sideways instead of vertically.
    """
    _guard("scroll")
    try:
        ctl.scroll(amount, horizontal=horizontal)
    except ControlError as exc:
        raise ToolError(str(exc)) from exc
    return f"scrolled {amount} lines {'horizontally' if horizontal else 'vertically'}"


@tool(category="input", mutating=True)
def keyboard_type(text: str) -> str:
    """Type text at the current focus.

    Args:
        text: What to type.
    """
    _guard("type")
    try:
        ctl.type_text(text)
    except ControlError as exc:
        raise ToolError(str(exc)) from exc
    return f"typed {len(text)} characters"


@tool(category="input", mutating=True)
def keyboard_press(key: str, modifiers: str = "") -> str:
    """Press a key, optionally with modifiers.

    Args:
        key: A single character, or a name like return, tab, escape, up, down.
        modifiers: Space-separated modifiers, e.g. "cmd shift".
    """
    _guard("press a key")
    try:
        ctl.press(key, *(m for m in modifiers.split() if m))
    except ControlError as exc:
        raise ToolError(str(exc)) from exc
    combo = f"{modifiers} {key}".strip()
    return f"pressed {combo}"


@tool(category="input")
def pointer_location() -> str:
    """Report where the mouse pointer currently is."""
    position = control_session.pointer_position()
    if position is None:
        raise ToolError("could not read the pointer position")
    return f"pointer is at ({position[0]:.0f}, {position[1]:.0f})"
