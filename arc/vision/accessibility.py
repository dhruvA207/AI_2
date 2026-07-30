"""The accessibility tree — reading the screen structurally.

§4.3 puts this first for good reason: it is exact, fast, and free. A vision model looks
at pixels and *infers* that something is a button; the accessibility tree is what the
application itself declares, including the element's role, its label, whether it is
enabled, and precisely where it is. Clicking an AX element's own frame does not drift
when a window moves or the resolution changes, which pixel targeting does constantly.

macOS exposes this through AXUIElement. It requires the Accessibility grant — the same
one input control needs — and returns nothing at all without it, so a missing grant is
detected and reported rather than looking like an empty screen.

The tree can be enormous. A browser window may expose tens of thousands of elements, so
traversal is depth- and count-limited, and elements with no label and no interactivity
are skipped: they are layout containers, and listing them buries the handful of things
an agent can actually act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc.errors import PlatformError
from arc.log import get_logger

_log = get_logger(__name__)

#: Traversal limits. A full browser tree is tens of thousands of nodes, and walking it
#: takes seconds — long enough that the UI has changed by the time you finish.
MAX_DEPTH = 14
MAX_ELEMENTS = 400

#: Roles worth reporting even without a label: the agent can act on these.
_INTERACTIVE_ROLES = frozenset(
    {
        "AXButton",
        "AXCheckBox",
        "AXRadioButton",
        "AXTextField",
        "AXTextArea",
        "AXLink",
        "AXMenuItem",
        "AXMenuButton",
        "AXPopUpButton",
        "AXComboBox",
        "AXSlider",
        "AXTab",
        "AXMenuBarItem",
        "AXToolbarButton",
        "AXDisclosureTriangle",
        "AXSearchField",
        "AXIncrementor",
    }
)

#: Pure layout. Traversed through, never reported — listing them buries the elements
#: that can actually be acted on.
_CONTAINER_ROLES = frozenset(
    {"AXGroup", "AXSplitGroup", "AXScrollArea", "AXLayoutArea", "AXLayoutItem", "AXUnknown"}
)


@dataclass
class Element:
    """One node of the accessibility tree."""

    role: str
    label: str = ""
    value: str = ""
    #: Screen rectangle: (x, y, width, height). None when the element has no position,
    #: which happens for offscreen or purely logical nodes.
    frame: tuple[float, float, float, float] | None = None
    enabled: bool = True
    depth: int = 0
    children: list[Element] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float] | None:
        """Where to click. None when the element has no frame."""
        if self.frame is None:
            return None
        x, y, width, height = self.frame
        return (x + width / 2, y + height / 2)

    @property
    def actionable(self) -> bool:
        """Whether an agent could plausibly do something with this."""
        return self.role in _INTERACTIVE_ROLES and self.enabled and self.frame is not None

    def describe(self) -> str:
        """One line, formatted for a model to read and act on."""
        name = self.label or self.value or "(unlabelled)"
        text = f"{self.role.removeprefix('AX')}: {name[:70]}"
        centre = self.center
        if centre is not None:
            text += f" @ ({centre[0]:.0f}, {centre[1]:.0f})"
        if not self.enabled:
            text += " [disabled]"
        return text

    def flatten(self) -> list[Element]:
        """This element and every descendant, depth-first."""
        found = [self]
        for child in self.children:
            found.extend(child.flatten())
        return found

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "role": self.role,
            "label": self.label,
            "value": self.value[:200],
            "frame": list(self.frame) if self.frame else None,
            "center": list(self.center) if self.center else None,
            "enabled": self.enabled,
            "actionable": self.actionable,
            "children": [c.to_dict() for c in self.children],
        }


def _services() -> Any:
    """Return ApplicationServices, or raise something actionable."""
    try:
        import ApplicationServices
    except ImportError as exc:  # pragma: no cover - non-macOS
        raise PlatformError(
            "reading the accessibility tree needs pyobjc-framework-ApplicationServices"
        ) from exc
    return ApplicationServices


def is_trusted() -> bool:
    """Whether this process may read the accessibility tree."""
    try:
        return bool(_services().AXIsProcessTrusted())
    except Exception:
        return False


def _attribute(services: Any, element: Any, name: str) -> Any:
    """Read one AX attribute, or None if it is absent.

    AX returns an error code alongside the value rather than raising, and unsupported
    attributes are extremely common — nearly every element lacks most of them — so this
    stays quiet rather than logging.
    """
    try:
        error, value = services.AXUIElementCopyAttributeValue(element, name, None)
    except Exception:
        return None
    return value if error == 0 else None


def _frame(services: Any, element: Any) -> tuple[float, float, float, float] | None:
    """Read an element's screen rectangle."""
    position = _attribute(services, element, services.kAXPositionAttribute)
    size = _attribute(services, element, services.kAXSizeAttribute)
    if position is None or size is None:
        return None

    try:
        # pyobjc *returns* (success, value) rather than filling a struct passed by
        # reference the way the C API does. Passing a pre-made CGPoint and ignoring
        # the return silently produced a frame of None for every element, which made
        # the entire tree look unactionable.
        got_point, point = services.AXValueGetValue(position, services.kAXValueTypeCGPoint, None)
        got_size, extent = services.AXValueGetValue(size, services.kAXValueTypeCGSize, None)
        if not got_point or not got_size:
            return None
        if extent.width <= 0 or extent.height <= 0:
            return None
        return (float(point.x), float(point.y), float(extent.width), float(extent.height))
    except Exception:
        return None


def _build(services: Any, element: Any, depth: int, budget: list[int]) -> Element | None:
    """Recursively convert an AXUIElement into an ``Element``."""
    if depth > MAX_DEPTH or budget[0] <= 0:
        return None
    budget[0] -= 1

    role = _attribute(services, element, services.kAXRoleAttribute) or "AXUnknown"

    def text(attribute: str) -> str:
        raw = _attribute(services, element, attribute)
        return str(raw).strip() if raw is not None else ""

    label = (
        text(services.kAXTitleAttribute)
        or text(services.kAXDescriptionAttribute)
        or text(services.kAXHelpAttribute)
    )
    enabled_raw = _attribute(services, element, services.kAXEnabledAttribute)

    node = Element(
        role=str(role),
        label=label,
        value=text(services.kAXValueAttribute),
        frame=_frame(services, element),
        enabled=True if enabled_raw is None else bool(enabled_raw),
        depth=depth,
    )

    children = _attribute(services, element, services.kAXChildrenAttribute) or []
    for child in children:
        if budget[0] <= 0:
            break
        built = _build(services, child, depth + 1, budget)
        if built is not None:
            node.children.append(built)

    # Prune nodes that are neither interesting themselves nor lead anywhere: pure
    # layout containers with nothing beneath them are noise.
    if not node.children and node.role in _CONTAINER_ROLES and not node.label and not node.value:
        return None
    return node


def frontmost_application() -> tuple[int, str]:
    """Return the (pid, name) of the frontmost application."""
    try:
        import AppKit
    except ImportError as exc:  # pragma: no cover - non-macOS
        raise PlatformError("needs AppKit") from exc

    app = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
    if app is None:
        raise PlatformError("could not determine the frontmost application")
    return (int(app.processIdentifier()), str(app.localizedName() or "unknown"))


def read_tree(pid: int | None = None, *, max_elements: int = MAX_ELEMENTS) -> Element:
    """Read the accessibility tree of an application, defaulting to the frontmost one."""
    services = _services()

    if not is_trusted():
        raise PlatformError(
            "accessibility access is not granted, so the screen cannot be read "
            "structurally. Grant it in System Settings > Privacy & Security > "
            "Accessibility, then restart the terminal."
        )

    if pid is None:
        pid, _name = frontmost_application()

    app_element = services.AXUIElementCreateApplication(pid)
    if app_element is None:
        raise PlatformError(f"could not attach to process {pid}")

    budget = [max_elements]
    tree = _build(services, app_element, 0, budget)
    if tree is None:
        raise PlatformError(f"process {pid} exposed no accessibility tree")

    _log.info(
        "read accessibility tree",
        extra={"pid": pid, "elements": max_elements - budget[0]},
    )
    return tree


def actionable_elements(tree: Element) -> list[Element]:
    """Return only the elements an agent could act on, which is usually what it wants."""
    return [element for element in tree.flatten() if element.actionable]


def find(tree: Element, text: str, *, actionable_only: bool = True) -> list[Element]:
    """Find elements whose label or value contains ``text``, case-insensitively.

    This is the targeting primitive §4.3 asks for: locate a control by what it *says*
    and click its own frame, rather than guessing pixel coordinates that break the
    moment a window moves.
    """
    needle = text.lower().strip()
    if not needle:
        return []

    candidates = actionable_elements(tree) if actionable_only else tree.flatten()
    exact = []
    partial = []
    for element in candidates:
        haystack = f"{element.label} {element.value}".lower()
        if needle == element.label.lower().strip():
            exact.append(element)
        elif needle in haystack:
            partial.append(element)
    # Exact label matches first: "Save" should not be beaten by "Save As...".
    return exact + partial


def summarize(tree: Element, *, limit: int = 60) -> str:
    """Render the actionable parts of a tree for a model."""
    elements = actionable_elements(tree)
    if not elements:
        return "no actionable elements found (the application may not expose one)"

    lines = [f"{len(elements)} actionable elements:"]
    lines.extend(f"  {element.describe()}" for element in elements[:limit])
    if len(elements) > limit:
        lines.append(f"  ... and {len(elements) - limit} more")
    return "\n".join(lines)
