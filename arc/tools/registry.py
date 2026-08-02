"""Tool registration.

A decorator turns a plain Python function into something the model can call. The JSON
schema is derived from type hints and the docstring rather than written by hand,
because a hand-maintained schema drifts from its implementation the first time someone
adds a parameter — and the failure is silent: the model simply never learns the
argument exists.

The same registry feeds both dispatch paths. A model with native tool calling gets
``ToolSchema`` objects; one without gets the same schemas rendered into a prompt
(§4.1). Neither the tools nor the registry know which path is in use.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar, Union, get_args, get_origin, get_type_hints, overload

from arc.errors import ToolError
from arc.log import get_logger
from arc.model.base import ToolSchema

_log = get_logger(__name__)

#: Bound to the decorated function's own type, so decorating preserves the signature.
F = TypeVar("F", bound=Callable[..., Any])

#: Python types mapped onto JSON Schema types. Anything unrecognised becomes "string",
#: which is the honest fallback: the model sends text and the tool coerces it.
#:
#: **Order matters.** ``bool`` is a subclass of ``int``, so a subclass check that meets
#: ``int`` first types every boolean as "integer" — and the model then dutifully sends
#: 1 and 0 for a flag. bool must be tested before int, exactly as in Config.typed().
_JSON_TYPES: tuple[tuple[type, str], ...] = (
    (bool, "boolean"),
    (str, "string"),
    (int, "integer"),
    (float, "number"),
    (list, "array"),
    (dict, "object"),
)


@dataclass(frozen=True, slots=True)
class Tool:
    """A registered callable, its schema, and its metadata."""

    name: str
    description: str
    function: Callable[..., Any]
    parameters: dict[str, Any]
    #: Tools that change the world are skipped under --dry-run; read-only ones still
    #: run, because a dry run that cannot even look at the filesystem tells you
    #: nothing about what the agent would have done.
    mutating: bool = False
    #: Grouping for `arc tools list`, and the unit `config/policy.yaml` enables or
    #: disables if the permissive default is ever narrowed.
    category: str = "general"

    def schema(self) -> ToolSchema:
        """Return the model-facing schema."""
        return ToolSchema(name=self.name, description=self.description, parameters=self.parameters)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "mutating": self.mutating,
            "parameters": self.parameters,
        }


def _json_type(annotation: Any) -> dict[str, Any]:
    """Translate a type annotation into a JSON Schema fragment."""
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)

    # `X | None` and `Optional[X]` describe the same thing to the model: X, optional.
    if origin in (Union, types.UnionType):
        variants = [a for a in get_args(annotation) if a is not type(None)]
        if len(variants) == 1:
            return _json_type(variants[0])
        return {"type": "string"}

    if origin in (list, set, tuple):
        element = get_args(annotation)
        return {"type": "array", "items": _json_type(element[0]) if element else {"type": "string"}}

    if origin is dict:
        return {"type": "object"}

    if isinstance(annotation, type):
        for python_type, json_type in _JSON_TYPES:
            if issubclass(annotation, python_type):
                return {"type": json_type}

    return {"type": "string"}


def _parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into a summary and per-parameter descriptions.

    Understands the ``Args:`` block this codebase already uses. Parameter descriptions
    matter more than they look: they are the only place the model learns what
    ``recursive`` or ``pattern`` actually mean.
    """
    if not doc:
        return "", {}

    lines = doc.strip().splitlines()
    summary: list[str] = []
    params: dict[str, str] = {}
    current: str | None = None
    in_args = False
    # The summary is the first paragraph only. Tracked with a flag rather than by
    # breaking out of the loop: the standard docstring format puts a blank line
    # *before* ``Args:``, so breaking there skipped the argument block entirely and
    # every tool shipped with undocumented parameters — the model never learned what
    # `recursive` or `pattern` meant.
    summary_done = False

    for raw in lines:
        line = raw.strip()
        if line.lower() in {"args:", "arguments:", "parameters:"}:
            in_args = True
            summary_done = True
            continue
        if in_args and line.lower() in {"returns:", "raises:", "yields:", "example:", "examples:"}:
            in_args = False
            current = None
            continue

        if in_args:
            # `name: description`, where the name is a single word. Continuation lines
            # are indented and have no colon of their own.
            head, separator, description = line.partition(":")
            if separator and head.strip() and len(head.split()) == 1:
                current = head.strip()
                params[current] = description.strip()
            elif current and line:
                params[current] += " " + line
        elif line and not summary_done:
            summary.append(line)
        elif not line and summary:
            summary_done = True

    return " ".join(summary), params


def build_parameters(function: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema object for a function's parameters.

    Derived rather than declared so the schema cannot drift from the implementation.
    A parameter without a default is required; one with a default is not.
    """
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}

    _, docs = _parse_docstring(function.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, parameter in signature.parameters.items():
        if name in {"self", "cls"} or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        schema = _json_type(hints.get(name, parameter.annotation))
        if name in docs:
            schema["description"] = docs[name]
        if parameter.default is not inspect.Parameter.empty and parameter.default is not None:
            schema["default"] = parameter.default

        properties[name] = schema
        if parameter.default is inspect.Parameter.empty:
            required.append(name)

    return {"type": "object", "properties": properties, "required": required}


@dataclass
class ToolRegistry:
    """The set of tools available to the agent."""

    tools: dict[str, Tool] = field(default_factory=dict)

    @overload
    def register(self, function: F) -> F: ...

    @overload
    def register(
        self,
        function: None = None,
        *,
        name: str | None = ...,
        mutating: bool = ...,
        category: str = ...,
    ) -> Callable[[F], F]: ...

    def register(
        self,
        function: F | None = None,
        *,
        name: str | None = None,
        mutating: bool = False,
        category: str = "general",
    ) -> F | Callable[[F], F]:
        """Register a function as a tool. Usable bare or with arguments.

        Overloaded so the decorator returns the *same* type it was given. Without that
        mypy treats every decorated tool as untyped and stops checking its body — which
        would quietly disable --strict across the entire tools package.
        """

        def decorate(target: F) -> F:
            tool_name = name or target.__name__
            summary, _ = _parse_docstring(target.__doc__)
            if not summary:
                raise ToolError(
                    f"tool {tool_name!r} needs a docstring: it is what the model reads "
                    "to decide whether to call it"
                )

            self.tools[tool_name] = Tool(
                name=tool_name,
                description=summary,
                function=target,
                parameters=build_parameters(target),
                mutating=mutating,
                category=category,
            )
            return target

        if function is not None:
            return decorate(function)
        return decorate

    def get(self, name: str) -> Tool:
        """Return a tool, with a message the model can act on if it is missing."""
        tool = self.tools.get(name)
        if tool is None:
            available = ", ".join(sorted(self.tools)) or "none"
            raise ToolError(f"no tool named {name!r}. Available tools: {available}")
        return tool

    def __contains__(self, name: str) -> bool:
        return name in self.tools

    def __len__(self) -> int:
        return len(self.tools)

    def category_of(self, name: str) -> str:
        """Which category a tool belongs to, or ``general`` if it is unknown.

        Never raises: this is used to colour a UI element, and a tool that has been
        renamed should not take a request down with it.
        """
        tool = self.tools.get(name)
        return tool.category if tool is not None else "general"

    def schemas(self, *, categories: list[str] | None = None) -> list[ToolSchema]:
        """Return schemas for the model, optionally filtered by category."""
        selected = [
            t for t in self.tools.values() if categories is None or t.category in categories
        ]
        return [t.schema() for t in sorted(selected, key=lambda t: t.name)]

    def describe(self) -> list[dict[str, Any]]:
        """Return every tool as a dict, for `arc tools list`."""
        return [t.to_dict() for t in sorted(self.tools.values(), key=lambda t: t.name)]

    def render_prompt(self, *, categories: list[str] | None = None) -> str:
        """Render the tool list for a model without native tool calling.

        This is the other half of §4.1's fallback. The format is deliberately plain —
        a small model follows a simple, explicit convention far more reliably than a
        clever one, and every token spent on formatting is one not spent reasoning.
        """
        lines: list[str] = []
        for schema in self.schemas(categories=categories):
            properties = schema.parameters.get("properties", {})
            required = set(schema.parameters.get("required", []))

            args = []
            for name, spec in properties.items():
                marker = "" if name in required else "?"
                description = spec.get("description", "")
                args.append(
                    f"    {name}{marker}: {spec.get('type', 'string')}"
                    + (f"  — {description}" if description else "")
                )

            lines.append(f"- {schema.name}: {schema.description}")
            if args:
                lines.extend(args)

        return "\n".join(lines)


#: The process-wide registry. Tool modules populate it at import time, which is why
#: `arc/tools/__init__.py` imports them all.
registry = ToolRegistry()


@overload
def tool[F: Callable[..., Any]](function: F) -> F: ...


@overload
def tool[F: Callable[..., Any]](
    function: None = None,
    *,
    name: str | None = ...,
    mutating: bool = ...,
    category: str = ...,
) -> Callable[[F], F]: ...


def tool[F: Callable[..., Any]](
    function: F | None = None,
    *,
    name: str | None = None,
    mutating: bool = False,
    category: str = "general",
) -> F | Callable[[F], F]:
    """Register a function on the default registry."""
    if function is not None:
        return registry.register(function)
    return registry.register(name=name, mutating=mutating, category=category)
