"""Tool dispatch.

Runs one tool call and turns whatever happens — success, failure, timeout, bad
arguments — into an observation the model can read. That framing is the whole design:
a tool error is *information*, not a crash. The agent loop is expected to see a failure,
understand it, and try something else, which only works if failures arrive as text
rather than exceptions.

Everything goes through the audit log (§0.3), and ``--dry-run`` short-circuits mutating
tools while still running read-only ones — a dry run that cannot even list a directory
tells you nothing about what the agent would have done.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from arc.audit import AuditLogger
from arc.errors import ToolError
from arc.log import get_logger
from arc.tools.registry import Tool, ToolRegistry

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Observation:
    """The result of one tool call, as the model will see it."""

    tool: str
    output: str
    ok: bool
    duration_ms: float
    arguments: dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False

    def render(self) -> str:
        """Format for inclusion in the next prompt."""
        status = "" if self.ok else " (failed)"
        return f"[{self.tool}{status}]\n{self.output}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "duration_ms": round(self.duration_ms, 2),
            "arguments": self.arguments,
            "dry_run": self.dry_run,
        }


def _resolved_hints(function: Any) -> dict[str, Any]:
    """Return a function's type hints as real types.

    ``inspect.Parameter.annotation`` yields *strings* for any module using
    ``from __future__ import annotations`` — which every tool module does. Comparing
    those strings against ``bool`` silently never matches, so coercion below did
    nothing at all in production while passing any test written without that import.
    ``get_type_hints`` evaluates them.
    """
    try:
        return get_type_hints(function)
    except Exception:  # pragma: no cover - unresolvable forward references
        return {}


def _coerce_arguments(tool: Tool, arguments: dict[str, Any]) -> dict[str, Any]:
    """Coerce model-supplied arguments toward the function's annotations.

    Models routinely send ``"true"`` for a bool or ``"5"`` for an int, especially the
    small ones and anything driven by the prompted fallback. Rejecting those would be
    technically correct and practically useless — the intent is unambiguous.
    """
    signature = inspect.signature(tool.function)
    hints = _resolved_hints(tool.function)
    coerced: dict[str, Any] = {}

    for name, value in arguments.items():
        parameter = signature.parameters.get(name)
        annotation = hints.get(name)
        if parameter is None or annotation is None:
            coerced[name] = value
            continue

        try:
            if annotation is bool and isinstance(value, str):
                coerced[name] = value.strip().lower() in {"true", "yes", "1", "on"}
            elif annotation is int and isinstance(value, str):
                coerced[name] = int(value.strip())
            elif annotation is float and isinstance(value, str):
                coerced[name] = float(value.strip())
            elif annotation is str and not isinstance(value, str):
                coerced[name] = str(value)
            else:
                coerced[name] = value
        except (TypeError, ValueError):
            # Leave it alone and let the call fail with a message naming the argument,
            # which is more useful to the model than a coercion error here.
            coerced[name] = value

    return coerced


def _validate(tool: Tool, arguments: dict[str, Any]) -> str | None:
    """Return an error message if the call cannot be made, else None."""
    signature = inspect.signature(tool.function)
    accepted = set(signature.parameters)

    unknown = set(arguments) - accepted
    if unknown:
        return (
            f"unknown argument(s): {', '.join(sorted(unknown))}. "
            f"{tool.name} accepts: {', '.join(sorted(accepted)) or 'no arguments'}"
        )

    missing = [
        name
        for name, parameter in signature.parameters.items()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and name not in arguments
    ]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"

    return None


class Executor:
    """Dispatches tool calls and records what happened."""

    def __init__(
        self,
        registry: ToolRegistry,
        *,
        audit: AuditLogger | None = None,
        dry_run: bool = False,
    ) -> None:
        self._registry = registry
        self._audit = audit
        self._dry_run = dry_run

    @property
    def dry_run(self) -> bool:
        """Whether mutating tools are being skipped."""
        return self._dry_run

    def execute(self, name: str, arguments: dict[str, Any] | None = None) -> Observation:
        """Run one tool call, returning an observation whatever the outcome.

        This never raises for a tool-level problem. An unknown tool, bad arguments, or
        an exception inside the tool all come back as ``ok=False`` observations, so the
        loop can feed them to the model and let it recover.
        """
        arguments = dict(arguments or {})
        started = time.perf_counter()

        try:
            tool = self._registry.get(name)
        except ToolError as exc:
            return self._observe(name, str(exc), ok=False, started=started, arguments=arguments)

        problem = _validate(tool, arguments)
        if problem:
            return self._observe(name, problem, ok=False, started=started, arguments=arguments)

        coerced = _coerce_arguments(tool, arguments)

        if self._dry_run and tool.mutating:
            return self._observe(
                name,
                f"[dry-run] would call {name}({_render_args(coerced)}) but made no changes",
                ok=True,
                started=started,
                arguments=coerced,
                dry_run=True,
            )

        try:
            result = tool.function(**coerced)
        except ToolError as exc:
            # The tool's own, deliberate failure. Expected and informative.
            return self._observe(name, str(exc), ok=False, started=started, arguments=coerced)
        except Exception as exc:
            # A bug in the tool, not a refusal. Surfaced with its type so the model
            # does not mistake a TypeError for a missing file, and logged loudly
            # because unlike a ToolError this one is ours to fix.
            _log.exception("tool %s raised", name)
            return self._observe(
                name,
                f"{type(exc).__name__}: {exc}",
                ok=False,
                started=started,
                arguments=coerced,
            )

        return self._observe(name, _stringify(result), ok=True, started=started, arguments=coerced)

    def _observe(
        self,
        tool: str,
        output: str,
        *,
        ok: bool,
        started: float,
        arguments: dict[str, Any],
        dry_run: bool = False,
    ) -> Observation:
        """Build an observation and record it in the audit log."""
        observation = Observation(
            tool=tool,
            output=output,
            ok=ok,
            duration_ms=(time.perf_counter() - started) * 1000,
            arguments=arguments,
            dry_run=dry_run,
        )

        if self._audit is not None:
            try:
                self._audit.record(
                    f"tool.{tool}",
                    status="dry_run" if dry_run else ("ok" if ok else "error"),
                    tool=tool,
                    args=arguments,
                    result=output if ok else None,
                    error=None if ok else output,
                    duration_ms=observation.duration_ms,
                    dry_run=dry_run,
                )
            except Exception as exc:  # pragma: no cover - audit is best-effort here
                _log.warning("could not audit tool call: %s", exc)

        return observation


def _stringify(result: Any) -> str:
    """Render a tool's return value as text for the model."""
    if result is None:
        return "(no output)"
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        return "\n".join(str(item) for item in result)
    return str(result)


def _render_args(arguments: dict[str, Any]) -> str:
    """Render arguments compactly for dry-run messages."""
    parts = []
    for key, value in arguments.items():
        text = repr(value)
        if len(text) > 60:
            text = text[:60] + "...'"
        parts.append(f"{key}={text}")
    return ", ".join(parts)
