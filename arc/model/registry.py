"""The model registry — ``config/models.yaml`` parsed into typed entries.

Kept separate from the router so that listing, validating, and selecting models never
requires a backend to be installed or weights to be present. ``arc model list`` works
on a machine with no models downloaded at all.

Every entry carries its licence and the date that licence was verified against the live
Hugging Face model card, because §3 forbids taking it from memory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from arc.config import Config
from arc.errors import ConfigError
from arc.model.base import ModelCapabilities

#: Licences permitted by docs/BRIEF.md §0.1. An entry with anything else is a hard
#: error, not a warning — the whole point is that this cannot drift.
_ALLOWED_LICENCES = frozenset({"Apache-2.0", "MIT"})

_VALID_BACKENDS = frozenset({"mlx", "llamacpp", "vllm", "transformers", "custom"})


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One registry entry."""

    key: str
    backend: str
    repo: str
    licence: str
    licence_verified: str
    context_length: int
    quantization: str | None = None
    #: Approximate resident size in GB, used to warn before a download or a load that
    #: will not fit. Advisory: the real figure depends on the quantization scheme.
    approx_size_gb: float | None = None
    #: GGUF repos hold many quantizations; this names the one to fetch.
    filename: str | None = None
    capabilities: ModelCapabilities = field(
        default_factory=lambda: ModelCapabilities(max_context=4096)
    )
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "key": self.key,
            "backend": self.backend,
            "repo": self.repo,
            "licence": self.licence,
            "licence_verified": self.licence_verified,
            "context_length": self.context_length,
            "quantization": self.quantization,
            "approx_size_gb": self.approx_size_gb,
            "filename": self.filename,
            "capabilities": self.capabilities.to_dict(),
            "notes": self.notes,
        }


def _parse_capabilities(key: str, raw: Any, context_length: int) -> ModelCapabilities:
    """Build capabilities from a registry entry, defaulting conservatively.

    Anything unspecified is False. A backend that forgets to declare tool calling gets
    the prompted-ReAct fallback, which works everywhere, rather than emitting native
    tool calls nothing is parsing.
    """
    if not isinstance(raw, dict):
        raise ConfigError(
            f"model {key!r} field 'capabilities' must be a mapping, got {type(raw).__name__}"
        )
    return ModelCapabilities(
        max_context=_as_int(
            key, "capabilities.max_context", raw.get("max_context", context_length)
        ),
        native_tool_calling=bool(raw.get("native_tool_calling", False)),
        vision=bool(raw.get("vision", False)),
        json_mode=bool(raw.get("json_mode", False)),
        thinking=bool(raw.get("thinking", False)),
    )


def _as_int(key: str, field_name: str, value: Any) -> int:
    """Coerce a config value to int, reporting the offending key on failure.

    Without this, ``context_length: "abc"`` escapes as a bare ``ValueError`` that the
    CLI's ``ArcError`` handler does not catch, so the user gets a traceback instead of
    a message naming the field.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"model {key!r} field {field_name!r} must be a number, got {value!r}"
        ) from exc


def _as_float(key: str, field_name: str, value: Any) -> float:
    """Coerce a config value to float, reporting the offending key on failure."""
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"model {key!r} field {field_name!r} must be a number, got {value!r}"
        ) from exc


def parse_entry(key: str, raw: Any) -> ModelEntry:
    """Validate and build one entry, raising ``ConfigError`` on anything malformed."""
    # Checked before the field loop because `"backend" not in raw` is a *substring*
    # test on a string and a TypeError on None, so a malformed entry would otherwise
    # produce either a misleading "missing required field" or an uncaught crash.
    if not isinstance(raw, dict):
        raise ConfigError(f"model {key!r} must be a mapping of fields, got {type(raw).__name__}")

    for required in ("backend", "repo", "licence", "licence_verified", "context_length"):
        if required not in raw:
            raise ConfigError(f"model {key!r} is missing required field {required!r}")

    backend = str(raw["backend"])
    if backend not in _VALID_BACKENDS:
        raise ConfigError(
            f"model {key!r} has unknown backend {backend!r}; "
            f"expected one of {sorted(_VALID_BACKENDS)}"
        )

    licence = str(raw["licence"])
    if licence not in _ALLOWED_LICENCES:
        raise ConfigError(
            f"model {key!r} is {licence}, which BRIEF §0.1 forbids. "
            f"Only {sorted(_ALLOWED_LICENCES)} are permitted."
        )

    context_length = _as_int(key, "context_length", raw["context_length"])

    return ModelEntry(
        key=key,
        backend=backend,
        repo=str(raw["repo"]),
        licence=licence,
        licence_verified=str(raw["licence_verified"]),
        context_length=context_length,
        quantization=raw.get("quantization"),
        approx_size_gb=(
            _as_float(key, "approx_size_gb", raw["approx_size_gb"])
            if raw.get("approx_size_gb") is not None
            else None
        ),
        filename=raw.get("filename"),
        capabilities=_parse_capabilities(key, raw.get("capabilities") or {}, context_length),
        notes=raw.get("notes"),
    )


def load_registry(config: Config) -> dict[str, ModelEntry]:
    """Parse every entry in ``models.registry``."""
    raw = config.section("models.registry")
    return {key: parse_entry(key, value) for key, value in raw.items()}


def active_key(config: Config, role: str = "chat") -> str | None:
    """Return the registry key selected for a role, or None if unset.

    Machine-local selection lives in ``~/.arc/config.yaml`` (written by
    ``arc model use``), so switching models never dirties the committed config.
    """
    value = config.get(f"models.active.{role}")
    return str(value) if value else None


def resolve(config: Config, role: str = "chat") -> ModelEntry:
    """Return the entry selected for a role, with an actionable error if there is none."""
    registry = load_registry(config)
    if not registry:
        raise ConfigError(
            "no models in the registry. Add one to config/models.yaml, "
            "or run `arc model pull <key>` to fetch a known model."
        )

    key = active_key(config, role)
    if key is None:
        raise ConfigError(
            f"no active {role} model. Choose one with `arc model use <key>`. "
            f"Available: {', '.join(sorted(registry))}"
        )
    if key not in registry:
        raise ConfigError(
            f"active {role} model {key!r} is not in the registry. "
            f"Available: {', '.join(sorted(registry))}"
        )
    return registry[key]
