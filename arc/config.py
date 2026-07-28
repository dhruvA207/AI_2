"""Configuration loading.

Three sources, later ones winning: the committed YAML in ``config/``, then
``~/.arc/config.yaml`` for machine-local overrides that must never be committed, then
environment variables. The brief's rule is "config over constants" — if a number
influences behaviour, it belongs in a YAML file, not in a literal somewhere in the
logic.

Layout: ``default.yaml`` merges at the root, and every other file merges under a key
named after itself, so ``policy.yaml`` is reachable as ``policy.*`` and
``models.yaml`` as ``models.*``. Predictable enough that you can guess a key's path
from the file it lives in.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, TypeVar

import yaml

from arc.errors import ConfigError
from arc.paths import arc_home, config_dir

T = TypeVar("T")

#: Environment overrides look like ARC_RUNTIME__LOG_LEVEL=debug -> runtime.log_level.
#: Double underscore separates path segments so single underscores stay usable inside
#: key names, which they frequently are.
_ENV_PREFIX = "ARC_"
_ENV_SEPARATOR = "__"

#: These env vars steer path resolution, not config values, so they are not folded in.
_ENV_EXCLUDED = frozenset({"ARC_HOME", "ARC_CONFIG_DIR"})

#: Files merged at the root rather than under their own namespace.
_ROOT_FILES = frozenset({"default"})


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict.

    Dicts merge key by key; every other type replaces wholesale. Notably lists
    replace rather than concatenate — appending would make it impossible to *remove*
    a default entry, which is the more common need.
    """
    result = dict(base)
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML file into a dict, tolerating an empty file."""
    try:
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _coerce(raw: str) -> Any:
    """Interpret an environment string as a YAML scalar.

    So ``ARC_RUNTIME__DRY_RUN=true`` becomes a bool and ``ARC_AGENT__MAX_STEPS=12``
    becomes an int, instead of every override arriving as a string that downstream
    code has to remember to convert.
    """
    try:
        return yaml.safe_load(raw)
    except yaml.YAMLError:
        return raw


def _env_overrides() -> dict[str, Any]:
    """Build a nested dict from ``ARC_*`` environment variables."""
    overrides: dict[str, Any] = {}
    for name, raw in os.environ.items():
        if not name.startswith(_ENV_PREFIX) or name in _ENV_EXCLUDED:
            continue
        path = name[len(_ENV_PREFIX) :].lower().split(_ENV_SEPARATOR)
        if not path or not path[0]:
            continue

        cursor = overrides
        for segment in path[:-1]:
            nxt = cursor.get(segment)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[segment] = nxt
            cursor = nxt
        cursor[path[-1]] = _coerce(raw)
    return overrides


class Config:
    """An immutable-by-convention view over merged configuration."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, directory: Path | None = None, *, use_env: bool = True) -> Config:
        """Load and merge every configuration source.

        ``use_env`` exists so tests can assert on file contents without the ambient
        environment leaking in.
        """
        source = directory if directory is not None else config_dir()
        if not source.is_dir():
            raise ConfigError(f"config directory not found: {source}")

        data: dict[str, Any] = {}
        for path in sorted(source.glob("*.yaml")):
            parsed = _read_yaml(path)
            if path.stem in _ROOT_FILES:
                data = _deep_merge(data, parsed)
            else:
                data = _deep_merge(data, {path.stem: parsed})

        local = arc_home() / "config.yaml"
        if local.is_file():
            data = _deep_merge(data, _read_yaml(local))

        if use_env:
            data = _deep_merge(data, _env_overrides())

        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        """Return the value at a dotted path, or ``default`` if any segment is missing."""
        cursor: Any = self._data
        for segment in dotted.split("."):
            if not isinstance(cursor, dict) or segment not in cursor:
                return default
            cursor = cursor[segment]
        return cursor

    def require(self, dotted: str) -> Any:
        """Return the value at a dotted path, raising if it is absent.

        Used for settings with no sensible default, where silently falling back would
        hide a broken install.
        """
        sentinel = object()
        value = self.get(dotted, sentinel)
        if value is sentinel:
            raise ConfigError(f"required config key missing: {dotted}")
        return value

    def typed(self, dotted: str, expected: type[T], default: T) -> T:
        """Return a value checked against ``expected``, raising on a type mismatch.

        A wrong type in YAML is a configuration bug, and reporting it here names the
        offending key. Discovering it later as a ``TypeError`` deep in the agent loop
        does not.
        """
        value = self.get(dotted, default)
        # bool is a subclass of int, so an isinstance check would let `true` satisfy
        # an int field. Reject that explicitly rather than silently coercing.
        if expected is not bool and isinstance(value, bool):
            raise ConfigError(f"config key {dotted} expected {expected.__name__}, got bool")
        if not isinstance(value, expected):
            raise ConfigError(
                f"config key {dotted} expected {expected.__name__}, got {type(value).__name__}"
            )
        return value

    def section(self, dotted: str) -> dict[str, Any]:
        """Return a sub-mapping, or an empty dict if absent."""
        value = self.get(dotted, {})
        if not isinstance(value, dict):
            raise ConfigError(f"config key {dotted} is not a mapping")
        return value

    def as_dict(self) -> dict[str, Any]:
        """Return a shallow copy of the merged configuration."""
        return dict(self._data)

    def __contains__(self, dotted: str) -> bool:
        sentinel = object()
        return self.get(dotted, sentinel) is not sentinel

    def __repr__(self) -> str:
        return f"Config(keys={sorted(self._data)})"
