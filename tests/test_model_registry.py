"""Tests for the model registry and its licence enforcement."""

from __future__ import annotations

from pathlib import Path

import pytest

from arc.config import Config
from arc.errors import ConfigError
from arc.model.registry import active_key, load_registry, parse_entry, resolve

MINIMAL = {
    "backend": "mlx",
    "repo": "org/model",
    "licence": "Apache-2.0",
    "licence_verified": "2026-07-29",
    "context_length": 4096,
}


def test_parse_minimal_entry() -> None:
    entry = parse_entry("m", dict(MINIMAL))
    assert entry.key == "m"
    assert entry.backend == "mlx"
    assert entry.context_length == 4096


@pytest.mark.parametrize(
    "missing", ["backend", "repo", "licence", "licence_verified", "context_length"]
)
def test_missing_required_field_raises(missing: str) -> None:
    raw = dict(MINIMAL)
    del raw[missing]
    with pytest.raises(ConfigError, match=f"missing required field '{missing}'"):
        parse_entry("m", raw)


def test_unknown_backend_raises() -> None:
    with pytest.raises(ConfigError, match="unknown backend"):
        parse_entry("m", {**MINIMAL, "backend": "witchcraft"})


@pytest.mark.parametrize("licence", ["GPL-3.0", "AGPL-3.0", "CC-BY-NC-4.0", "llama-community"])
def test_forbidden_licence_is_a_hard_error(licence: str) -> None:
    """BRIEF §0.1 is a hard rule, so a non-permissive licence must not merely warn."""
    with pytest.raises(ConfigError, match="BRIEF"):
        parse_entry("m", {**MINIMAL, "licence": licence})


@pytest.mark.parametrize("licence", ["Apache-2.0", "MIT"])
def test_permitted_licences_pass(licence: str) -> None:
    assert parse_entry("m", {**MINIMAL, "licence": licence}).licence == licence


def test_capabilities_default_to_false() -> None:
    """A backend that forgets to declare tool calling must get the safe fallback."""
    caps = parse_entry("m", dict(MINIMAL)).capabilities
    assert caps.native_tool_calling is False
    assert caps.vision is False
    assert caps.json_mode is False
    assert caps.thinking is False


def test_capabilities_are_read_when_present() -> None:
    raw = {**MINIMAL, "capabilities": {"native_tool_calling": True, "thinking": True}}
    caps = parse_entry("m", raw).capabilities
    assert caps.native_tool_calling is True
    assert caps.thinking is True
    assert caps.vision is False


def test_capabilities_max_context_defaults_to_context_length() -> None:
    assert parse_entry("m", dict(MINIMAL)).capabilities.max_context == 4096


def write_models(directory: Path, body: str) -> Config:
    """Build a Config from a models.yaml body."""
    (directory / "models.yaml").write_text(body, encoding="utf-8")
    return Config.load(directory=directory, use_env=False)


def test_load_registry_from_config(config_dir: Path) -> None:
    config = write_models(
        config_dir,
        """
registry:
  a:
    backend: mlx
    repo: org/a
    licence: MIT
    licence_verified: "2026-07-29"
    context_length: 2048
""",
    )
    registry = load_registry(config)
    assert set(registry) == {"a"}


def test_empty_registry(config_dir: Path) -> None:
    assert load_registry(write_models(config_dir, "registry: {}\n")) == {}


def test_resolve_reports_empty_registry(config_dir: Path) -> None:
    config = write_models(config_dir, "registry: {}\nactive:\n  chat: null\n")
    with pytest.raises(ConfigError, match="no models in the registry"):
        resolve(config)


def test_resolve_reports_unset_active(config_dir: Path) -> None:
    config = write_models(
        config_dir,
        """
registry:
  a: {backend: mlx, repo: org/a, licence: MIT, licence_verified: "x", context_length: 8}
active:
  chat: null
""",
    )
    with pytest.raises(ConfigError, match="no active chat model"):
        resolve(config)


def test_resolve_reports_dangling_active(config_dir: Path) -> None:
    """A stale ~/.arc/config.yaml pointing at a removed model must say so clearly."""
    config = write_models(
        config_dir,
        """
registry:
  a: {backend: mlx, repo: org/a, licence: MIT, licence_verified: "x", context_length: 8}
active:
  chat: gone
""",
    )
    with pytest.raises(ConfigError, match="not in the registry"):
        resolve(config)


def test_resolve_returns_the_active_entry(config_dir: Path) -> None:
    config = write_models(
        config_dir,
        """
registry:
  a: {backend: mlx, repo: org/a, licence: MIT, licence_verified: "x", context_length: 8}
active:
  chat: a
""",
    )
    assert resolve(config).key == "a"
    assert active_key(config) == "a"


@pytest.mark.parametrize("value", ["just a string", [1, 2, 3], None, 5, True])
def test_non_mapping_entry_is_reported_clearly(value: object) -> None:
    """Regression: `"backend" not in raw` is a substring test on a string and a
    TypeError on None, so a malformed entry used to produce either a misleading
    "missing required field" or an uncaught crash."""
    with pytest.raises(ConfigError, match="must be a mapping of fields"):
        parse_entry("m", value)


@pytest.mark.parametrize("value", ["abc", None, [1], {}])
def test_non_numeric_context_length_raises_config_error(value: object) -> None:
    """Regression: int("abc") raised a bare ValueError that escaped the CLI's
    ArcError handler, so the user saw a traceback instead of the field name."""
    with pytest.raises(ConfigError, match="context_length"):
        parse_entry("m", {**MINIMAL, "context_length": value})


def test_non_numeric_size_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="approx_size_gb"):
        parse_entry("m", {**MINIMAL, "approx_size_gb": "big"})


def test_non_mapping_capabilities_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="capabilities"):
        parse_entry("m", {**MINIMAL, "capabilities": "yes please"})


def test_non_numeric_max_context_raises_config_error() -> None:
    with pytest.raises(ConfigError, match="max_context"):
        parse_entry("m", {**MINIMAL, "capabilities": {"max_context": "lots"}})


def test_malformed_entry_in_registry_names_the_key(config_dir: Path) -> None:
    """The error must name which entry is broken, not just that something is."""
    config = write_models(config_dir, "registry:\n  broken: null\n")
    with pytest.raises(ConfigError, match="'broken'"):
        load_registry(config)


def test_committed_registry_is_valid() -> None:
    """The real config/models.yaml must parse and satisfy the licence rule."""
    registry = load_registry(Config.load(use_env=False))
    assert registry, "registry should not be empty by Phase 2"
    for entry in registry.values():
        assert entry.licence in {"Apache-2.0", "MIT"}
        assert entry.licence_verified
