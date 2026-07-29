"""Tests for backend selection.

The router is pure decision-making over hardware.json plus config, so all of it is
testable without a backend installed or weights present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc.config import Config
from arc.model.registry import parse_entry
from arc.model.router import available_accelerators, choose_backend
from arc.paths import hardware_file

ENTRY = {
    "backend": "mlx",
    "repo": "org/model",
    "licence": "Apache-2.0",
    "licence_verified": "2026-07-29",
    "context_length": 4096,
}


def write_hardware(accelerators: list[str]) -> None:
    """Write a minimal hardware.json with the given accelerators."""
    target = hardware_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"hardware": {"accelerators": accelerators}}), encoding="utf-8")


def config_with(directory: Path, body: str = "") -> Config:
    """Build a Config with an optional models.yaml body."""
    (directory / "models.yaml").write_text(body or "registry: {}\n", encoding="utf-8")
    return Config.load(directory=directory, use_env=False)


def test_accelerators_read_from_hardware_file(arc_home_tmp: Path) -> None:
    write_hardware(["mlx", "metal", "cpu"])
    assert available_accelerators() == ["mlx", "metal", "cpu"]


def test_accelerators_default_to_cpu_when_file_missing(arc_home_tmp: Path) -> None:
    """A fresh install with no probe yet must still be able to route."""
    assert available_accelerators() == ["cpu"]


def test_accelerators_default_to_cpu_when_field_empty(arc_home_tmp: Path) -> None:
    write_hardware([])
    assert available_accelerators() == ["cpu"]


def test_preferred_backend_used_when_supported(arc_home_tmp: Path, config_dir: Path) -> None:
    write_hardware(["mlx", "metal", "cpu"])
    choice = choose_backend(parse_entry("m", dict(ENTRY)), config_with(config_dir))
    assert choice.backend == "mlx"
    assert choice.fallback_from is None


def test_falls_back_to_llamacpp_when_unsupported(arc_home_tmp: Path, config_dir: Path) -> None:
    """An MLX entry on a CUDA box must land on llama.cpp, not fail."""
    write_hardware(["cuda", "cpu"])
    choice = choose_backend(parse_entry("m", dict(ENTRY)), config_with(config_dir))
    assert choice.backend == "llamacpp"
    assert choice.fallback_from == "mlx"


def test_fallback_reason_names_both_sides(arc_home_tmp: Path, config_dir: Path) -> None:
    """The reason has to be readable — the user should not reverse-engineer it."""
    write_hardware(["cuda", "cpu"])
    reason = choose_backend(parse_entry("m", dict(ENTRY)), config_with(config_dir)).reason
    assert "mlx" in reason
    assert "cuda" in reason


def test_llamacpp_entry_runs_on_cpu_only_machine(arc_home_tmp: Path, config_dir: Path) -> None:
    write_hardware(["cpu"])
    entry = parse_entry("m", {**ENTRY, "backend": "llamacpp"})
    choice = choose_backend(entry, config_with(config_dir))
    assert choice.backend == "llamacpp"
    assert choice.fallback_from is None


def test_force_backend_overrides_everything(arc_home_tmp: Path, config_dir: Path) -> None:
    """Being able to force a wrong answer is what makes an override useful."""
    write_hardware(["cpu"])
    config = config_with(config_dir, "registry: {}\nrouter:\n  force_backend: vllm\n")
    choice = choose_backend(parse_entry("m", dict(ENTRY)), config)
    assert choice.backend == "vllm"
    assert "forced" in choice.reason


def test_auto_select_can_be_disabled(arc_home_tmp: Path, config_dir: Path) -> None:
    write_hardware(["cuda", "cpu"])
    config = config_with(config_dir, "registry: {}\nrouter:\n  auto_select_backend: false\n")
    choice = choose_backend(parse_entry("m", dict(ENTRY)), config)
    assert choice.backend == "mlx"
    assert "auto-select disabled" in choice.reason


@pytest.mark.parametrize(
    ("backend", "accelerators", "expected"),
    [
        ("vllm", ["cuda", "cpu"], "vllm"),
        ("vllm", ["mlx", "metal", "cpu"], "llamacpp"),
        ("transformers", ["cpu"], "transformers"),
        ("custom", ["mlx", "metal", "cpu"], "custom"),
    ],
)
def test_backend_requirements(
    arc_home_tmp: Path,
    config_dir: Path,
    backend: str,
    accelerators: list[str],
    expected: str,
) -> None:
    write_hardware(accelerators)
    entry = parse_entry("m", {**ENTRY, "backend": backend})
    assert choose_backend(entry, config_with(config_dir)).backend == expected
