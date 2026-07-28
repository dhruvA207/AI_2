"""Tests for the hardware probe, sizing table, and hardware.json persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arc.errors import HardwareProbeError
from arc.hardware import (
    _weights_gb,
    load_hardware_file,
    recommend_model,
    refresh,
    write_hardware_file,
)
from arc.paths import hardware_file
from arc.platform.base import HARDWARE_SCHEMA_VERSION, HardwareInfo


def make_info(
    *,
    ram_gb: float = 16.0,
    unified: bool = True,
    vram_gb: float | None = None,
    fanless: bool | None = False,
    chassis: str | None = "Test Machine",
) -> HardwareInfo:
    """Build a HardwareInfo without touching the real machine."""
    return HardwareInfo(
        schema_version=HARDWARE_SCHEMA_VERSION,
        probed_at="2026-07-28T00:00:00+00:00",
        os_name="macos",
        os_version="26.5.1",
        os_build="25F80",
        arch="arm64",
        python_version="3.12.13",
        cpu_model="Test CPU",
        cpu_cores_physical=8,
        cpu_cores_logical=8,
        cpu_performance_cores=4,
        cpu_efficiency_cores=4,
        ram_total_gb=ram_gb,
        unified_memory=unified,
        gpu_vendor="Test",
        gpu_model="Test GPU",
        gpu_cores=10,
        vram_gb=vram_gb,
        disk_free_gb=200.0,
        chassis=chassis,
        fanless=fanless,
        accelerators=["cpu"],
    )


@pytest.mark.parametrize(
    ("ram_gb", "expected"),
    [
        (8.0, "3-4B"),
        (11.9, "3-4B"),
        (12.0, "7-8B"),
        (16.0, "7-8B"),
        (24.0, "14B"),
        (48.0, "30B-class MoE"),
        (96.0, "70B"),
        (256.0, "70B"),
    ],
)
def test_sizing_table_boundaries(ram_gb: float, expected: str) -> None:
    """Bounds are exclusive upper, so 12.0 GB lands in the *next* bucket up."""
    assert recommend_model(make_info(ram_gb=ram_gb)).params_label == expected


def test_weights_estimate_includes_overhead() -> None:
    """8B at 4-bit is ~3.7 GB of raw tensor bytes; the 15% uplift covers the rest."""
    raw = 8e9 * 4 / 8 / 1024**3
    assert _weights_gb(8.0, "4-bit") == pytest.approx(round(raw * 1.15, 1))


def test_quantization_affects_weights() -> None:
    assert _weights_gb(8.0, "8-bit") > _weights_gb(8.0, "4-bit")


def test_unified_memory_reserves_headroom_for_the_os() -> None:
    """A model that "fits" in total RAM will swap on a unified-memory machine."""
    sizing = recommend_model(make_info(ram_gb=16.0, unified=True))
    assert sizing.usable_memory_gb < 16.0
    assert sizing.usable_memory_gb == pytest.approx(11.5, abs=0.1)


def test_discrete_gpu_sizes_against_vram_not_ram() -> None:
    """VRAM, not system RAM, is what a discrete-GPU machine can actually hold."""
    info = make_info(ram_gb=32.0, unified=False, vram_gb=8.0)
    assert recommend_model(info).usable_memory_gb == pytest.approx(8.0)


def test_fanless_machine_gets_a_thermal_note() -> None:
    notes = recommend_model(make_info(fanless=True, chassis="MacBook Air")).notes
    assert any("fanless" in note for note in notes)


def test_cooled_machine_gets_no_thermal_note() -> None:
    notes = recommend_model(make_info(fanless=False)).notes
    assert not any("fanless" in note for note in notes)


def test_unknown_cooling_gets_no_thermal_note() -> None:
    """None means "could not tell" and must not be reported as a hardware limit."""
    notes = recommend_model(make_info(fanless=None)).notes
    assert not any("fanless" in note for note in notes)


def test_small_machine_cannot_coreside_a_vlm() -> None:
    sizing = recommend_model(make_info(ram_gb=8.0))
    assert sizing.can_coreside_vlm is False
    assert any("co-reside" in note for note in sizing.notes)


def test_write_then_load_roundtrip(arc_home_tmp: Path) -> None:
    info = make_info()
    path = write_hardware_file(info, recommend_model(info))
    loaded = load_hardware_file()
    assert loaded is not None
    assert loaded["hardware"]["chassis"] == "Test Machine"
    assert loaded["hardware"]["schema_version"] == HARDWARE_SCHEMA_VERSION
    assert path == hardware_file()


def test_write_leaves_no_temp_file(arc_home_tmp: Path) -> None:
    """Atomic write: temp file plus rename, so a crash cannot leave a truncated file."""
    info = make_info()
    write_hardware_file(info, recommend_model(info))
    assert list(arc_home_tmp.glob("*.tmp")) == []


def test_write_is_atomic_replacement(arc_home_tmp: Path) -> None:
    info = make_info()
    write_hardware_file(info, recommend_model(info))
    write_hardware_file(make_info(chassis="Second"), recommend_model(info))
    loaded = load_hardware_file()
    assert loaded is not None
    assert loaded["hardware"]["chassis"] == "Second"


def test_load_returns_none_when_absent(arc_home_tmp: Path) -> None:
    assert load_hardware_file() is None


def test_load_returns_none_on_corrupt_json(arc_home_tmp: Path) -> None:
    """Corruption must not crash `arc doctor` — it should re-probe instead."""
    target = hardware_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{ truncated", encoding="utf-8")
    assert load_hardware_file() is None


def test_write_failure_raises(arc_home_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("arc.hardware.os.replace", boom)
    info = make_info()
    with pytest.raises(HardwareProbeError, match="could not write"):
        write_hardware_file(info, recommend_model(info))


def test_refresh_probes_this_machine_and_persists(arc_home_tmp: Path) -> None:
    """The one integration point: a real probe of whatever machine runs the suite."""
    info, sizing, path = refresh()
    assert path.is_file()
    assert info.ram_total_gb > 0
    assert sizing.params_label
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["hardware"]["schema_version"] == HARDWARE_SCHEMA_VERSION
