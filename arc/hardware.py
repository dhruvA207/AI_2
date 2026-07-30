"""Hardware probe and model sizing.

The brief makes this the first thing built in Phase 1 for a reason: everything
downstream — which model to pull, how hard to quantize it, what batch size to use —
should read measured facts from ``~/.arc/hardware.json`` instead of assuming anything
about the machine. This module produces that file.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from arc.errors import HardwareProbeError
from arc.paths import arc_home, hardware_file
from arc.platform import HardwareInfo, get_platform

#: Sizing buckets from docs/BRIEF.md §2, keyed on *total* machine memory.
#: Each row is (exclusive upper bound in GB, label, billions of params, quantization).
#: The final row is the catch-all for very large machines.
_SIZING_TABLE: tuple[tuple[float, str, float, str], ...] = (
    (12.0, "3-4B", 4.0, "4-bit"),
    (24.0, "7-8B", 8.0, "4-bit"),
    (48.0, "14B", 14.0, "4-bit"),
    (96.0, "30B-class MoE", 30.0, "4-bit"),
    (float("inf"), "70B", 70.0, "4-bit"),
)

_BITS = {"4-bit": 4, "8-bit": 8, "16-bit": 16}

#: Rough resident cost of a quantized vision-language model, in GB. Used only to warn
#: when the main model plus a VLM will not co-reside — the situation on a 16 GB Mac.
_VLM_APPROX_GB = 4.0

#: Rough resident cost of a small local embedding model, in GB.
_EMBEDDER_APPROX_GB = 0.5


@dataclass(frozen=True, slots=True)
class ModelSizing:
    """What this machine can realistically run, and the caveats that come with it."""

    params_label: str
    quantization: str
    approx_weights_gb: float
    usable_memory_gb: float
    headroom_gb: float
    can_coreside_vlm: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


def _weights_gb(params_b: float, quantization: str) -> float:
    """Estimate resident weight size in GB.

    ``params x bits / 8`` gives the raw tensor bytes; the 15% uplift covers the
    unquantized embedding and norm layers plus allocator overhead, which real GGUF
    and MLX files consistently show.
    """
    bits = _BITS.get(quantization, 4)
    raw_gb = params_b * 1e9 * bits / 8 / 1024**3
    return round(raw_gb * 1.15, 1)


def probe() -> HardwareInfo:
    """Inspect the current machine.

    Delegates to the platform layer so this function stays identical on Windows.
    """
    return get_platform().probe_hardware()


def _sizing_settings() -> dict[str, float]:
    """Read the sizing knobs from config.

    These were declared in ``config/default.yaml`` and then read from module constants
    instead, so editing them did nothing at all — a silent no-op, which is worse than
    not offering the setting. Falls back to the historical defaults when config cannot
    be loaded, since sizing must still work before anything else is set up.
    """
    try:
        from arc.config import Config

        section = Config.load().section("hardware")
    except Exception:
        section = {}

    return {
        "vlm_gb": float(section.get("vlm_estimate_gb", _VLM_APPROX_GB)),
        "embedder_gb": float(section.get("embedder_estimate_gb", _EMBEDDER_APPROX_GB)),
        "min_headroom_gb": float(section.get("min_headroom_gb", 2.0)),
    }


def recommend_model(info: HardwareInfo) -> ModelSizing:
    """Pick a model size for this machine, with honest caveats.

    The bucket comes from the brief's table, but the caveats come from measured
    headroom. That distinction matters: the table says a 16 GB machine runs an 8B at
    4-bit, and it does — right up until you also want a vision model resident, at
    which point the machine starts swapping. Callers deserve to know that up front
    rather than discovering it as mysterious slowness.
    """
    total = info.ram_total_gb
    label, params_b, quant = next(
        (lbl, p, q) for bound, lbl, p, q in _SIZING_TABLE if total < bound
    )

    weights = _weights_gb(params_b, quant)
    usable = round(info.model_memory_gb, 1)
    headroom = round(usable - weights, 1)

    notes: list[str] = []
    if info.unified_memory:
        notes.append(
            f"Unified memory: {total:.0f} GB shared with the OS and GPU, "
            f"~{usable:.1f} GB realistically available to a model."
        )

    settings = _sizing_settings()
    coreside_need = weights + settings["vlm_gb"] + settings["embedder_gb"]
    can_coreside = coreside_need <= usable
    if not can_coreside:
        notes.append(
            f"A vision model (~{settings['vlm_gb']:.0f} GB) and embedder "
            f"(~{settings['embedder_gb']:.1f} GB) will not co-reside with a "
            f"{label} model here ({coreside_need:.1f} GB needed, {usable:.1f} GB available). "
            "Phase 2's router must load and evict the VLM around screen tasks."
        )

    if headroom < settings["min_headroom_gb"]:
        notes.append(
            f"Only {headroom:.1f} GB headroom after weights — KV cache growth on long "
            "contexts will be the binding constraint. Consider the next size down."
        )

    if not info.unified_memory and info.vram_gb is None:
        notes.append("No discrete GPU detected; CPU inference will be slow.")

    # Memory decides what *fits*; cooling decides what stays fast. A fanless chassis
    # benchmarks fine for the first few minutes and then loses a third of its
    # throughput, so a table lookup on RAM alone quietly over-promises here.
    if info.fanless:
        chassis = info.chassis or "this machine"
        notes.append(
            f"{chassis} is fanless: sustained throughput drops to roughly 50-70% of peak "
            f"after 10-20 minutes under load. A {label} model fits, but consider the next "
            f"size down for long sessions, and expect any training run to throttle."
        )

    return ModelSizing(
        params_label=label,
        quantization=quant,
        approx_weights_gb=weights,
        usable_memory_gb=usable,
        headroom_gb=headroom,
        can_coreside_vlm=can_coreside,
        notes=notes,
    )


def write_hardware_file(info: HardwareInfo, sizing: ModelSizing) -> Path:
    """Write ``~/.arc/hardware.json`` atomically.

    Temp file plus ``os.replace`` so a crash mid-write can never leave a truncated
    file that downstream code would happily parse as authoritative. The same pattern
    is required for training checkpoints later (brief §6.3).
    """
    target = hardware_file()
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = {"hardware": info.to_dict(), "recommendation": sizing.to_dict()}
    tmp = target.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise HardwareProbeError(f"could not write {target}: {exc}") from exc

    return target


def load_hardware_file() -> dict[str, Any] | None:
    """Read a previously written ``hardware.json``, or None if absent or corrupt.

    Corruption returns None rather than raising so that ``arc doctor`` can report
    "stale or unreadable, re-probing" instead of dying on a bad file.
    """
    target = hardware_file()
    if not target.is_file():
        return None
    try:
        with target.open(encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data


def refresh() -> tuple[HardwareInfo, ModelSizing, Path]:
    """Probe, recommend, and persist in one step.

    The startup path and ``arc probe`` both use this so there is exactly one way the
    file gets written.
    """
    arc_home().mkdir(parents=True, exist_ok=True)
    info = probe()
    sizing = recommend_model(info)
    path = write_hardware_file(info, sizing)
    return info, sizing, path
