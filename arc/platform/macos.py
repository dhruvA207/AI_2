"""macOS implementation of the platform interface.

Everything macOS-specific in ARC should be in this file. It shells out to ``sysctl``,
``system_profiler``, and ``sw_vers`` rather than taking a dependency on ``psutil``:
those tools ship with the OS, the parsing is about fifty lines, and ``psutil`` is
BSD-3-Clause rather than the Apache-2.0/MIT the brief restricts us to (see
docs/DECISIONS.md).
"""

from __future__ import annotations

import json
import os
import platform as _platform
import shutil
import signal
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from arc.errors import PlatformError
from arc.platform.base import HARDWARE_SCHEMA_VERSION, HardwareInfo, Platform

_BYTES_PER_GB = 1024**3

#: system_profiler is slow (~1s) and occasionally hangs on a busy machine, so every
#: call is bounded. A missing GPU section degrades the report; it must not hang ARC.
_PROFILER_TIMEOUT_S = 15.0
_SYSCTL_TIMEOUT_S = 5.0


def _run(cmd: list[str], timeout: float) -> str | None:
    """Run a command and return stripped stdout, or None if it failed.

    Returns None rather than raising because every caller here is collecting
    best-effort diagnostics: one missing field should degrade the hardware report,
    not abort startup.
    """
    try:
        # Fixed argv, never a shell string, so no injection surface.
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _sysctl(key: str) -> str | None:
    """Read a single sysctl value."""
    return _run(["/usr/sbin/sysctl", "-n", key], _SYSCTL_TIMEOUT_S)


def _sysctl_int(key: str) -> int | None:
    """Read a sysctl value as an int, tolerating absence or garbage."""
    raw = _sysctl(key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _gpu_info() -> tuple[str | None, str | None, int | None]:
    """Return ``(vendor, model, core_count)`` from system_profiler.

    Uses the JSON output rather than scraping the human-readable form, which changes
    between macOS releases.
    """
    raw = _run(
        ["/usr/sbin/system_profiler", "-json", "SPDisplaysDataType"],
        _PROFILER_TIMEOUT_S,
    )
    if raw is None:
        return None, None, None
    try:
        data = json.loads(raw)
        displays = data.get("SPDisplaysDataType", [])
    except (json.JSONDecodeError, AttributeError):
        return None, None, None
    if not displays:
        return None, None, None

    entry = displays[0]
    model = entry.get("sppci_model")
    vendor = entry.get("spdisplays_vendor")
    if vendor is None and isinstance(model, str) and model.startswith("Apple"):
        vendor = "Apple"

    cores_raw = entry.get("sppci_cores")
    cores: int | None = None
    if cores_raw is not None:
        try:
            cores = int(str(cores_raw).strip())
        except ValueError:
            cores = None

    return vendor, model, cores


def _chassis() -> str | None:
    """Return the marketing product name, e.g. ``MacBook Air``.

    Read from ``system_profiler`` rather than mapping ``hw.model`` identifiers
    ("Mac15,12") through a lookup table. Apple stopped encoding the product line in
    those identifiers, so a table would need editing for every new machine Apple
    ships — and would silently mis-report an unknown one.
    """
    raw = _run(
        ["/usr/sbin/system_profiler", "-json", "SPHardwareDataType"],
        _PROFILER_TIMEOUT_S,
    )
    if raw is None:
        return None
    try:
        entries = json.loads(raw).get("SPHardwareDataType", [])
    except (json.JSONDecodeError, AttributeError):
        return None
    if not entries:
        return None
    name = entries[0].get("machine_name")
    return str(name) if name else None


def _is_fanless(chassis: str | None, is_apple_silicon: bool) -> bool | None:
    """Whether the machine has no active cooling.

    Every Apple Silicon Mac has a fan except the MacBook Air. Intel Airs did have
    fans, hence the architecture check. Returns None when the chassis is unknown,
    because "we could not tell" and "it has a fan" lead to different advice.
    """
    if chassis is None:
        return None
    return is_apple_silicon and chassis == "MacBook Air"


class MacOSPlatform(Platform):
    """Platform implementation for macOS on Apple Silicon or Intel."""

    @property
    def name(self) -> str:
        return "macos"

    @property
    def implemented(self) -> bool:
        return True

    def accelerator_backends(self) -> list[str]:
        """Return usable inference backends, best first.

        On Apple Silicon, MLX is the fast path and Metal (via llama.cpp) is the
        portable one. Intel Macs get neither and fall back to CPU.
        """
        if _platform.machine() == "arm64":
            return ["mlx", "metal", "cpu"]
        return ["cpu"]

    def probe_hardware(self) -> HardwareInfo:
        """Inspect the machine via the OS's own reporting tools."""
        mem_bytes = _sysctl_int("hw.memsize")
        if mem_bytes is None:
            raise PlatformError("could not read hw.memsize; cannot size a model safely")

        arch = _platform.machine()
        is_apple_silicon = arch == "arm64"
        gpu_vendor, gpu_model, gpu_cores = _gpu_info()

        disk_free_gb: float | None
        try:
            disk_free_gb = round(shutil.disk_usage(Path.home()).free / _BYTES_PER_GB, 1)
        except OSError:
            disk_free_gb = None

        logical = _sysctl_int("hw.logicalcpu") or os.cpu_count() or 1
        physical = _sysctl_int("hw.physicalcpu") or logical
        chassis = _chassis()

        return HardwareInfo(
            schema_version=HARDWARE_SCHEMA_VERSION,
            probed_at=datetime.now(UTC).isoformat(),
            os_name="macos",
            os_version=_platform.mac_ver()[0] or "unknown",
            os_build=_run(["/usr/bin/sw_vers", "-buildVersion"], _SYSCTL_TIMEOUT_S),
            arch=arch,
            python_version=_platform.python_version(),
            cpu_model=_sysctl("machdep.cpu.brand_string") or "unknown",
            cpu_cores_physical=physical,
            cpu_cores_logical=logical,
            cpu_performance_cores=_sysctl_int("hw.perflevel0.logicalcpu"),
            cpu_efficiency_cores=_sysctl_int("hw.perflevel1.logicalcpu"),
            ram_total_gb=round(mem_bytes / _BYTES_PER_GB, 1),
            # Apple Silicon shares one memory pool between CPU and GPU, which is why
            # "VRAM" is meaningless here and headroom matters more than on a discrete GPU.
            unified_memory=is_apple_silicon,
            gpu_vendor=gpu_vendor,
            gpu_model=gpu_model,
            gpu_cores=gpu_cores,
            # Always None on macOS: unified-memory Macs have no separate pool, and
            # system_profiler does not report a usable figure for discrete Intel GPUs
            # either. Sizing goes through model_memory_gb instead.
            vram_gb=None,
            disk_free_gb=disk_free_gb,
            chassis=chassis,
            fanless=_is_fanless(chassis, is_apple_silicon),
            accelerators=self.accelerator_backends(),
        )

    def open_application(self, name: str) -> None:
        """Launch or focus an application using ``open -a``."""
        try:
            # argv form, no shell.
            proc = subprocess.run(
                ["/usr/bin/open", "-a", name],
                capture_output=True,
                text=True,
                timeout=30.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise PlatformError(f"failed to launch {name!r}: {exc}") from exc
        if proc.returncode != 0:
            raise PlatformError(f"failed to launch {name!r}: {proc.stderr.strip()}")

    def kill_process_tree(self, pid: int) -> list[int]:
        """SIGKILL ``pid`` and every descendant, children first.

        Children are killed before parents so a supervisor cannot respawn a child in
        the window between the two signals. Already-dead PIDs are skipped rather than
        raising: the kill switch must succeed even when it has partly already worked.
        """
        pids = self._descendants(pid)
        pids.append(pid)

        killed: list[int] = []
        for target in pids:
            try:
                os.kill(target, signal.SIGKILL)
            except ProcessLookupError:
                continue  # Already gone; nothing to do.
            except PermissionError:
                continue  # Not ours to kill.
            killed.append(target)
        return killed

    def _descendants(self, pid: int) -> list[int]:
        """Return all descendant PIDs, deepest first.

        Walks ``pgrep -P`` breadth-first then reverses, which orders leaves ahead of
        their parents without needing a recursive tree structure.
        """
        ordered: list[int] = []
        frontier = [pid]
        seen = {pid}

        while frontier:
            current = frontier.pop(0)
            raw = _run(["/usr/bin/pgrep", "-P", str(current)], _SYSCTL_TIMEOUT_S)
            if raw is None:
                continue
            for line in raw.splitlines():
                try:
                    child = int(line.strip())
                except ValueError:
                    continue
                if child in seen:
                    continue
                seen.add(child)
                ordered.append(child)
                frontier.append(child)

        ordered.reverse()
        return ordered
