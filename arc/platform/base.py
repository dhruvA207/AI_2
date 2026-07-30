"""The OS abstraction boundary.

Why this is load-bearing: the brief says a Windows move is coming, so no business
logic may ever call an OS-specific API directly. Everything platform-shaped goes
behind this ABC, and ``arc/platform/__init__.py`` picks an implementation at runtime.
The rule of thumb from the brief: if you need ``osascript``, it belongs in
``macos.py`` behind a name like ``open_application()`` that ``windows.py`` satisfies
its own way.

``HardwareInfo`` lives here rather than in ``arc/hardware.py`` because it is the
*output* of a platform call. Putting it here keeps the dependency pointing one way
(``hardware`` -> ``platform``) instead of forming a cycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

#: Bumped whenever the shape of hardware.json changes, so a stale file written by an
#: older ARC can be detected and re-probed rather than silently misread.
#: v2 added ``chassis`` and ``fanless``.
HARDWARE_SCHEMA_VERSION = 2


def _reserve_settings() -> tuple[float, float]:
    """Return (floor GB, fraction) reserved for the OS, from config.

    Defaults match the historical constants so sizing still works before config exists
    — the probe runs on a fresh install where nothing is set up yet.
    """
    try:
        from arc.config import Config

        section = Config.load().section("hardware")
        return (
            float(section.get("os_reserve_gb", 4.0)),
            float(section.get("os_reserve_fraction", 0.28)),
        )
    except Exception:
        return (4.0, 0.28)


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """A snapshot of the machine ARC is running on.

    Serialized to ``~/.arc/hardware.json``. Downstream code reads that file rather
    than re-probing or, worse, assuming — model size, quantization, and batch size
    all derive from these numbers.
    """

    schema_version: int
    probed_at: str

    os_name: str
    os_version: str
    os_build: str | None
    arch: str
    python_version: str

    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    cpu_performance_cores: int | None
    cpu_efficiency_cores: int | None

    ram_total_gb: float
    unified_memory: bool

    gpu_vendor: str | None
    gpu_model: str | None
    gpu_cores: int | None
    vram_gb: float | None

    disk_free_gb: float | None

    #: Human-readable product line, e.g. "MacBook Air". None when the platform cannot
    #: determine it.
    chassis: str | None = None

    #: Whether this machine has no active cooling. Load-bearing for sizing: a fanless
    #: chassis sustains only 50-70% of peak once it heats up, so a model that
    #: benchmarks well in a burst will disappoint over a long session. None means
    #: unknown, which callers must treat differently from False.
    fanless: bool | None = None

    #: Inference backends this machine can actually run, best first.
    accelerators: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)

    @property
    def model_memory_gb(self) -> float:
        """Memory a model may realistically claim, in GB.

        Deliberately not the same as total RAM. On unified-memory Macs the GPU and OS
        share one pool, so a model that "fits" on paper will swap in practice. We
        reserve headroom for the OS and other applications; the brief's sizing table
        is then applied to what is left.
        """
        if self.unified_memory:
            # macOS itself plus a browser and an editor comfortably occupy ~5 GB.
            # Read from config: these were declared there and ignored, so tuning them
            # silently did nothing.
            floor, fraction = _reserve_settings()
            reserve = max(floor, self.ram_total_gb * fraction)
            return max(0.0, self.ram_total_gb - reserve)
        if self.vram_gb:
            return self.vram_gb
        return max(0.0, self.ram_total_gb - 4.0)


class Platform(ABC):
    """Everything ARC needs from the host operating system.

    Kept small on purpose. Filesystem, shell, screen, and input control arrive in
    later phases; adding their methods now would be speculative design against
    requirements we have not met yet.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short platform identifier: ``macos``, ``windows``, or ``linux``."""

    @property
    @abstractmethod
    def implemented(self) -> bool:
        """Whether this platform is actually usable yet.

        ``arc doctor`` reports this so a stubbed platform produces "Windows support is
        not implemented yet" instead of a confusing crash deep in a probe.
        """

    @abstractmethod
    def probe_hardware(self) -> HardwareInfo:
        """Inspect the machine and return a fresh snapshot."""

    @abstractmethod
    def accelerator_backends(self) -> list[str]:
        """Return available inference backends, best first.

        Names are stable strings (``mlx``, ``metal``, ``cuda``, ``cpu``) that the
        Phase 2 model router matches against.
        """

    @abstractmethod
    def open_application(self, name: str) -> None:
        """Launch or focus an application by name.

        The canonical example of the abstraction: macOS shells out to ``open``,
        Windows will use ``start`` or the shell API, and no caller needs to know.
        """

    @abstractmethod
    def kill_process_tree(self, pid: int) -> list[int]:
        """SIGKILL a process and all its descendants; return the PIDs signalled.

        Used by the kill switch. Must not raise when a PID has already exited — a
        kill switch that fails because it was too effective is not a kill switch.
        """
