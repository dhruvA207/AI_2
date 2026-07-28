"""Windows implementation — stubbed until Phase 8.

Intentionally present and intentionally incomplete. Having the file exist with the
right shape means the factory in ``__init__.py`` resolves on Windows today and
``arc doctor`` can report exactly what is missing, rather than failing with an
import error that says nothing useful.

Phase 8 fills these in: WMI or ``ctypes`` for the hardware probe, ``nvidia-smi`` for
CUDA detection, ``TerminateProcess`` via ``ctypes`` for the kill switch.
"""

from __future__ import annotations

from arc.errors import UnsupportedPlatformError
from arc.platform.base import HardwareInfo, Platform

_NOT_YET = "Windows support is stubbed; scheduled for Phase 8 (see docs/BRIEF.md §5)."


class WindowsPlatform(Platform):
    """Placeholder so the platform factory resolves on Windows."""

    @property
    def name(self) -> str:
        return "windows"

    @property
    def implemented(self) -> bool:
        return False

    def probe_hardware(self) -> HardwareInfo:
        raise UnsupportedPlatformError(_NOT_YET)

    def accelerator_backends(self) -> list[str]:
        raise UnsupportedPlatformError(_NOT_YET)

    def open_application(self, name: str) -> None:
        raise UnsupportedPlatformError(_NOT_YET)

    def kill_process_tree(self, pid: int) -> list[int]:
        raise UnsupportedPlatformError(_NOT_YET)
