"""Linux implementation — stubbed.

Not on the brief's roadmap, but the stub exists for two practical reasons: the rented
GPU instances for Track B pretraining will be Linux, and CI may eventually run there.
When that lands, ``/proc`` covers the probe and ``nvidia-smi`` covers CUDA detection.
"""

from __future__ import annotations

from arc.errors import UnsupportedPlatformError
from arc.platform.base import HardwareInfo, Platform

_NOT_YET = "Linux support is stubbed; needed for Track B GPU instances."


class LinuxPlatform(Platform):
    """Placeholder so the platform factory resolves on Linux."""

    @property
    def name(self) -> str:
        return "linux"

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
