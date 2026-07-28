"""Runtime selection of the platform implementation.

Business logic calls ``get_platform()`` and never imports ``macos`` or ``windows``
directly. That single rule is what keeps the Windows port from becoming a rewrite.

The result is cached because a process cannot change operating system mid-run, and
the probe underneath is not free.
"""

from __future__ import annotations

import sys
from functools import lru_cache

from arc.errors import UnsupportedPlatformError
from arc.platform.base import HardwareInfo, Platform

__all__ = ["HardwareInfo", "Platform", "get_platform", "platform_name"]


def platform_name() -> str:
    """Return the short name of the current platform.

    Split out from ``get_platform()`` so callers that only need to *report* the
    platform do not pay for instantiating it.
    """
    # Read through a local to defeat mypy's static narrowing of sys.platform. On a
    # darwin checkout mypy proves the later branches unreachable and errors on them,
    # but they are exactly the branches that must survive for the Windows port.
    current: str = sys.platform
    if current == "darwin":
        return "macos"
    if current == "win32":
        return "windows"
    if current.startswith("linux"):
        return "linux"
    return current


@lru_cache(maxsize=1)
def get_platform() -> Platform:
    """Return the platform implementation for the host OS.

    Imports are deferred into the branches so that importing, say, the macOS module
    on Windows can never happen — the modules are free to use OS-specific imports at
    module scope in future without breaking other platforms.
    """
    name = platform_name()

    if name == "macos":
        from arc.platform.macos import MacOSPlatform

        return MacOSPlatform()

    if name == "windows":
        from arc.platform.windows import WindowsPlatform

        return WindowsPlatform()

    if name == "linux":
        from arc.platform.linux import LinuxPlatform

        return LinuxPlatform()

    raise UnsupportedPlatformError(f"no platform implementation for {name!r}")
