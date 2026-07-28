"""Exception hierarchy for ARC.

Why a hierarchy rather than bare ``Exception``: the brief asks us to fail loudly but
recover gracefully. The agent loop (Phase 4) needs to tell apart errors it can feed
back to the model as an observation and retry (``ToolError``) from errors that mean
the process is misconfigured and retrying is pointless (``ConfigError``). A flat
exception space makes that distinction impossible to draw.
"""

from __future__ import annotations


class ArcError(Exception):
    """Base for every error ARC raises deliberately.

    Catching this catches our errors without also swallowing genuine bugs like
    ``AttributeError``, which we always want to surface.
    """


class ConfigError(ArcError):
    """Configuration is missing, malformed, or internally inconsistent.

    Not retryable. The process should report clearly and exit.
    """


class PlatformError(ArcError):
    """An OS-level operation failed, or was attempted on an unsupported platform."""


class UnsupportedPlatformError(PlatformError):
    """ARC is running somewhere without an implementation yet.

    Distinct from ``PlatformError`` so that ``arc doctor`` can say "Windows support
    is stubbed" rather than "an OS call failed", which are very different messages.
    """


class HardwareProbeError(ArcError):
    """The hardware probe could not determine something downstream code depends on."""


class AuditError(ArcError):
    """The audit log could not be written.

    Treated as fatal rather than ignored: an action log with silent gaps is worse
    than no action log, because it invites false confidence when debugging.
    """
