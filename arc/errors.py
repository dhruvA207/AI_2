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


class ControlError(ArcError):
    """ARC does not have input control, or it was taken back mid-task.

    Raised before every synthetic mouse or keyboard event rather than only at the
    start, so the user reclaiming the pointer stops the *next* event rather than
    being noticed some time later.
    """


class WebError(ArcError):
    """A page could not be fetched, or was refused.

    Covers network failures, HTTP errors, and robots.txt refusals alike. All of them
    are things the agent should report and route around rather than crash on, so they
    surface as observations the same way tool errors do.
    """


class ToolError(ArcError):
    """A tool could not run, or ran and failed.

    The one error class the agent loop treats as *recoverable*: it is fed back to the
    model as an observation so it can adapt, rather than aborting the task. That is
    the distinction the whole hierarchy exists to draw (see the module docstring).
    """


class ModelError(ArcError):
    """A language model could not be loaded, or generation failed.

    Covers both "the weights are missing" and "the backend blew up mid-generation".
    The agent loop treats these as retryable at a different level than a tool error:
    a tool failure becomes an observation the model reasons about, whereas a model
    failure means there is nothing left to do the reasoning.
    """


class MemoryError(ArcError):
    """The memory store could not be opened, read, or written.

    Named to match the subsystem rather than avoiding the builtin clash: callers
    import it as ``from arc.errors import MemoryError`` inside modules that never
    raise the builtin, and ``ArcMemoryError`` would be the only alias in the hierarchy
    that did not match its subsystem.
    """


class AuditError(ArcError):
    """The audit log could not be written.

    Treated as fatal rather than ignored: an action log with silent gaps is worse
    than no action log, because it invites false confidence when debugging.
    """
