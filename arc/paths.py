"""Canonical filesystem layout for ARC.

Why this exists as its own module: the brief separates the *code* root (``~/arc``,
version controlled) from the *runtime data* root (``~/.arc``, machine-local and never
committed). Keeping every path in one place means the macOS -> Windows migration is a
single edit, and ``pathlib`` everywhere means we never concatenate separators by hand.

``ARC_HOME`` overrides the runtime root, which is what makes the whole tree testable
without touching the real one.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Environment variable that relocates the runtime data root. Tests rely on this.
ARC_HOME_ENV = "ARC_HOME"


def arc_home() -> Path:
    """Return the runtime data root (``~/.arc`` unless ``ARC_HOME`` overrides it).

    Read from the environment on every call rather than cached at import time, because
    tests set ``ARC_HOME`` after the module is already imported.
    """
    override = os.environ.get(ARC_HOME_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".arc"


def project_root() -> Path:
    """Return the repository root — the directory holding ``config/`` and ``docs/``.

    Derived from this file's location rather than the working directory, so ``arc``
    behaves identically no matter where it is invoked from.
    """
    return Path(__file__).resolve().parent.parent


def config_dir() -> Path:
    """Return the directory holding the committed ``*.yaml`` defaults."""
    override = os.environ.get("ARC_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "config"


def audit_dir() -> Path:
    """Return the append-only action log directory (brief §0.3)."""
    return arc_home() / "audit"


def log_dir() -> Path:
    """Return the directory for structured operational logs.

    Kept separate from the audit log on purpose: operational logs are debug noise and
    may be rotated or deleted, while the audit log is an append-only record we never
    truncate.
    """
    return arc_home() / "logs"


def run_dir() -> Path:
    """Return the directory holding PID files for live ARC processes."""
    return arc_home() / "run"


def models_dir() -> Path:
    """Return the local model weight cache (populated in Phase 2)."""
    return arc_home() / "models"


def hardware_file() -> Path:
    """Return the path to ``hardware.json``, written by the probe.

    Everything downstream — model size, quantization, batch size — reads this file
    instead of assuming anything about the machine (brief §2).
    """
    return arc_home() / "hardware.json"


def ensure_runtime_dirs() -> None:
    """Create the runtime tree if it is missing.

    Idempotent and safe to call on every startup.
    """
    for path in (arc_home(), audit_dir(), log_dir(), run_dir(), models_dir()):
        path.mkdir(parents=True, exist_ok=True)
