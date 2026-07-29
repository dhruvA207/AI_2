"""Capabilities.

Importing this package registers every built-in tool on the shared registry. The
imports look unused and are not: each module's ``@tool`` decorators run at import time,
and dropping them would silently produce an agent with no tools.

Access is unrestricted by design (§0.3) — no permission prompts, no deny-list. The
safeguards are the audit log, the kill switch, and ``--dry-run``.
"""

from arc.tools import filesystem, shell, web  # noqa: F401 - imported for registration
from arc.tools.registry import Tool, ToolRegistry, registry, tool

__all__ = ["Tool", "ToolRegistry", "registry", "tool"]
