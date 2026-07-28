"""ARC — a local-first, fully-owned AI assistant.

The package is deliberately layered so the two load-bearing abstractions from the
brief stay clean: ``arc.model`` (the swappable brain, Phase 2) and ``arc.platform``
(the OS abstraction, so the macOS -> Windows move is a port and not a rewrite).
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
