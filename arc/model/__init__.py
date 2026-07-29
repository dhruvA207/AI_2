"""The swappable brain.

Only ``base`` is imported eagerly. Backends pull in heavy, platform-specific
dependencies — MLX exists on Apple Silicon and nowhere else, llama.cpp needs a compiled
extension — so importing this package must not require any of them to be installed.
Use ``arc.model.router.load_model()`` to get a concrete backend.
"""

from arc.model.base import (
    Completion,
    FinishReason,
    LanguageModel,
    Message,
    ModelCapabilities,
    Role,
    Token,
    ToolCall,
    ToolSchema,
    Usage,
)

__all__ = [
    "Completion",
    "FinishReason",
    "LanguageModel",
    "Message",
    "ModelCapabilities",
    "Role",
    "Token",
    "ToolCall",
    "ToolSchema",
    "Usage",
]
