"""llama.cpp backend — the portable path.

Runs GGUF weights on CPU, Metal, or CUDA, which makes it the backend that survives the
Windows move unchanged. It is slower than MLX on Apple Silicon, so on this machine it
is the fallback rather than the default; on a CUDA box it becomes the primary.

``llama-cpp-python`` is MIT (docs/DEPENDENCIES.md). It is an optional dependency —
importing this module without it installed raises a clear ``ModelError`` rather than an
``ImportError`` from three frames down.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from arc.errors import ModelError
from arc.log import get_logger
from arc.model.base import (
    Completion,
    FinishReason,
    LanguageModel,
    Message,
    ModelCapabilities,
    Token,
    ToolCall,
    ToolSchema,
    Usage,
)

_log = get_logger(__name__)

#: -1 offloads every layer to the GPU. llama.cpp silently falls back to CPU for what
#: does not fit, so this is safe to request even on a small GPU.
_ALL_LAYERS_ON_GPU = -1


class LlamaCppModel(LanguageModel):
    """A GGUF model served by llama.cpp."""

    def __init__(
        self,
        model_path: Path,
        *,
        context_length: int,
        capabilities: ModelCapabilities,
        name: str | None = None,
        n_gpu_layers: int = _ALL_LAYERS_ON_GPU,
        n_threads: int | None = None,
    ) -> None:
        """Load a GGUF file.

        The import is deferred into the constructor so the router can reason about this
        backend on a machine where ``llama-cpp-python`` is not installed.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ModelError(
                "llama-cpp-python is not installed. Install it with: pip install 'arc[llamacpp]'"
            ) from exc

        if not model_path.is_file():
            raise ModelError(f"GGUF file not found: {model_path}")

        self._name = name or model_path.stem
        self._context_length = context_length
        self._capabilities = capabilities

        try:
            self._llama = Llama(
                model_path=str(model_path),
                n_ctx=context_length,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                verbose=False,
            )
        except Exception as exc:
            raise ModelError(f"could not load GGUF model {model_path}: {exc}") from exc

        _log.info(
            "loaded llama.cpp model",
            extra={"path": str(model_path), "context": context_length},
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    def count_tokens(self, text: str) -> int:
        """Count with llama.cpp's own tokenizer."""
        return len(self._llama.tokenize(text.encode("utf-8"), add_bos=False))

    def _request(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
        stream: bool,
    ) -> Any:
        """Build and issue a chat-completion call.

        Tools are only forwarded when the model declares native tool calling. Passing
        them to a model without it makes llama.cpp inject a template the model was
        never trained on, which produces worse output than leaving the prompted ReAct
        fallback to handle it.
        """
        kwargs: dict[str, Any] = {
            "messages": [m.to_dict() for m in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if stop:
            kwargs["stop"] = stop
        if tools and self._capabilities.native_tool_calling:
            kwargs["tools"] = [t.to_dict() for t in tools]

        try:
            return self._llama.create_chat_completion(**kwargs)
        except Exception as exc:
            raise ModelError(f"generation failed on {self._name!r}: {exc}") from exc

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Completion:
        """Generate a single completion."""
        raw = self._request(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=False,
        )

        choice = raw["choices"][0]
        message = choice.get("message", {})
        usage = raw.get("usage", {})

        return Completion(
            text=message.get("content") or "",
            finish_reason=_map_finish(choice.get("finish_reason")),
            usage=Usage(
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
            ),
            tool_calls=_parse_tool_calls(message.get("tool_calls")),
        )

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[Token]:
        """Stream tokens as they are produced."""
        raw = self._request(
            messages,
            tools=tools,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            stream=True,
        )

        for chunk in raw:
            choice = chunk["choices"][0]
            delta = choice.get("delta", {})
            text = delta.get("content")
            reason = choice.get("finish_reason")
            if text is None and reason is None:
                # Role-announcement chunk carries no content; nothing to emit.
                continue
            yield Token(
                text=text or "",
                finish_reason=_map_finish(reason) if reason else None,
            )


def _map_finish(reason: str | None) -> FinishReason:
    """Translate llama.cpp's finish reason into ours."""
    if reason == "length":
        return "length"
    if reason == "tool_calls":
        return "tool_call"
    return "stop"


def _parse_tool_calls(raw: list[dict[str, Any]] | None) -> list[ToolCall]:
    """Convert native tool calls into our shape.

    Arguments arrive as a JSON *string*. A malformed one is dropped with a warning
    rather than raising: the executor's tolerant parser (Phase 4) is the right place to
    attempt repair, and losing one call is better than losing the whole completion.
    """
    if not raw:
        return []

    import json

    calls: list[ToolCall] = []
    for entry in raw:
        function = entry.get("function", {})
        name = function.get("name")
        if not name:
            continue
        try:
            arguments = json.loads(function.get("arguments") or "{}")
        except json.JSONDecodeError:
            _log.warning("dropping tool call with unparseable arguments", extra={"tool": name})
            continue
        if not isinstance(arguments, dict):
            _log.warning("dropping tool call with non-object arguments", extra={"tool": name})
            continue
        calls.append(
            ToolCall(
                id=str(entry.get("id") or uuid.uuid4().hex[:12]), name=name, arguments=arguments
            )
        )
    return calls
