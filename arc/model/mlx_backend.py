"""MLX backend — the Apple Silicon fast path.

MLX is Apple's array framework; it runs on the unified memory the GPU already shares
with the CPU, so there is no host-to-device copy and a 4-bit model starts generating
almost immediately. On this machine it is meaningfully faster than llama.cpp's Metal
path, which is why it exists as a separate backend rather than a flag.

It is also Apple-Silicon-only. Nothing here may be imported on Windows, which is why
``arc/model/router.py`` decides and this module is never imported directly by callers.
``mlx-lm`` is MIT (docs/DEPENDENCIES.md).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from arc.errors import ModelError
from arc.log import get_logger
from arc.model.base import (
    Completion,
    FinishReason,
    LanguageModel,
    Message,
    ModelCapabilities,
    Token,
    ToolSchema,
    Usage,
)

if TYPE_CHECKING:  # pragma: no cover - import-time only
    pass

_log = get_logger(__name__)

#: mlx-lm reports its own reason strings; map them onto ours rather than leaking a
#: backend vocabulary into the interface.
_FINISH_REASONS: dict[str, FinishReason] = {"stop": "stop", "length": "length"}


class MLXModel(LanguageModel):
    """A GGUF-free MLX model loaded from a local path or a Hugging Face repo id."""

    def __init__(
        self,
        repo: str,
        *,
        context_length: int,
        capabilities: ModelCapabilities,
        name: str | None = None,
    ) -> None:
        """Load the model and tokenizer.

        The import is inside ``__init__`` rather than at module scope so that merely
        importing this module on a machine without MLX raises nothing — the router
        needs to be able to reason about backends it cannot instantiate.
        """
        try:
            from mlx_lm import load
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ModelError(
                "mlx-lm is not installed. Install it with: pip install 'arc[mlx]'"
            ) from exc

        self._repo = repo
        self._name = name or repo
        self._context_length = context_length
        self._capabilities = capabilities

        try:
            # load() returns (model, tokenizer), or a 3-tuple when return_config is
            # set. Unpack positionally so a future signature change surfaces here
            # rather than as an attribute error during the first generation.
            loaded = load(repo)
            self._model, self._tokenizer = loaded[0], loaded[1]
        except Exception as exc:
            raise ModelError(f"could not load MLX model {repo!r}: {exc}") from exc

        _log.info("loaded MLX model", extra={"repo": repo, "context": context_length})

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
        """Count with the model's real tokenizer, not an estimate."""
        return len(self._tokenizer.encode(text))

    def count_message_tokens(self, messages: list[Message]) -> int:
        """Count by applying the model's actual chat template.

        Overrides the base class's per-message allowance: the template is what really
        gets sent, and its scaffolding is far from a constant 4 tokens per message.
        """
        return len(self._tokenizer.encode(self._render(messages, tools=None)))

    def _render(self, messages: list[Message], tools: list[ToolSchema] | None) -> str:
        """Apply the model's chat template, including tools when it supports them.

        Falls back to a plain role-prefixed transcript if the tokenizer has no
        template. That fallback is what a from-scratch model will need, so it is worth
        having even though every model we ship today has a template.
        """
        payload = [m.to_dict() for m in messages]
        template = getattr(self._tokenizer, "apply_chat_template", None)
        if template is None:
            return self._render_plain(messages)

        kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        if tools and self._capabilities.native_tool_calling:
            kwargs["tools"] = [t.to_dict() for t in tools]

        try:
            rendered = template(payload, **kwargs)
        except Exception as exc:
            _log.warning("chat template failed, falling back to plain transcript: %s", exc)
            return self._render_plain(messages)
        return str(rendered)

    @staticmethod
    def _render_plain(messages: list[Message]) -> str:
        """Role-prefixed transcript, for models without a chat template."""
        parts = [f"{m.role}: {m.content}" for m in messages]
        parts.append("assistant:")
        return "\n\n".join(parts)

    def _sampler(self, temperature: float) -> Any:
        """Build a sampler. Temperature 0 means greedy, which mlx-lm encodes as temp=0."""
        from mlx_lm.sample_utils import make_sampler

        return make_sampler(temp=temperature)

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Completion:
        """Generate by draining the stream.

        Implemented on top of ``stream`` so stop-sequence handling and token accounting
        exist in exactly one place; a second implementation would drift.
        """
        chunks: list[str] = []
        finish: FinishReason = "stop"
        completion_tokens = 0

        for token in self._stream_raw(
            messages, tools=tools, max_tokens=max_tokens, temperature=temperature, stop=stop
        ):
            chunks.append(token.text)
            completion_tokens += 1
            if token.finish_reason is not None:
                finish = token.finish_reason

        prompt_tokens = len(self._tokenizer.encode(self._render(messages, tools)))
        return Completion(
            text="".join(chunks),
            finish_reason=finish,
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
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
        return self._stream_raw(
            messages, tools=tools, max_tokens=max_tokens, temperature=temperature, stop=stop
        )

    def _stream_raw(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None,
        max_tokens: int,
        temperature: float,
        stop: list[str] | None,
    ) -> Iterator[Token]:
        """The single generation path both public methods use.

        Stop sequences are the subtle part. A stop string can straddle a token
        boundary — "STOPHERE" may well arrive as "STOP" then "HERE" — so checking each
        token in isolation misses it, and emitting each token as it arrives leaks the
        first half before we know it was a stop.

        So we withhold a tail. Anything within ``max(len(s)) - 1`` characters of the
        end could still turn out to be the start of a stop sequence, and is held back
        until either more text proves it innocent or a match proves it guilty. Only
        text that can no longer be part of a stop sequence is emitted.
        """
        from mlx_lm import stream_generate

        prompt = self._render(messages, tools)
        # Longest stop sequence determines how much tail we must withhold. Clamped at
        # zero so a degenerate stop list (only empty strings) withholds nothing rather
        # than computing a negative hold.
        hold = max(max((len(s) for s in stop if s), default=0) - 1, 0) if stop else 0

        accumulated = ""
        emitted = 0
        # Why generation ended, captured from the backend rather than assumed. Without
        # this the withheld-tail flush below would report "stop" for a run that
        # actually hit max_tokens, and a caller cannot distinguish a finished answer
        # from one truncated mid-sentence.
        ended: FinishReason = "stop"

        try:
            for response in stream_generate(
                self._model,
                self._tokenizer,
                prompt,
                max_tokens=max_tokens,
                sampler=self._sampler(temperature),
            ):
                accumulated += response.text

                if stop:
                    hit = self._first_stop(accumulated, stop)
                    if hit is not None:
                        if hit > emitted:
                            yield Token(text=accumulated[emitted:hit], finish_reason="stop")
                        else:
                            yield Token(text="", finish_reason="stop")
                        return

                    safe = max(len(accumulated) - hold, emitted)
                    if safe > emitted:
                        yield Token(text=accumulated[emitted:safe])
                        emitted = safe

                    if response.finish_reason:
                        ended = _FINISH_REASONS.get(response.finish_reason, "stop")
                        break
                    continue

                reason = _FINISH_REASONS.get(response.finish_reason or "")
                yield Token(text=response.text, id=response.token, finish_reason=reason)
        except Exception as exc:
            raise ModelError(f"generation failed on {self._name!r}: {exc}") from exc

        # Generation ended without hitting a stop sequence: flush whatever was withheld,
        # reporting why it really ended rather than assuming a clean stop.
        if stop:
            yield Token(text=accumulated[emitted:], finish_reason=ended)

    @staticmethod
    def _first_stop(text: str, stop: list[str]) -> int | None:
        """Return the earliest index at which any stop sequence begins, or None."""
        hits = [text.index(s) for s in stop if s and s in text]
        return min(hits) if hits else None
