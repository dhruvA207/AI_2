"""A fake LanguageModel for tests.

The suite must not download 2 GB of weights or need Apple Silicon, so everything that
exercises the model layer runs against this instead. It also serves as a check on the
interface itself: if implementing ``LanguageModel`` for a trivial echo model is
awkward, the interface is too wide for a from-scratch model to satisfy later (§4.1).
"""

from __future__ import annotations

from collections.abc import Iterator

from arc.model.base import (
    Completion,
    LanguageModel,
    Message,
    ModelCapabilities,
    Token,
    ToolSchema,
    Usage,
)


class FakeModel(LanguageModel):
    """Returns scripted text, one whitespace-delimited word per streamed token."""

    def __init__(
        self,
        reply: str = "hello there friend",
        *,
        name: str = "fake",
        context_length: int = 1024,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self._reply = reply
        self._name = name
        self._context_length = context_length
        self._capabilities = capabilities or ModelCapabilities(max_context=context_length)
        #: Recorded so tests can assert on what the caller actually asked for.
        self.calls: list[list[Message]] = []

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
        """One token per whitespace-delimited word — enough for budget arithmetic."""
        return len(text.split())

    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Completion:
        self.calls.append(list(messages))
        text = "".join(t.text for t in self._tokens(stop))
        return Completion(
            text=text,
            finish_reason="stop",
            usage=Usage(
                prompt_tokens=self.count_message_tokens(messages),
                completion_tokens=self.count_tokens(text),
            ),
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
        self.calls.append(list(messages))
        return iter(self._tokens(stop))

    def _tokens(self, stop: list[str] | None) -> list[Token]:
        """Split the scripted reply into tokens, truncating at a stop sequence."""
        text = self._reply
        if stop:
            hits = [text.index(s) for s in stop if s and s in text]
            if hits:
                text = text[: min(hits)]

        words = text.split()
        tokens = [Token(text=w + " ") for w in words[:-1]]
        if words:
            tokens.append(Token(text=words[-1], finish_reason="stop"))
        else:
            tokens.append(Token(text="", finish_reason="stop"))
        return tokens


class ExplodingModel(FakeModel):
    """Raises on generation, for testing error paths in the REPL."""

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[Token]:
        from arc.errors import ModelError

        raise ModelError("backend exploded")
