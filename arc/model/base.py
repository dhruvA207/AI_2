"""The swappable brain.

This is the most important interface in ARC (docs/BRIEF.md §4.1). Every backend
implements it — llama.cpp, MLX, and eventually a model trained from scratch — and the
agent loop is written against nothing else. Swapping the brain is a config change.

**It is deliberately narrow.** The temptation is to expose everything a modern
inference server can do: logit bias, beam search, speculative decoding, structured
output grammars. Every one of those would be another thing a from-scratch model has to
implement before it can drive the agent, and the whole point of Track B is that it
eventually can. So the rule is: if a capability is not required to run the agent loop,
it does not belong here.

Capabilities that genuinely vary between models are *reported* rather than assumed, via
``ModelCapabilities``. That is what lets ``agent/executor.py`` fall back from native
tool calling to a prompted ReAct protocol without the model knowing anything about it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

#: Why generation stopped. ``length`` means the token budget ran out mid-thought, which
#: the caller usually wants to treat differently from a clean ``stop``.
FinishReason = Literal["stop", "length", "tool_call", "error"]


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a conversation.

    ``tool_call_id`` and ``name`` are only meaningful on tool-role messages; they tie a
    result back to the call that produced it. Kept on the one dataclass rather than
    split into a hierarchy because every backend has to serialise this into a flat
    chat template anyway.
    """

    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the OpenAI-style dict shape that most chat templates expect."""
        payload: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


@dataclass(frozen=True, slots=True)
class ToolSchema:
    """A tool the model may call, described as JSON Schema.

    ``parameters`` is a JSON Schema object. Phase 4's registry generates these from
    decorated Python functions; here we only care that they can be handed to a backend
    or rendered into a prompt.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the function-calling shape used by most native tool-calling APIs."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A model's request to invoke a tool.

    ``arguments`` is the parsed object. Backends without native tool calling produce
    these by running the tolerant parser over a fenced JSON block, so by the time a
    ``ToolCall`` exists the arguments are already structured — the executor never has
    to care which path produced it.
    """

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Usage:
    """Token accounting for one generation."""

    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class Completion:
    """The result of a non-streaming generation."""

    text: str
    finish_reason: FinishReason
    usage: Usage
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: Backend-specific extras (timings, sampler state). Never load-bearing — anything
    #: the agent loop needs belongs in a typed field instead.
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Token:
    """One streamed token.

    Carries the decoded text rather than the token id because every consumer so far
    wants to print it. ``id`` is kept for backends that can supply it cheaply.
    """

    text: str
    id: int | None = None
    #: Set on the final token so a streaming caller learns why generation ended
    #: without needing a second call.
    finish_reason: FinishReason | None = None


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """What a particular model can actually do.

    Exists so the agent loop can adapt instead of assuming. The critical one is
    ``native_tool_calling``: small open models and anything from Track B will not have
    it, and §4.1 requires the executor to drop to a prompted ReAct protocol when it is
    False. Defaulting it to False is deliberate — a backend must opt *in*, so a new
    backend that forgets to declare its capabilities degrades to the safe path rather
    than silently emitting tool calls nothing parses.
    """

    max_context: int
    native_tool_calling: bool = False
    vision: bool = False
    json_mode: bool = False
    #: Whether the model emits explicit reasoning blocks (Qwen3's thinking mode).
    #: The agent loop strips these before storing a turn in memory.
    thinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "max_context": self.max_context,
            "native_tool_calling": self.native_tool_calling,
            "vision": self.vision,
            "json_mode": self.json_mode,
            "thinking": self.thinking,
        }


class LanguageModel(ABC):
    """The interface every brain implements.

    Five members. A from-scratch transformer with a tokenizer can satisfy all of them,
    which is the constraint that keeps this honest — see ``arc/model/custom.py`` when
    Track B gets there.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for logs, the audit trail, and ``arc model list``."""

    @abstractmethod
    def generate(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Completion:
        """Generate a single completion.

        ``tools`` may be passed to any backend. One without native tool calling is
        expected to render them into the prompt rather than raise — the fallback is the
        executor's business, but refusing the argument here would push every caller to
        branch on capabilities before it could call at all.
        """

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[Token]:
        """Stream a completion token by token.

        Separate from ``generate`` rather than a flag on it, because the return types
        genuinely differ and a union return would make every call site unwrap it.
        """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in ``text`` using this model's own tokenizer.

        Must be the real tokenizer, not an estimate. Phase 3's working-memory budget
        divides the context window against this number; a heuristic that is 20% off
        produces prompts that overflow only on certain inputs, which is a miserable
        class of bug.
        """

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Usable context window in tokens."""

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """What this model supports."""

    def count_message_tokens(self, messages: list[Message]) -> int:
        """Estimate tokens for a whole conversation.

        Concrete rather than abstract: the default sums the parts and adds a small
        per-message allowance for chat-template scaffolding, which is close enough for
        budgeting. Backends that can apply their real template should override it.
        """
        return sum(self.count_tokens(m.content) + 4 for m in messages)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, context={self.context_length})"
