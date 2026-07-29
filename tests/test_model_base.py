"""Tests for the LanguageModel interface and its value types.

The point of these is that the interface stays implementable. If satisfying
``LanguageModel`` for a trivial echo model needs anything clever, a from-scratch model
will not manage it either (§4.1).
"""

from __future__ import annotations

import pytest

from arc.model.base import (
    LanguageModel,
    Message,
    ModelCapabilities,
    ToolCall,
    ToolSchema,
    Usage,
)
from tests.fakes import FakeModel


def test_fake_model_satisfies_the_interface() -> None:
    """A ~60-line echo model implements it fully. That is the design constraint."""
    assert isinstance(FakeModel(), LanguageModel)


def test_interface_is_five_members() -> None:
    """Guards against the interface widening by accident.

    Every added abstract member is another thing a from-scratch model must implement
    before it can drive the agent. Changing this number should be a deliberate act.
    """
    assert LanguageModel.__abstractmethods__ == frozenset(
        {"name", "generate", "stream", "count_tokens", "context_length", "capabilities"}
    )


def test_message_to_dict_omits_empty_optionals() -> None:
    assert Message(role="user", content="hi").to_dict() == {"role": "user", "content": "hi"}


def test_message_to_dict_includes_tool_fields() -> None:
    payload = Message(role="tool", content="42", name="calc", tool_call_id="c1").to_dict()
    assert payload["name"] == "calc"
    assert payload["tool_call_id"] == "c1"


def test_tool_schema_serializes_to_function_shape() -> None:
    schema = ToolSchema(name="read", description="Read a file", parameters={"type": "object"})
    payload = schema.to_dict()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "read"
    assert payload["function"]["parameters"] == {"type": "object"}


def test_usage_totals() -> None:
    assert Usage(prompt_tokens=10, completion_tokens=5).total_tokens == 15


def test_capabilities_default_conservatively() -> None:
    """Opt-in, not opt-out: a backend that declares nothing gets the safe path."""
    caps = ModelCapabilities(max_context=2048)
    assert caps.native_tool_calling is False
    assert caps.vision is False
    assert caps.json_mode is False
    assert caps.thinking is False


def test_capabilities_roundtrip() -> None:
    caps = ModelCapabilities(max_context=100, native_tool_calling=True)
    assert caps.to_dict()["native_tool_calling"] is True
    assert caps.to_dict()["max_context"] == 100


def test_generate_returns_usage_and_text() -> None:
    model = FakeModel("one two three")
    result = model.generate([Message(role="user", content="go")])
    assert result.text == "one two three"
    assert result.finish_reason == "stop"
    assert result.usage.completion_tokens == 3


def test_stream_yields_tokens_that_concatenate_to_the_completion() -> None:
    model = FakeModel("alpha beta gamma")
    streamed = "".join(t.text for t in model.stream([Message(role="user", content="go")]))
    generated = model.generate([Message(role="user", content="go")]).text
    assert streamed == generated


def test_final_streamed_token_carries_finish_reason() -> None:
    """A streaming caller must learn why generation ended without a second call."""
    tokens = list(FakeModel("a b").stream([Message(role="user", content="go")]))
    assert tokens[-1].finish_reason == "stop"
    assert all(t.finish_reason is None for t in tokens[:-1])


def test_stop_sequence_truncates() -> None:
    model = FakeModel("keep this STOP drop this")
    assert "drop" not in model.generate([Message(role="user", content="go")], stop=["STOP"]).text


def test_count_message_tokens_default_adds_scaffolding_allowance() -> None:
    """The base implementation budgets for chat-template overhead, not just content."""
    model = FakeModel()
    messages = [Message(role="user", content="one two")]
    assert model.count_message_tokens(messages) > model.count_tokens("one two")


def test_model_records_what_it_was_asked() -> None:
    model = FakeModel()
    model.generate([Message(role="system", content="be terse"), Message(role="user", content="hi")])
    assert [m.role for m in model.calls[0]] == ["system", "user"]


def test_repr_is_useful() -> None:
    assert "fake" in repr(FakeModel(name="fake"))


def test_tool_call_holds_parsed_arguments() -> None:
    """By the time a ToolCall exists, arguments are structured — never a JSON string."""
    call = ToolCall(id="1", name="read", arguments={"path": "/tmp/x"})
    assert call.arguments["path"] == "/tmp/x"


def test_value_types_are_immutable() -> None:
    message = Message(role="user", content="hi")
    with pytest.raises(AttributeError):
        message.content = "changed"  # type: ignore[misc]
