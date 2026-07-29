"""Tests for the MLX backend's stop-sequence and finish-reason handling.

The generation loop is exercised against a stubbed ``stream_generate`` rather than a
real model, so these run anywhere and in milliseconds. This is the fiddliest logic in
the backend and the part that has already produced two bugs, so it earns direct tests
rather than being covered incidentally.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from arc.model.base import Message, ModelCapabilities, Token


@dataclass
class FakeResponse:
    """Mimics mlx_lm's GenerationResponse."""

    text: str
    finish_reason: str | None = None
    token: int = 0


class StubbedMLXModel:
    """The real ``_stream_raw`` bound to a scripted token sequence.

    Built by copying the bound method onto a stand-in rather than loading MLX, which
    keeps the logic under test identical to production while needing no weights.
    """

    def __init__(self, responses: list[FakeResponse]) -> None:
        from arc.model.mlx_backend import MLXModel

        self._responses = responses
        self._name = "stub"
        self._capabilities = ModelCapabilities(max_context=1024)
        # _stream_raw passes these straight to stream_generate, which the stub ignores.
        self._model = object()
        self._tokenizer = object()
        self._stream_raw = MLXModel._stream_raw.__get__(self)  # type: ignore[attr-defined]
        self._first_stop = MLXModel._first_stop

    def _render(self, messages: list[Message], tools: Any) -> str:
        return "prompt"

    def _sampler(self, temperature: float) -> Any:
        return None

    def stream(self, stop: list[str] | None) -> Iterator[Token]:
        import sys
        import types

        # Inject a module exposing stream_generate, since _stream_raw imports it
        # locally at call time.
        module = types.ModuleType("mlx_lm")
        module.stream_generate = lambda *a, **k: iter(self._responses)  # type: ignore[attr-defined]
        saved = sys.modules.get("mlx_lm")
        sys.modules["mlx_lm"] = module
        try:
            yield from self._stream_raw(
                [Message(role="user", content="x")],
                tools=None,
                max_tokens=100,
                temperature=0.0,
                stop=stop,
            )
        finally:
            if saved is not None:
                sys.modules["mlx_lm"] = saved
            else:
                del sys.modules["mlx_lm"]


def collect(responses: list[FakeResponse], stop: list[str] | None) -> tuple[str, str | None]:
    """Run the stream and return (full text, final finish_reason)."""
    tokens = list(StubbedMLXModel(responses).stream(stop))
    return "".join(t.text for t in tokens), tokens[-1].finish_reason if tokens else None


def test_stop_sequence_straddling_tokens_does_not_leak() -> None:
    """Regression: "STOPHERE" arriving as "STOP" + "HERE" leaked "STOP" downstream,
    because each token was emitted before the match became detectable."""
    text, reason = collect(
        [FakeResponse("alpha "), FakeResponse("STOP"), FakeResponse("HERE"), FakeResponse(" more")],
        ["STOPHERE"],
    )
    assert text == "alpha "
    assert "STOP" not in text
    assert reason == "stop"


def test_stop_sequence_within_one_token() -> None:
    text, _ = collect([FakeResponse("keep "), FakeResponse("XXstop"), FakeResponse("drop")], ["XX"])
    assert text == "keep "


def test_text_before_stop_is_fully_emitted() -> None:
    text, _ = collect(
        [FakeResponse("one "), FakeResponse("two "), FakeResponse("three END")], ["END"]
    )
    assert text == "one two three "


def test_no_stop_hit_flushes_everything() -> None:
    """The withheld tail must be released when generation ends without a match."""
    text, _ = collect([FakeResponse("all "), FakeResponse("of "), FakeResponse("it")], ["NEVER"])
    assert text == "all of it"


def test_length_finish_reason_survives_the_stop_path() -> None:
    """Regression: the withheld-tail flush hardcoded "stop", so a run truncated by
    max_tokens reported a clean finish and callers could not tell it was cut off."""
    text, reason = collect(
        [FakeResponse("partial "), FakeResponse("answer", finish_reason="length")], ["NEVER"]
    )
    assert text == "partial answer"
    assert reason == "length"


def test_length_finish_reason_without_stop_sequences() -> None:
    _, reason = collect([FakeResponse("a"), FakeResponse("b", finish_reason="length")], None)
    assert reason == "length"


def test_stop_reason_reported_when_model_stops_naturally() -> None:
    _, reason = collect([FakeResponse("done", finish_reason="stop")], ["NEVER"])
    assert reason == "stop"


def test_multiple_stop_sequences_take_the_earliest() -> None:
    text, _ = collect([FakeResponse("a BBB b AAA c")], ["AAA", "BBB"])
    assert text == "a "


def test_degenerate_empty_stop_string_withholds_nothing() -> None:
    """stop=[""] must not compute a negative hold or swallow output."""
    text, _ = collect([FakeResponse("hello "), FakeResponse("world")], [""])
    assert text == "hello world"


def test_streaming_without_stop_emits_each_token_immediately() -> None:
    tokens = list(StubbedMLXModel([FakeResponse("a"), FakeResponse("b")]).stream(None))
    assert [t.text for t in tokens] == ["a", "b"]


@pytest.mark.parametrize("stop", [["Z"], ["ZZ"], ["ZZZZZZZZ"]])
def test_hold_size_does_not_affect_correctness(stop: list[str]) -> None:
    """Whatever the withheld-tail length, non-matching output arrives intact."""
    text, _ = collect([FakeResponse("abc "), FakeResponse("def "), FakeResponse("ghi")], stop)
    assert text == "abc def ghi"
