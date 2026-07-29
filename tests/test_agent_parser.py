"""Tests for the tolerant tool-call parser.

§4.1 calls this out specifically as what will let a from-scratch model drive the agent,
so it gets the most adversarial testing in the codebase. Small models emit *nearly*
valid JSON; every case below is something a real model actually does.
"""

from __future__ import annotations

import pytest

from arc.agent.parser import (
    ParsedCall,
    _balanced_spans,
    parse_tool_call,
    repair_json,
    strip_tool_call,
)


def call(text: str) -> ParsedCall:
    """Parse, asserting something was found."""
    result = parse_tool_call(text)
    assert result is not None, f"failed to parse: {text!r}"
    return result


# ── Well-formed input ───────────────────────────────────────────────────────────


def test_clean_fenced_json() -> None:
    parsed = call('```json\n{"name": "read_file", "arguments": {"path": "/tmp/x"}}\n```')
    assert parsed.name == "read_file"
    assert parsed.arguments == {"path": "/tmp/x"}
    assert not parsed.was_repaired


def test_bare_json_without_a_fence() -> None:
    parsed = call('{"name": "read_file", "arguments": {"path": "/tmp/x"}}')
    assert parsed.name == "read_file"


def test_preamble_is_captured() -> None:
    parsed = call('Let me look.\n```json\n{"name": "read_file", "arguments": {}}\n```')
    assert "Let me look" in parsed.preamble


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "t", "arguments": {"a": 1}}',
        '{"tool": "t", "args": {"a": 1}}',
        '{"tool_name": "t", "parameters": {"a": 1}}',
        '{"action": "t", "action_input": {"a": 1}}',
        '{"function": {"name": "t", "arguments": "{\\"a\\": 1}"}}',
    ],
)
def test_wrapper_shapes_all_accepted(text: str) -> None:
    """Models disagree on the wrapper, sometimes between turns of one conversation."""
    parsed = call(text)
    assert parsed.name == "t"
    assert parsed.arguments == {"a": 1}


# ── Malformed input the parser must repair ──────────────────────────────────────


def test_trailing_comma() -> None:
    parsed = call('{"name": "t", "arguments": {"a": 1,}}')
    assert parsed.arguments == {"a": 1}
    assert "removed trailing comma" in parsed.repairs


def test_single_quotes() -> None:
    parsed = call("{'name': 't', 'args': {'a': 'b'}}")
    assert parsed.arguments == {"a": "b"}


def test_unquoted_keys() -> None:
    parsed = call('{name: "t", arguments: {path: "/tmp/x"}}')
    assert parsed.name == "t"


def test_python_literals() -> None:
    parsed = call('{"name": "t", "arguments": {"flag": True, "other": None}}')
    assert parsed.arguments == {"flag": True, "other": None}


def test_truncated_object() -> None:
    """The commonest real failure: the model ran out of tokens mid-call."""
    parsed = call('{"name": "t", "arguments": {"path": "/tmp/x"')
    assert parsed.arguments == {"path": "/tmp/x"}
    assert "closed unterminated object" in parsed.repairs


def test_truncated_string() -> None:
    parsed = call('{"name": "t", "arguments": {"path": "/tmp/unfinis')
    assert parsed.name == "t"
    assert "closed unterminated string" in parsed.repairs


def test_truncated_array() -> None:
    parsed = call('{"name": "t", "arguments": {"items": [1, 2')
    assert parsed.arguments == {"items": [1, 2]}


def test_unclosed_fence() -> None:
    parsed = call('```json\n{"name": "t", "arguments": {"a": 1}}')
    assert parsed.name == "t"


def test_dangling_key_is_dropped_on_truncation() -> None:
    """A `"key":` with no value cannot be closed into valid JSON."""
    parsed = call('{"name": "t", "arguments": {"a": 1}, "extra":')
    assert parsed.name == "t"


# ── Things it must NOT do ───────────────────────────────────────────────────────


def test_plain_prose_is_not_a_tool_call() -> None:
    assert parse_tool_call("I think you should read the file yourself.") is None


def test_prose_with_braces_is_not_a_tool_call() -> None:
    """Inventing a call the model never made is far worse than missing one."""
    assert parse_tool_call("Use {} to make a dict, or {'a': 1} for one with a key.") is None


def test_empty_input() -> None:
    assert parse_tool_call("") is None
    assert parse_tool_call("   \n  ") is None


def test_json_without_a_name_is_rejected() -> None:
    assert parse_tool_call('{"arguments": {"path": "/tmp/x"}}') is None


def test_non_object_json_is_rejected() -> None:
    assert parse_tool_call("[1, 2, 3]") is None
    assert parse_tool_call('"just a string"') is None


# ── Brace balancing ─────────────────────────────────────────────────────────────


def test_braces_inside_strings_do_not_confuse_the_scanner() -> None:
    """A path like "{tmp}/x" would throw off a naive counter and corrupt every span."""
    parsed = call('{"name": "write_file", "arguments": {"content": "a { b } c"}}')
    assert parsed.arguments == {"content": "a { b } c"}


def test_escaped_quotes_inside_strings() -> None:
    parsed = call('{"name": "t", "arguments": {"text": "say \\"hi\\""}}')
    assert parsed.arguments == {"text": 'say "hi"'}


def test_balanced_spans_finds_outermost_objects() -> None:
    spans = _balanced_spans('prefix {"a": {"b": 1}} middle {"c": 2} suffix')
    assert spans == ['{"a": {"b": 1}}', '{"c": 2}']


def test_balanced_spans_ignores_stray_closing_brace() -> None:
    assert _balanced_spans('} {"a": 1}') == ['{"a": 1}']


def test_balanced_spans_on_unbalanced_input() -> None:
    assert _balanced_spans('{"a": 1') == []


# ── Repair mechanics ────────────────────────────────────────────────────────────


def test_repair_reports_what_it_changed() -> None:
    """A parser that silently guesses is worse than one that fails."""
    _, repairs = repair_json('{"a": 1,}')
    assert repairs == ["removed trailing comma"]


def test_repair_leaves_valid_json_alone() -> None:
    _, repairs = repair_json('{"a": 1}')
    assert repairs == []


def test_mixed_quotes_are_not_converted() -> None:
    """Converting quotes in mixed input is ambiguous, and guessing makes it worse."""
    _, repairs = repair_json("""{"a": "it's fine"}""")
    assert "converted single quotes" not in repairs


# ── Stripping ───────────────────────────────────────────────────────────────────


def test_strip_removes_the_fenced_block() -> None:
    text = 'Here goes.\n```json\n{"name": "t"}\n```'
    assert strip_tool_call(text) == "Here goes."


def test_strip_removes_bare_objects() -> None:
    assert strip_tool_call('Thinking. {"name": "t"} done.') == "Thinking.  done."


def test_strip_leaves_prose_untouched() -> None:
    assert strip_tool_call("Just prose here.") == "Just prose here."
