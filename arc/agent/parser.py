"""Tolerant parsing of tool calls from model output.

§4.1 singles this out: *"Build that parser properly, with repair heuristics for
truncated or malformed JSON. This is specifically what will let my own model drive the
agent later, so don't treat it as an afterthought."*

The problem is that small models — and anything trained from scratch — emit *nearly*
valid JSON. They add trailing commas, use single quotes, wrap the block in prose,
forget a closing brace, or run out of tokens mid-object. A strict ``json.loads`` throws
all of that away, and the agent stalls on output that was 95% correct.

So parsing runs as a ladder, cheapest and most reliable first:

1. Strict JSON on a fenced block.
2. Strict JSON on any balanced ``{...}`` span in the text.
3. Repaired JSON — quotes normalised, trailing commas removed, unclosed structures
   completed.
4. Key-value scraping, when the structure is beyond repair but the intent is legible.

Every repair is recorded on the result, because a parser that silently guesses is
worse than one that fails: you need to know the model is producing garbage before you
conclude the *agent* is broken.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from arc.log import get_logger

_log = get_logger(__name__)

#: Fenced blocks, with or without a language tag.
_FENCE = re.compile(r"```(?:json|tool_call|tool|python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)

#: A fence the model started but never closed — the classic truncation signature.
_UNCLOSED_FENCE = re.compile(r"```(?:json|tool_call|tool)?\s*\n?(.*)$", re.DOTALL | re.IGNORECASE)

#: Bare `name: value` pairs, the last-resort scrape.
_KEY_VALUE = re.compile(r'["\']?(\w+)["\']?\s*[:=]\s*("(?:[^"\\]|\\.)*"|\'[^\']*\'|[^,\n}]+)')

#: Trailing comma before a closing brace or bracket.
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")

#: Unquoted object keys: `{name: "x"}` rather than `{"name": "x"}`.
_UNQUOTED_KEY = re.compile(r"([{,]\s*)([A-Za-z_]\w*)(\s*:)")


@dataclass(frozen=True, slots=True)
class ParsedCall:
    """One tool call recovered from model output."""

    name: str
    arguments: dict[str, Any]
    #: Which repairs were applied. Empty means the model emitted valid JSON.
    repairs: list[str] = field(default_factory=list)
    #: Text outside the tool call — usually the model's reasoning.
    preamble: str = ""

    @property
    def was_repaired(self) -> bool:
        """Whether anything had to be fixed to read this."""
        return bool(self.repairs)


def _balanced_spans(text: str) -> list[str]:
    """Return every balanced ``{...}`` span, outermost first.

    A brace counter rather than a regex, because JSON nests and regexes cannot. Braces
    inside string literals are skipped — otherwise a path like ``"{tmp}/x"`` throws the
    count off and every span after it is wrong.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False

    for index, character in enumerate(text):
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue

        if character == "{":
            if depth == 0:
                start = index
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                spans.append(text[start : index + 1])
                start = -1
            elif depth < 0:
                # A stray closing brace; resynchronise rather than going negative.
                depth = 0

    return spans


def repair_json(text: str) -> tuple[str, list[str]]:
    """Apply repair heuristics, returning the fixed text and what was changed."""
    repairs: list[str] = []
    fixed = text.strip()

    if _UNQUOTED_KEY.search(fixed):
        fixed = _UNQUOTED_KEY.sub(r'\1"\2"\3', fixed)
        repairs.append("quoted unquoted keys")

    # Single-quoted strings are Python, not JSON. Only converted when there are no
    # double quotes to confuse: mixed quoting is ambiguous and guessing makes it worse.
    if "'" in fixed and '"' not in fixed:
        fixed = fixed.replace("'", '"')
        repairs.append("converted single quotes")

    if _TRAILING_COMMA.search(fixed):
        fixed = _TRAILING_COMMA.sub(r"\1", fixed)
        repairs.append("removed trailing comma")

    for literal, replacement in (("True", "true"), ("False", "false"), ("None", "null")):
        if re.search(rf"\b{literal}\b", fixed):
            fixed = re.sub(rf"\b{literal}\b", replacement, fixed)
            repairs.append(f"lowercased {literal}")

    # Truncation: close whatever is still open. Counting outside string literals so a
    # brace in a value does not inflate the total.
    opens = closes = 0
    bracket_opens = bracket_closes = 0
    in_string = escaped = False
    for character in fixed:
        if escaped:
            escaped = False
            continue
        if character == "\\":
            escaped = True
            continue
        if character == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        opens += character == "{"
        closes += character == "}"
        bracket_opens += character == "["
        bracket_closes += character == "]"

    if in_string:
        fixed += '"'
        repairs.append("closed unterminated string")
    if bracket_opens > bracket_closes:
        fixed += "]" * (bracket_opens - bracket_closes)
        repairs.append("closed unterminated array")
    if opens > closes:
        # A dangling `"key":` with no value would still be invalid; drop it first.
        fixed = re.sub(r',?\s*"[^"]*"\s*:\s*$', "", fixed)
        fixed += "}" * (opens - closes)
        repairs.append("closed unterminated object")

    return fixed, repairs


def _coerce(raw: str) -> Any:
    """Turn a scraped value into a real type."""
    value = raw.strip().strip(",").strip()
    if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
        return value[1:-1]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _normalize(payload: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    """Pull a (name, arguments) pair out of a parsed object.

    Models disagree on the wrapper: ``{"tool": ..., "args": ...}``,
    ``{"name": ..., "arguments": ...}``, and the OpenAI-style
    ``{"function": {"name": ...}}`` all occur, sometimes from the same model on
    different turns. Accepting all of them costs a few lines and removes an entire
    class of "the agent randomly does nothing".
    """
    if "function" in payload and isinstance(payload["function"], dict):
        payload = {**payload["function"], **{k: v for k, v in payload.items() if k != "function"}}

    name = None
    for key in ("name", "tool", "tool_name", "action", "function_name"):
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()
            break
    if name is None:
        return None

    arguments: dict[str, Any] = {}
    for key in ("arguments", "args", "parameters", "params", "input", "action_input"):
        candidate = payload.get(key)
        if isinstance(candidate, dict):
            arguments = candidate
            break
        if isinstance(candidate, str):
            # Some models double-encode the arguments as a JSON string.
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                arguments = decoded
                break

    return name, arguments


def parse_tool_call(text: str) -> ParsedCall | None:
    """Recover a tool call from model output, repairing what can be repaired.

    Returns None when there is no tool call to find — which is the normal case for a
    plain conversational reply, not an error.
    """
    if not text or not text.strip():
        return None

    candidates: list[tuple[str, list[str]]] = []

    fenced = _FENCE.findall(text)
    candidates.extend((block, []) for block in fenced)

    if not fenced:
        unclosed = _UNCLOSED_FENCE.search(text)
        if unclosed:
            candidates.append((unclosed.group(1), ["recovered unclosed code fence"]))

    candidates.extend((span, []) for span in _balanced_spans(text))
    # The whole text last: a bare object with no fence and no balance.
    candidates.append((text, []))

    for block, prior in candidates:
        stripped = block.strip()
        if not stripped:
            continue

        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(payload, dict):
                normalized = _normalize(payload)
                if normalized:
                    return ParsedCall(
                        name=normalized[0],
                        arguments=normalized[1],
                        repairs=prior,
                        preamble=_preamble(text, block),
                    )
            continue

        fixed, repairs = repair_json(stripped)
        if not repairs and fixed == stripped:
            continue
        try:
            payload = json.loads(fixed)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        normalized = _normalize(payload)
        if normalized:
            return ParsedCall(
                name=normalized[0],
                arguments=normalized[1],
                repairs=[*prior, *repairs],
                preamble=_preamble(text, block),
            )

    return _scrape(text)


def _scrape(text: str) -> ParsedCall | None:
    """Last resort: pull key-value pairs out of unstructured text.

    Only fires when a tool name is identifiable. Without that anchor this would happily
    "recover" a tool call from ordinary prose, and inventing calls the model never made
    is far worse than failing to parse one it did.
    """
    pairs = {key: _coerce(value) for key, value in _KEY_VALUE.findall(text)}
    if not pairs:
        return None

    name = None
    for key in ("name", "tool", "tool_name", "action", "function_name"):
        candidate = pairs.pop(key, None)
        if isinstance(candidate, str) and candidate.strip():
            name = candidate.strip()
            break
    if name is None:
        return None

    for noise in ("arguments", "args", "parameters", "params"):
        pairs.pop(noise, None)

    _log.warning("scraped a tool call from unstructured output", extra={"tool": name})
    return ParsedCall(
        name=name,
        arguments=pairs,
        repairs=["scraped key-value pairs from unstructured text"],
        preamble="",
    )


def _preamble(full: str, block: str) -> str:
    """Return whatever the model said before the tool call."""
    index = full.find(block)
    if index <= 0:
        return ""
    return full[:index].replace("```json", "").replace("```", "").strip()


def strip_tool_call(text: str) -> str:
    """Remove the tool-call block, leaving the model's prose.

    Used when a reply mixes reasoning with a call and only the reasoning should be
    shown to the user.
    """
    without_fences = _FENCE.sub("", text)
    for span in _balanced_spans(without_fences):
        without_fences = without_fences.replace(span, "")
    return without_fences.strip()
