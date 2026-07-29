"""Tests for the agent loop.

Driven by a scripted model rather than a real one, so the control flow is tested
deterministically: which paths are taken, how failures propagate, and when the loop
stops. Whether a real model chooses good tools is a different question and not one a
test suite can answer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from arc.agent.executor import Executor
from arc.agent.loop import Agent
from arc.errors import ToolError
from arc.model.base import (
    Completion,
    LanguageModel,
    Message,
    ModelCapabilities,
    Token,
    ToolCall,
    ToolSchema,
    Usage,
)
from arc.tools.registry import ToolRegistry


class ScriptedModel(LanguageModel):
    """Returns a fixed sequence of completions, one per call."""

    def __init__(
        self,
        replies: list[str | Completion],
        *,
        native_tools: bool = False,
        context_length: int = 4096,
    ) -> None:
        self._replies = list(replies)
        self._native = native_tools
        self._context_length = context_length
        #: Every prompt the loop sent, so tests can assert on what the model saw.
        self.prompts: list[list[Message]] = []

    @property
    def name(self) -> str:
        return "scripted"

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(max_context=self._context_length, native_tool_calling=self._native)

    def count_tokens(self, text: str) -> int:
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
        self.prompts.append(list(messages))
        if not self._replies:
            # Running dry means the loop iterated more than the test scripted for.
            return Completion(text="done", finish_reason="stop", usage=Usage(0, 0))

        reply = self._replies.pop(0)
        if isinstance(reply, Completion):
            return reply
        return Completion(text=reply, finish_reason="stop", usage=Usage(0, 0))

    def stream(
        self,
        messages: list[Message],
        *,
        tools: list[ToolSchema] | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stop: list[str] | None = None,
    ) -> Iterator[Token]:
        yield Token(text=self.generate(messages).text, finish_reason="stop")


@pytest.fixture
def registry() -> ToolRegistry:
    calls: list[str] = []
    tools = ToolRegistry()

    @tools.register
    def echo(text: str) -> str:
        """Return the text given."""
        calls.append(text)
        return f"echoed: {text}"

    @tools.register
    def fail() -> str:
        """Always fails."""
        raise ToolError("this tool is broken")

    @tools.register(mutating=True)
    def change(target: str) -> str:
        """Change something."""
        return f"changed {target}"

    return tools


def block(name: str, **arguments: object) -> str:
    """Build a fenced tool-call block the way a model would."""
    import json

    return f"```json\n{json.dumps({'name': name, 'arguments': arguments})}\n```"


# ── Basic flow ──────────────────────────────────────────────────────────────────


def test_plain_answer_without_tools(registry: ToolRegistry) -> None:
    """No tool call means the model is answering. That is the normal exit."""
    agent = Agent(ScriptedModel(["The answer is 42."]), registry)
    result = agent.run("what is 6 times 7")
    assert result.answer == "The answer is 42."
    assert result.steps == []
    assert not result.exhausted


def test_single_tool_call_then_answer(registry: ToolRegistry) -> None:
    model = ScriptedModel([block("echo", text="hello"), "It said hello."])
    result = Agent(model, registry).run("use the echo tool")
    assert result.tools_used == ["echo"]
    assert result.answer == "It said hello."


def test_multi_step_sequence(registry: ToolRegistry) -> None:
    model = ScriptedModel([block("echo", text="one"), block("echo", text="two"), "Both done."])
    result = Agent(model, registry).run("echo twice")
    assert result.tools_used == ["echo", "echo"]
    assert len(result.steps) == 2


def test_observations_are_fed_back_to_the_model(registry: ToolRegistry) -> None:
    """The loop only works if the model can see what its call produced."""
    model = ScriptedModel([block("echo", text="hello"), "done"])
    Agent(model, registry).run("task")
    second_prompt = model.prompts[1]
    assert any("echoed: hello" in m.content for m in second_prompt)


def test_tool_failure_is_an_observation_not_a_crash(registry: ToolRegistry) -> None:
    """§7: the loop catches tool errors, feeds them back, and adapts."""
    model = ScriptedModel([block("fail"), block("echo", text="plan b"), "recovered"])
    result = Agent(model, registry).run("task")

    assert result.answer == "recovered"
    assert result.steps[0].observation is not None
    assert not result.steps[0].observation.ok
    assert "this tool is broken" in model.prompts[1][-1].content


# ── Step limits ─────────────────────────────────────────────────────────────────


def test_step_limit_stops_a_runaway_loop(registry: ToolRegistry) -> None:
    """A model that has misunderstood will call tools forever."""
    model = ScriptedModel([block("echo", text="again")] * 20 + ["final"])
    result = Agent(model, registry, max_steps=3).run("task")
    assert result.exhausted
    assert len(result.steps) == 3


def test_exhausted_run_still_produces_an_answer(registry: ToolRegistry) -> None:
    """A partial result with its reasoning beats silence."""
    model = ScriptedModel([block("echo", text="x")] * 5 + ["here is what I found"])
    result = Agent(model, registry, max_steps=2).run("task")
    assert result.answer
    assert result.exhausted


# ── Dispatch paths ──────────────────────────────────────────────────────────────


def test_prompted_fallback_renders_tools_into_the_prompt(registry: ToolRegistry) -> None:
    """§4.1: this is the path a from-scratch model will use."""
    model = ScriptedModel(["done"], native_tools=False)
    agent = Agent(model, registry)
    assert not agent.native_tools

    agent.run("task")
    system = model.prompts[0][0].content
    assert "echo: Return the text given." in system
    assert "fenced JSON block" in system


def test_native_path_does_not_render_tools_into_the_prompt(registry: ToolRegistry) -> None:
    model = ScriptedModel(["done"], native_tools=True)
    Agent(model, registry).run("task")
    assert "fenced JSON block" not in model.prompts[0][0].content


def test_native_tool_calls_are_used(registry: ToolRegistry) -> None:
    completion = Completion(
        text="",
        finish_reason="tool_call",
        usage=Usage(0, 0),
        tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "native"})],
    )
    model = ScriptedModel([completion, "done"], native_tools=True)
    result = Agent(model, registry).run("task")
    assert result.tools_used == ["echo"]


def test_fenced_block_is_read_even_from_a_native_model(registry: ToolRegistry) -> None:
    """Native-capable models sometimes emit a fenced block anyway; refusing to read it
    would stall the loop over a formatting choice."""
    model = ScriptedModel([block("echo", text="x"), "done"], native_tools=True)
    assert Agent(model, registry).run("task").tools_used == ["echo"]


def test_malformed_json_is_repaired_and_recorded(registry: ToolRegistry) -> None:
    model = ScriptedModel(['{"name": "echo", "arguments": {"text": "x",}}', "done"])
    result = Agent(model, registry).run("task")
    assert result.tools_used == ["echo"]
    assert result.steps[0].repairs


def test_hallucinated_tool_ends_the_run_cleanly(registry: ToolRegistry) -> None:
    """Dispatching a tool that does not exist would be worse than stopping."""
    model = ScriptedModel([block("teleport", destination="mars")])
    result = Agent(model, registry).run("task")
    assert result.tools_used == []


# ── Dry run ─────────────────────────────────────────────────────────────────────


def test_dry_run_skips_mutating_tools(registry: ToolRegistry) -> None:
    model = ScriptedModel([block("change", target="prod"), "done"])
    result = Agent(model, registry, dry_run=True).run("task")
    observation = result.steps[0].observation
    assert observation is not None
    assert observation.dry_run
    assert "[dry-run]" in observation.output


def test_dry_run_still_runs_read_only_tools(registry: ToolRegistry) -> None:
    model = ScriptedModel([block("echo", text="safe"), "done"])
    result = Agent(model, registry, dry_run=True).run("task")
    observation = result.steps[0].observation
    assert observation is not None
    assert observation.output == "echoed: safe"


# ── Instrumentation ─────────────────────────────────────────────────────────────


def test_on_step_callback_fires(registry: ToolRegistry) -> None:
    """A CLI needs progress, or a thirty-second task looks like a hang."""
    seen: list[str] = []
    model = ScriptedModel([block("echo", text="x"), "done"])
    Agent(model, registry, on_step=lambda s: seen.append(s.tool or "")).run("task")
    assert seen == ["echo"]


def test_run_is_audited(registry: ToolRegistry, arc_home_tmp: Path) -> None:
    from arc.audit import AuditLogger

    audit = AuditLogger()
    model = ScriptedModel([block("echo", text="x"), "done"])
    Agent(model, registry, audit=audit).run("task")

    events = [r["event"] for r in audit.read_recent(limit=50)]
    assert "agent.start" in events
    assert "tool.echo" in events
    assert "agent.finish" in events


def test_result_serializes(registry: ToolRegistry) -> None:
    model = ScriptedModel([block("echo", text="x"), "done"])
    payload = Agent(model, registry).run("task").to_dict()
    assert payload["tools_used"] == ["echo"]
    assert payload["steps"][0]["observation"]["ok"] is True


def test_executor_can_be_injected(registry: ToolRegistry) -> None:
    """So a caller can share one audited executor across several agents."""
    executor = Executor(registry, dry_run=True)
    model = ScriptedModel([block("change", target="x"), "done"])
    result = Agent(model, registry, executor=executor).run("task")
    observation = result.steps[0].observation
    assert observation is not None
    assert observation.dry_run
