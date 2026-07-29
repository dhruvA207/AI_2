"""The agent loop.

perceive → retrieve → plan → act → observe → store, iterated until the model says it is
done or the step limit stops it.

Two dispatch paths, chosen from the model's declared capabilities (§4.1):

- **Native tool calling.** The model returns structured calls; we run them.
- **Prompted ReAct.** The tools are rendered into the prompt and the model emits a
  fenced JSON block that ``agent/parser.py`` extracts tolerantly.

Neither the tools nor the loop's control flow differ between the two. That is the point:
when Track B produces a model with no native tool calling, it drops into the same socket
and the second path carries it.

**The step limit is not a safety feature.** It is there because a model that has
misunderstood a task will happily call tools forever, and an agent that gives up after
twelve steps and explains itself is more useful than one that runs until killed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from arc.agent.executor import Executor, Observation
from arc.agent.parser import parse_tool_call, strip_tool_call
from arc.audit import AuditLogger
from arc.log import get_logger
from arc.memory.service import MemoryService
from arc.model.base import LanguageModel, Message, ToolCall
from arc.tools.registry import ToolRegistry

_log = get_logger(__name__)

DEFAULT_MAX_STEPS = 12

#: Told to the model in the prompted path. Kept short and explicit — a small model
#: follows a plain convention far more reliably than a clever one.
_REACT_INSTRUCTIONS = """\
You have tools. To use one, reply with ONLY a fenced JSON block:

```json
{"name": "tool_name", "arguments": {"arg": "value"}}
```

Rules:
- One tool call per reply. Say nothing after the block.
- When you have enough information, reply with your final answer as plain text and no
  JSON block.
- If a tool fails, read the error and try a different approach.

Available tools:
"""


@dataclass
class Step:
    """One iteration of the loop."""

    number: int
    thought: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    observation: Observation | None = None
    repairs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "number": self.number,
            "thought": self.thought,
            "tool": self.tool,
            "arguments": self.arguments,
            "observation": self.observation.to_dict() if self.observation else None,
            "repairs": self.repairs,
        }


@dataclass
class AgentResult:
    """The outcome of running a task."""

    answer: str
    steps: list[Step] = field(default_factory=list)
    #: True when the loop stopped because it ran out of steps rather than finishing.
    exhausted: bool = False

    @property
    def tools_used(self) -> list[str]:
        """Which tools were called, in order."""
        return [s.tool for s in self.steps if s.tool]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "answer": self.answer,
            "exhausted": self.exhausted,
            "steps": [s.to_dict() for s in self.steps],
            "tools_used": self.tools_used,
        }


class Agent:
    """Runs multi-step tasks against tools."""

    def __init__(
        self,
        model: LanguageModel,
        registry: ToolRegistry,
        *,
        executor: Executor | None = None,
        memory: MemoryService | None = None,
        audit: AuditLogger | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        dry_run: bool = False,
        on_step: Callable[[Step], None] | None = None,
    ) -> None:
        self._model = model
        self._registry = registry
        self._executor = executor or Executor(registry, audit=audit, dry_run=dry_run)
        self._memory = memory
        self._audit = audit
        self._max_steps = max_steps
        #: Called after each step, so a CLI can show progress rather than appearing
        #: to hang for thirty seconds.
        self._on_step = on_step

    @property
    def native_tools(self) -> bool:
        """Whether this model's own tool calling is being used."""
        return self._model.capabilities.native_tool_calling

    def _system_prompt(self, task: str) -> str:
        """Build the system prompt, including memory and the tool protocol."""
        sections = [
            "You are ARC, a local-first assistant with full access to this machine. "
            "Work through the task step by step using the tools available. "
            "Be concise and factual. If something fails, adapt rather than repeating "
            "the same call."
        ]

        if self._memory is not None:
            hits = self._memory.recall(task)
            if hits:
                from arc.memory.working import WorkingMemory

                working = WorkingMemory.for_model(self._model)
                sections.append(working.render_memories(working.pack_memories(hits)))

        if not self.native_tools:
            sections.append(_REACT_INSTRUCTIONS + self._registry.render_prompt())

        return "\n\n".join(s for s in sections if s)

    def run(self, task: str) -> AgentResult:
        """Work on ``task`` until done or out of steps."""
        messages = [
            Message(role="system", content=self._system_prompt(task)),
            Message(role="user", content=task),
        ]
        steps: list[Step] = []

        if self._audit is not None:
            self._audit.record(
                "agent.start", tool="agent", args={"task": task, "max_steps": self._max_steps}
            )

        for number in range(1, self._max_steps + 1):
            completion = self._model.generate(
                messages,
                tools=self._registry.schemas() if self.native_tools else None,
                max_tokens=1024,
            )

            call, repairs = self._extract(completion.text, completion.tool_calls)

            if call is None:
                # No tool call: the model is answering. That is the normal exit.
                answer = completion.text.strip()
                self._finish(task, answer, steps)
                return AgentResult(answer=answer, steps=steps)

            thought = strip_tool_call(completion.text) if not self.native_tools else completion.text
            step = Step(
                number=number,
                thought=thought.strip(),
                tool=call.name,
                arguments=call.arguments,
                repairs=repairs,
            )

            step.observation = self._executor.execute(call.name, call.arguments)
            steps.append(step)
            if self._on_step is not None:
                self._on_step(step)

            # The transcript carries the model's own call and the result, so it can see
            # what it did rather than inferring it from the observation alone.
            messages.append(Message(role="assistant", content=completion.text))
            messages.append(
                Message(role="user", content=f"Observation:\n{step.observation.render()}")
            )

        # Out of steps. Ask for a final answer from what was gathered rather than
        # returning nothing — a partial result with its reasoning beats silence.
        messages.append(
            Message(
                role="user",
                content=(
                    f"You have used all {self._max_steps} steps. Give your best answer now "
                    "from what you have learned, and say what remains unresolved."
                ),
            )
        )
        final = self._model.generate(messages, max_tokens=1024).text.strip()
        self._finish(task, final, steps, exhausted=True)
        return AgentResult(answer=final, steps=steps, exhausted=True)

    def _extract(
        self, text: str, native_calls: list[ToolCall]
    ) -> tuple[ToolCall | None, list[str]]:
        """Get the next tool call from a completion, by whichever path applies."""
        if self.native_tools and native_calls:
            return native_calls[0], []

        # Attempted even for native-capable models: they sometimes emit a fenced block
        # instead of a structured call, and refusing to read it would stall the loop
        # over a formatting choice.
        parsed = parse_tool_call(text)
        if parsed is None:
            return None, []
        if parsed.name not in self._registry:
            # Hallucinated tool name. Returning None ends the run cleanly rather than
            # dispatching something that does not exist.
            _log.info("model referenced unknown tool %r; treating as final answer", parsed.name)
            return None, []

        return (
            ToolCall(id=f"call_{parsed.name}", name=parsed.name, arguments=parsed.arguments),
            parsed.repairs,
        )

    def _finish(
        self, task: str, answer: str, steps: list[Step], *, exhausted: bool = False
    ) -> None:
        """Record the outcome in memory and the audit log."""
        if self._audit is not None:
            self._audit.record(
                "agent.finish",
                status="ok",
                tool="agent",
                args={"task": task},
                result={
                    "steps": len(steps),
                    "tools": [s.tool for s in steps if s.tool],
                    "exhausted": exhausted,
                },
            )

        if self._memory is not None and steps:
            # Stored as an event rather than a turn: this was a task with a trace, not
            # a conversational exchange, and Track B will want these traces as SFT data
            # (§6.1 stage 5).
            tools = ", ".join(s.tool for s in steps if s.tool)
            self._memory.episodic.record_event(
                f"Task: {task}\nTools used: {tools}\nResult: {answer[:500]}",
                kind="task",
                source="agent",
                metadata={"steps": len(steps), "exhausted": exhausted},
            )
