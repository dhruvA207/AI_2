"""Working memory — the context-window budget manager.

Retrieval returns the most relevant memories. This decides how many of them actually
fit, which is a different question with a hard constraint attached: exceed the context
window and generation fails outright.

The budget is split rather than shared. Recent conversation and retrieved memories
compete for the same tokens, and without explicit allocation a long conversation
starves retrieval entirely — you would silently stop remembering anything older than
the current session, which is the exact failure the memory system exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from arc.log import get_logger
from arc.memory.retrieval import Hit
from arc.memory.store import MemoryRecord

_log = get_logger(__name__)


class SupportsTokenCount(Protocol):
    """Anything that can count tokens — in practice, a ``LanguageModel``."""

    def count_tokens(self, text: str) -> int: ...

    @property
    def context_length(self) -> int: ...


@dataclass(frozen=True, slots=True)
class Budget:
    """How the context window is divided.

    Fractions of the *usable* window, which is the model's context minus room for the
    reply. They need not sum to 1.0 — what is left over is slack, and slack is what
    stops a slightly-wrong token estimate from becoming a failed generation.
    """

    total: int
    reserved_for_reply: int
    system: float = 0.10
    procedural: float = 0.10
    retrieved: float = 0.35
    history: float = 0.35

    @property
    def usable(self) -> int:
        """Tokens available for the prompt."""
        return max(0, self.total - self.reserved_for_reply)

    def allocation(self, name: str) -> int:
        """Tokens allotted to one section."""
        fraction = float(getattr(self, name))
        return int(self.usable * fraction)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "total": self.total,
            "usable": self.usable,
            "reserved_for_reply": self.reserved_for_reply,
            "sections": {
                name: self.allocation(name)
                for name in ("system", "procedural", "retrieved", "history")
            },
        }


@dataclass
class WorkingMemory:
    """Assembles a prompt that fits."""

    counter: SupportsTokenCount
    budget: Budget
    #: Populated as sections are packed, so callers can report what was dropped.
    dropped: dict[str, int] = field(default_factory=dict)

    @classmethod
    def for_model(
        cls, model: SupportsTokenCount, *, reserve: int = 2048, **fractions: float
    ) -> WorkingMemory:
        """Build a budget sized to a model's real context window."""
        return cls(
            counter=model,
            budget=Budget(total=model.context_length, reserved_for_reply=reserve, **fractions),
        )

    def pack_memories(self, hits: list[Hit], *, section: str = "retrieved") -> list[Hit]:
        """Take hits in order until the section's budget is spent.

        Greedy by rank rather than trying to maximise how many fit. A packing that
        swaps a highly-relevant long memory for three marginal short ones optimises
        the wrong thing — relevance order is the whole point of having ranked them.
        """
        allowance = self.budget.allocation(section)
        kept: list[Hit] = []
        used = 0

        for hit in hits:
            cost = self.counter.count_tokens(hit.record.content) + 8  # framing overhead
            if used + cost > allowance:
                continue  # Skip, don't stop: a shorter later hit may still fit.
            kept.append(hit)
            used += cost

        self.dropped[section] = len(hits) - len(kept)
        return kept

    def pack_history(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        """Fit conversation turns, keeping the most recent.

        Walks backwards from the newest, because dropping the start of a conversation
        is far less damaging than dropping what was just said.
        """
        allowance = self.budget.allocation("history")
        kept: list[MemoryRecord] = []
        used = 0

        for record in reversed(records):
            cost = self.counter.count_tokens(record.content) + 8
            if used + cost > allowance:
                break
            kept.append(record)
            used += cost

        kept.reverse()
        self.dropped["history"] = len(records) - len(kept)
        return kept

    def render_memories(self, hits: list[Hit]) -> str:
        """Format retrieved memories for the prompt.

        Provenance used to be appended inline as ``- fact [episodic, 2026-07-30]``.
        Small models copy that: asked to reply "pong", the model answered
        "pong [episodic, 2026-07-30]", and because the reply was then stored and
        recalled, the next turn produced two markers. Telling it not to did not help —
        a format that looks like content gets imitated regardless.

        So facts are plain prose now, and provenance appears only where it earns its
        place: a source URL, which §4.4 requires for citation, and a low-confidence
        note, which changes how the fact should be used. Both are phrased as sentences
        rather than bracketed tags.
        """
        if not hits:
            return ""

        lines = [
            "Things you already know (use them naturally; this list is context, "
            "not a format to copy):",
            "",
        ]
        for hit in hits:
            record = hit.record
            line = f"- {record.content.rstrip('.')}."
            if record.source_url:
                line += f" Source: {record.source_url}"
            if record.confidence < 0.7:
                line += " (uncertain)"
            lines.append(line)
        return "\n".join(lines)

    def render_procedures(self, records: list[MemoryRecord]) -> str:
        """Format learned preferences and workflows."""
        if not records:
            return ""
        lines = ["## What I know about how you work", ""]
        lines.extend(f"- {r.content}" for r in records)
        return "\n".join(lines)

    def report(self) -> dict[str, Any]:
        """Summarise the packing, for `/tokens` and debugging."""
        return {"budget": self.budget.to_dict(), "dropped": dict(self.dropped)}
