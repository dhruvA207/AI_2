"""Procedural memory — learned workflows and preferences.

"When Dhruv says 'clean up downloads' he means this." The layer that makes ARC feel
like it knows you rather than like a fresh chat window.

Two ways things get here. The user states a preference directly, or consolidation
notices the same pattern in episodic memory enough times to promote it (§4.2). The
second path is the interesting one and the reason ``trigger`` is stored separately
from ``content``: matching an incoming request against a remembered trigger is a
different operation from embedding the workflow itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from arc.log import get_logger
from arc.memory.episodic import SupportsEmbedding
from arc.memory.store import MemoryRecord, MemoryStore

_log = get_logger(__name__)

#: How many times a pattern must recur before consolidation promotes it. Low enough to
#: learn quickly, high enough that one coincidence does not become a rule.
DEFAULT_PROMOTION_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class Procedure:
    """A learned workflow or preference."""

    id: int
    trigger: str
    content: str
    kind: str
    confidence: float
    observed_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "id": self.id,
            "trigger": self.trigger,
            "content": self.content,
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "observed_count": self.observed_count,
        }

    @classmethod
    def from_record(cls, record: MemoryRecord) -> Procedure:
        """Build from a stored memory."""
        metadata = record.metadata or {}
        return cls(
            id=record.id,
            trigger=str(metadata.get("trigger", "")),
            content=record.content,
            kind=record.kind,
            confidence=record.confidence,
            observed_count=int(metadata.get("observed_count", 1)),
        )


class ProceduralMemory:
    """Preferences and workflows ARC has learned."""

    def __init__(self, store: MemoryStore, embedder: SupportsEmbedding) -> None:
        self._store = store
        self._embedder = embedder

    def add_preference(
        self,
        content: str,
        *,
        trigger: str = "",
        confidence: float = 1.0,
        source: str = "user",
        session_id: str | None = None,
    ) -> int:
        """Store a stated preference.

        Confidence defaults to 1.0 because the user said it. Promoted patterns arrive
        lower, so a directly stated preference outranks an inferred one when the two
        disagree — which is the right precedence.
        """
        return self._store.add(
            layer="procedural",
            kind="preference",
            content=content,
            embedding=self._embedder.embed(content),
            confidence=confidence,
            source=source,
            session_id=session_id,
            metadata={"trigger": trigger, "observed_count": 1},
            # Procedural memories start high: they are few, deliberate, and should not
            # have to compete with chatter to be retrieved.
            salience=1.5,
        )

    def add_workflow(
        self,
        trigger: str,
        steps: list[str],
        *,
        confidence: float = 0.8,
        observed_count: int = 1,
        source: str = "consolidation",
    ) -> int:
        """Store a multi-step workflow learned from repeated behaviour."""
        content = f"When asked to {trigger}: " + "; ".join(steps)
        return self._store.add(
            layer="procedural",
            kind="workflow",
            content=content,
            embedding=self._embedder.embed(content),
            confidence=confidence,
            source=source,
            metadata={"trigger": trigger, "steps": steps, "observed_count": observed_count},
            salience=1.5,
        )

    def reinforce(self, memory_id: int, *, confidence_step: float = 0.05) -> None:
        """Record that a procedure was confirmed again.

        Confidence rises asymptotically towards 1.0 rather than linearly: repeated
        observation should increase belief, but an inferred pattern should never quite
        reach the certainty of something the user stated outright.
        """
        record = self._store.get(memory_id)
        if record is None:
            return

        metadata = dict(record.metadata or {})
        metadata["observed_count"] = int(metadata.get("observed_count", 1)) + 1
        new_confidence = min(0.99, record.confidence + confidence_step)

        import json

        with self._store.transaction() as conn:
            conn.execute(
                "UPDATE memories SET confidence = ?, metadata = ? WHERE id = ?",
                (new_confidence, json.dumps(metadata), memory_id),
            )

    def all_procedures(self, *, limit: int = 100) -> list[Procedure]:
        """Return every live procedural memory, most salient first."""
        rows = (
            self._store.connection()
            .execute(
                """
            SELECT * FROM memories
             WHERE layer = 'procedural' AND superseded_by IS NULL
             ORDER BY salience DESC, confidence DESC
             LIMIT ?
            """,
                (limit,),
            )
            .fetchall()
        )
        return [Procedure.from_record(MemoryRecord.from_row(r)) for r in rows]

    def matching(self, text: str, *, limit: int = 5) -> list[Procedure]:
        """Return procedures whose trigger appears in ``text``.

        Substring matching on the trigger, not vector similarity. A trigger is a
        remembered phrase ("clean up downloads"), and for those an exact match is both
        more precise and far cheaper than embedding every incoming message against
        every stored procedure.
        """
        lowered = text.lower()
        matches = [p for p in self.all_procedures() if p.trigger and p.trigger.lower() in lowered]
        return matches[:limit]
