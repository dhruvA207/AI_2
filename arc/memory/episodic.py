"""Episodic memory — what happened, and when.

Conversation turns and events, timestamped. Answers "what did we do last Tuesday?"

This is the layer that accumulates fastest and matters least per-item: one turn is
rarely worth keeping forever, but the *pattern* across a hundred turns is what
consolidation promotes into semantic and procedural memory. So episodic entries are
written cheaply and decayed aggressively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from arc.memory.store import MemoryRecord, MemoryStore


class SupportsEmbedding(Protocol):
    """Anything that turns text into vectors.

    A Protocol rather than importing ``Embedder`` so tests can pass the hash-based
    stand-in without a subclassing relationship, and so a future MLX embedder drops in
    without touching this module.
    """

    @property
    def dimension(self) -> int: ...

    @property
    def name(self) -> str: ...

    def embed(self, text: str, *, is_query: bool = False) -> list[float]: ...

    def embed_batch(self, texts: list[str], *, is_query: bool = False) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class Turn:
    """One conversational exchange, as stored."""

    role: str
    content: str
    session_id: str


class EpisodicMemory:
    """Timestamped events and conversation turns."""

    def __init__(self, store: MemoryStore, embedder: SupportsEmbedding) -> None:
        self._store = store
        self._embedder = embedder

    def record_turn(
        self,
        role: str,
        content: str,
        *,
        session_id: str,
        occurred_at: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store one conversation turn.

        Both sides of the exchange are stored, not just the user's. What ARC said is
        as much a part of "what happened" as what was asked, and Phase 4's tool traces
        will need the assistant side to reconstruct what it decided.
        """
        return self._store.add(
            layer="episodic",
            kind=f"turn:{role}",
            content=content,
            embedding=self._embedder.embed(content),
            occurred_at=occurred_at,
            source="chat",
            session_id=session_id,
            metadata={**(metadata or {}), "role": role},
            # Turns start below full salience: most are unremarkable, and starting at
            # 1.0 would make every passing remark compete with a deliberate fact.
            salience=0.6,
        )

    def record_event(
        self,
        content: str,
        *,
        kind: str = "event",
        session_id: str | None = None,
        occurred_at: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Store a non-conversational event — a tool call, a file change, a run."""
        return self._store.add(
            layer="episodic",
            kind=kind,
            content=content,
            embedding=self._embedder.embed(content),
            occurred_at=occurred_at,
            source=source or "system",
            session_id=session_id,
            metadata=metadata,
            salience=0.7,
        )

    def session_history(self, session_id: str, *, limit: int = 50) -> list[MemoryRecord]:
        """Return a session's turns in chronological order.

        Reversed after fetching because the index is on ``occurred_at DESC`` — taking
        the newest N and then flipping gives the *last* N turns in reading order,
        whereas ordering ascending in SQL would give the first N.
        """
        rows = self._store.recent(limit=limit, layer="episodic", session_id=session_id)
        return list(reversed(rows))

    def recent(self, *, limit: int = 20) -> list[MemoryRecord]:
        """Return the most recent episodic memories across all sessions."""
        return self._store.recent(limit=limit, layer="episodic")
