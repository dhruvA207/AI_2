"""Semantic memory — facts, entities, and how they relate.

Facts with typed relationships stored as a graph: entities are nodes, relations are
directed labelled edges. Answers "who is X and how do they relate to Y?", including
multi-hop, which a pure vector store cannot do at all — no embedding of "Dhruv"
retrieves a memory about a project he owns unless that memory happens to say his name.

The graph lives in SQLite tables and multi-hop traversal is a recursive CTE (§4.2).
That is fast enough at personal-memory scale and costs no extra dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from arc.log import get_logger
from arc.memory.episodic import SupportsEmbedding
from arc.memory.store import MemoryRecord, MemoryStore, now

_log = get_logger(__name__)

#: Facts from the open web are not facts from the user. Anything time-sensitive gets
#: re-verified rather than trusted forever (§4.4).
_DEFAULT_WEB_CONFIDENCE = 0.7

#: Crude capitalised-phrase extractor, used only when no explicit entities are given.
#: Deliberately not an NER model: that would be another dependency and another model
#: resident in 11.5 GB of memory. Phase 4 can have the LLM extract entities properly.
_CANDIDATE = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}(?:\s+[A-Z][a-zA-Z0-9_-]+)*")

#: Words that pass the capitalisation test but are never entities.
_STOPWORDS = frozenset(
    {
        "The",
        "This",
        "That",
        "These",
        "Those",
        "There",
        "Then",
        "They",
        "When",
        "What",
        "Where",
        "Which",
        "While",
        "With",
        "Would",
        "Could",
        "Should",
        "And",
        "But",
        "For",
        "From",
        "Have",
        "How",
        "Its",
        "Not",
        "You",
        "Your",
        "I",
        "If",
        "In",
        "It",
        "Is",
        "As",
        "At",
        "Be",
        "By",
        "Do",
        "He",
        "She",
    }
)


@dataclass(frozen=True, slots=True)
class Entity:
    """A node in the knowledge graph."""

    id: int
    name: str
    kind: str
    mention_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "mention_count": self.mention_count,
        }


@dataclass(frozen=True, slots=True)
class Relation:
    """A directed, labelled edge."""

    subject: str
    predicate: str
    object: str
    confidence: float = 1.0

    def __str__(self) -> str:
        return f"{self.subject} --{self.predicate}--> {self.object}"


class SemanticMemory:
    """Facts, entities, and the graph connecting them."""

    def __init__(self, store: MemoryStore, embedder: SupportsEmbedding) -> None:
        self._store = store
        self._embedder = embedder

    # ── Entities ────────────────────────────────────────────────────────────────

    def upsert_entity(self, name: str, kind: str = "concept") -> int:
        """Create an entity or bump its mention count, returning its id.

        Identity is the lowercased name alone, so "Dhruv" and "dhruv" are one node —
        and so are "ARC the project" and "ARC the concept". Keying on (name, kind)
        seemed more precise but split the graph: the same real thing mentioned under
        two kinds became two disconnected nodes, and traversal from one could not see
        edges attached to the other.

        ``kind`` is refined rather than fixed. A node first seen as the generic
        ``concept`` is promoted the moment something more specific is asserted, but a
        specific kind is never demoted back to ``concept``.
        """
        normalized = name.strip().lower()
        if not normalized:
            raise ValueError("entity name cannot be empty")

        timestamp = now()
        with self._store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO entities (name, kind, normalized, first_seen, last_seen)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(normalized) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    mention_count = mention_count + 1,
                    kind = CASE
                        WHEN entities.kind = 'concept' AND excluded.kind != 'concept'
                        THEN excluded.kind
                        ELSE entities.kind
                    END
                """,
                (name.strip(), kind, normalized, timestamp, timestamp),
            )
            row = conn.execute(
                "SELECT id FROM entities WHERE normalized = ?", (normalized,)
            ).fetchone()
        return int(row["id"])

    def find_entity(self, name: str, kind: str | None = None) -> Entity | None:
        """Look up an entity by name, optionally constrained to a kind."""
        normalized = name.strip().lower()
        row = (
            self._store.connection()
            .execute("SELECT * FROM entities WHERE normalized = ?", (normalized,))
            .fetchone()
        )
        # `kind` is a filter on the single node, not part of the lookup key, so asking
        # for the wrong kind reports "not found" rather than creating a rival node.
        if row is not None and kind and row["kind"] != kind:
            return None
        if row is None:
            return None
        return Entity(
            id=row["id"], name=row["name"], kind=row["kind"], mention_count=row["mention_count"]
        )

    def extract_entities(self, text: str) -> list[str]:
        """Pull candidate entity names out of free text.

        A heuristic, and honestly a weak one: capitalised phrases minus a stopword
        list. It exists so the graph is not empty before Phase 4 can have the model do
        this properly. Over-extraction is the safer failure — a spurious node costs a
        row, a missed one costs a retrieval path.
        """
        found: list[str] = []
        for match in _CANDIDATE.finditer(text):
            candidate = match.group().strip()
            if candidate in _STOPWORDS:
                continue
            if candidate.split()[0] in _STOPWORDS and len(candidate.split()) == 1:
                continue
            if candidate not in found:
                found.append(candidate)
        return found

    def link(self, memory_id: int, entity_ids: list[int]) -> None:
        """Associate a memory with entities it mentions."""
        if not entity_ids:
            return
        with self._store.transaction() as conn:
            conn.executemany(
                "INSERT OR IGNORE INTO memory_entities(memory_id, entity_id) VALUES (?, ?)",
                [(memory_id, e) for e in entity_ids],
            )

    # ── Facts ───────────────────────────────────────────────────────────────────

    def add_fact(
        self,
        content: str,
        *,
        entities: list[str] | None = None,
        entity_kind: str = "concept",
        confidence: float = 1.0,
        source: str | None = None,
        source_url: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_extract: bool = True,
    ) -> int:
        """Store a fact and wire it into the entity graph."""
        memory_id = self._store.add(
            layer="semantic",
            kind="fact",
            content=content,
            embedding=self._embedder.embed(content),
            confidence=confidence,
            source=source,
            source_url=source_url,
            session_id=session_id,
            metadata=metadata,
        )

        names = list(entities or [])
        if auto_extract and not names:
            names = self.extract_entities(content)

        if names:
            self.link(memory_id, [self.upsert_entity(n, entity_kind) for n in names])

        return memory_id

    def add_web_fact(
        self, content: str, *, source_url: str, entities: list[str] | None = None
    ) -> int:
        """Store something learned from the web, with provenance and lower confidence.

        §4.4 requires that a web-sourced fact carries its URL and retrieval date and
        can cite them. Confidence starts below 1.0 so a page that turns out to be
        wrong does not outrank something the user said directly.
        """
        return self.add_fact(
            content,
            entities=entities,
            confidence=_DEFAULT_WEB_CONFIDENCE,
            source="web",
            source_url=source_url,
            metadata={"retrieved_at": now()},
        )

    # ── Relations ───────────────────────────────────────────────────────────────

    def relate(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        subject_kind: str = "concept",
        object_kind: str = "concept",
        confidence: float = 1.0,
        memory_id: int | None = None,
    ) -> None:
        """Record a typed relationship between two entities."""
        subject_id = self.upsert_entity(subject, subject_kind)
        object_id = self.upsert_entity(obj, object_kind)

        with self._store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO relations (subject_id, predicate, object_id, confidence,
                                       created_at, memory_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(subject_id, predicate, object_id) DO UPDATE SET
                    confidence = MAX(confidence, excluded.confidence)
                """,
                (subject_id, predicate, object_id, confidence, now(), memory_id),
            )

    def relations_for(self, name: str, *, limit: int = 50) -> list[Relation]:
        """Return every relation touching an entity, in either direction."""
        entity = self.find_entity(name)
        if entity is None:
            return []

        rows = (
            self._store.connection()
            .execute(
                """
            SELECT s.name AS subject, r.predicate, o.name AS object, r.confidence
              FROM relations r
              JOIN entities s ON s.id = r.subject_id
              JOIN entities o ON o.id = r.object_id
             WHERE r.subject_id = ? OR r.object_id = ?
             LIMIT ?
            """,
                (entity.id, entity.id, limit),
            )
            .fetchall()
        )
        return [Relation(r["subject"], r["predicate"], r["object"], r["confidence"]) for r in rows]

    def neighbours(self, name: str, *, hops: int = 2, limit: int = 50) -> list[Entity]:
        """Return entities within ``hops`` edges, nearest first.

        A recursive CTE walking edges in both directions. This is the multi-hop
        capability §4.2 asks for: "who is X and how do they relate to Y" needs to reach
        entities that share no text with the query at all.
        """
        entity = self.find_entity(name)
        if entity is None:
            return []

        rows = (
            self._store.connection()
            .execute(
                """
            WITH RECURSIVE walk(id, depth) AS (
                SELECT ?, 0
                UNION
                SELECT CASE WHEN r.subject_id = w.id THEN r.object_id ELSE r.subject_id END,
                       w.depth + 1
                  FROM relations r
                  JOIN walk w ON r.subject_id = w.id OR r.object_id = w.id
                 WHERE w.depth < ?
            )
            SELECT e.*, MIN(w.depth) AS depth
              FROM walk w
              JOIN entities e ON e.id = w.id
             WHERE w.depth > 0
               AND e.id != ?          -- an undirected walk returns to its origin at
                                      -- depth 2; nothing is its own neighbour
             GROUP BY e.id            -- keep the shortest path to each entity, not one
                                      -- row per route that reaches it
             ORDER BY depth, e.mention_count DESC
             LIMIT ?
            """,
                (entity.id, hops, entity.id, limit),
            )
            .fetchall()
        )

        return [Entity(r["id"], r["name"], r["kind"], r["mention_count"]) for r in rows]

    def memories_for_entities(self, names: list[str], *, limit: int = 20) -> list[MemoryRecord]:
        """Return memories linked to any of the named entities."""
        if not names:
            return []

        normalized = [n.strip().lower() for n in names if n.strip()]
        if not normalized:
            return []

        placeholders = ",".join("?" * len(normalized))
        rows = (
            self._store.connection()
            .execute(
                f"""
            SELECT DISTINCT m.*
              FROM memories m
              JOIN memory_entities me ON me.memory_id = m.id
              JOIN entities e ON e.id = me.entity_id
             WHERE e.normalized IN ({placeholders})
               AND m.superseded_by IS NULL
             ORDER BY m.salience DESC, m.occurred_at DESC
             LIMIT ?
            """,
                (*normalized, limit),
            )
            .fetchall()
        )
        return [MemoryRecord.from_row(r) for r in rows]
