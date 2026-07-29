"""Hybrid retrieval.

§4.2 specifies four strategies run in parallel, then merged, deduped, and reranked:

1. **Dense vector** similarity — catches paraphrase. "What theme does he use?" finds
   "Dhruv prefers dark mode" with no shared words.
2. **BM25 keyword** (FTS5) — catches exact terms vectors are bad at: identifiers,
   error codes, filenames, rare proper nouns.
3. **Entity graph** traversal — reaches memories that share no text with the query,
   only a connection through the graph.
4. **Recency** — because "what were we just doing" is a real question that similarity
   answers badly.

Each is weak alone and they fail in different directions, which is exactly why the
combination beats any one of them.

**Scores are not comparable across strategies.** Cosine distance, BM25 rank, and graph
depth live on unrelated scales, so this fuses by *rank* (Reciprocal Rank Fusion) rather
than by normalising scores. RRF only needs an ordering from each source, which makes it
robust to a strategy returning wildly-scaled numbers — a normalisation approach breaks
silently the moment one source's distribution shifts.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from arc.log import get_logger
from arc.memory.episodic import SupportsEmbedding
from arc.memory.semantic import SemanticMemory
from arc.memory.store import MemoryRecord, MemoryStore, serialize_vector

_log = get_logger(__name__)

#: RRF's damping constant. 60 is the value from the original paper and is not
#: sensitive: it controls how sharply rank 1 beats rank 2, and anything in 10-100
#: behaves similarly.
RRF_K = 60

#: Default weights per strategy. Vector leads because paraphrase is the common case;
#: recency is lowest because it is a tiebreaker, not evidence of relevance.
DEFAULT_WEIGHTS = {"vector": 1.0, "keyword": 0.8, "graph": 0.7, "recency": 0.4}


@dataclass(frozen=True, slots=True)
class Hit:
    """One retrieved memory with its provenance."""

    record: MemoryRecord
    score: float
    #: Which strategies found it, and at what rank. Retained because "why did this come
    #: back?" is the first question when retrieval misbehaves.
    sources: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {**self.record.to_dict(), "score": round(self.score, 4), "sources": self.sources}


class Retriever:
    """Hybrid search across every memory layer."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: SupportsEmbedding,
        semantic: SemanticMemory | None = None,
        *,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._semantic = semantic or SemanticMemory(store, embedder)
        self._weights = {**DEFAULT_WEIGHTS, **(weights or {})}

    # ── Individual strategies ───────────────────────────────────────────────────

    def by_vector(self, query: str, *, limit: int = 20) -> list[int]:
        """Nearest neighbours by embedding distance, closest first."""
        try:
            vector = self._embedder.embed(query, is_query=True)
        except Exception as exc:
            _log.warning("vector search unavailable: %s", exc)
            return []

        rows = (
            self._store.connection()
            .execute(
                """
            SELECT v.memory_id
              FROM memory_vectors v
              JOIN memories m ON m.id = v.memory_id
             WHERE v.embedding MATCH ? AND k = ?
               AND m.superseded_by IS NULL
             ORDER BY v.distance
            """,
                (serialize_vector(vector), limit),
            )
            .fetchall()
        )
        return [int(r["memory_id"]) for r in rows]

    def by_keyword(self, query: str, *, limit: int = 20) -> list[int]:
        """BM25 keyword matches, best first."""
        sanitized = _sanitize_fts_query(query)
        if not sanitized:
            return []

        try:
            rows = (
                self._store.connection()
                .execute(
                    """
                SELECT f.rowid AS id
                  FROM memories_fts f
                  JOIN memories m ON m.id = f.rowid
                 WHERE memories_fts MATCH ?
                   AND m.superseded_by IS NULL
                 ORDER BY bm25(memories_fts)
                 LIMIT ?
                """,
                    (sanitized, limit),
                )
                .fetchall()
            )
        except Exception as exc:
            # A malformed FTS expression must not take down retrieval; the other three
            # strategies can still answer.
            _log.warning("keyword search failed for %r: %s", sanitized, exc)
            return []
        return [int(r["id"]) for r in rows]

    def by_graph(self, query: str, *, limit: int = 20) -> list[int]:
        """Memories reachable through entities mentioned in the query."""
        names = self._semantic.extract_entities(query)
        if not names:
            return []

        # Include one hop out, so a query about a person reaches memories about the
        # projects they own without naming them.
        expanded = list(names)
        for name in names[:3]:
            expanded.extend(e.name for e in self._semantic.neighbours(name, hops=1, limit=5))

        records = self._semantic.memories_for_entities(expanded, limit=limit)
        return [r.id for r in records]

    def by_recency(self, *, limit: int = 20, layer: str | None = None) -> list[int]:
        """The most recent memories, newest first."""
        return [r.id for r in self._store.recent(limit=limit, layer=layer)]

    # ── Fusion ──────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        per_strategy: int = 20,
        layer: str | None = None,
        touch: bool = True,
    ) -> list[Hit]:
        """Run every strategy, fuse the rankings, and return the best memories.

        Strategies run concurrently. They are all I/O- or CPU-light and independent,
        and the embedding call is the slow one — overlapping it with three SQL queries
        is close to free.
        """
        if not query.strip():
            return []

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {
                "vector": pool.submit(self.by_vector, query, limit=per_strategy),
                "keyword": pool.submit(self.by_keyword, query, limit=per_strategy),
                "graph": pool.submit(self.by_graph, query, limit=per_strategy),
                "recency": pool.submit(self.by_recency, limit=per_strategy, layer=layer),
            }
            rankings: dict[str, list[int]] = {}
            for name, future in futures.items():
                try:
                    rankings[name] = future.result()
                except Exception as exc:
                    # One failing strategy degrades quality; it must not fail the query.
                    _log.warning("retrieval strategy %s failed: %s", name, exc)
                    rankings[name] = []

        fused = self._fuse(rankings)
        if not fused:
            return []

        records = self._fetch(list(fused), layer=layer)
        hits = [
            Hit(record=records[mid], score=score, sources=sources)
            for mid, (score, sources) in fused.items()
            if mid in records
        ]

        hits.sort(key=self._final_score, reverse=True)
        top = hits[:limit]

        if touch and top:
            self._store.touch([h.record.id for h in top])

        return top

    @staticmethod
    def _final_score(hit: Hit) -> float:
        """Fused rank, gently modulated by salience and confidence.

        Salience and confidence describe the *memory*, not its relevance to this
        query, so they must only break ties — never overturn the ranking.

        This was originally a plain multiplication, which was badly wrong. Layer
        defaults span 0.6 (a conversation turn) to 1.5 (a stated preference), a 2.5x
        factor, while RRF scores across a result set span roughly 3x. The multiplier
        therefore dominated: a query containing the literal words "router bug"
        returned a preference about tidying Downloads, because that preference
        happened to start with higher salience.

        Damped to +/-15% each, so a strong relevance signal always wins and these act
        as the tiebreakers they were meant to be.
        """
        modifier = 1.0 + 0.15 * (hit.record.salience - 1.0) + 0.15 * (hit.record.confidence - 1.0)
        return hit.score * max(0.5, modifier)

    def _fuse(self, rankings: dict[str, list[int]]) -> dict[int, tuple[float, dict[str, int]]]:
        """Reciprocal Rank Fusion across strategies.

        Each strategy contributes ``weight / (RRF_K + rank)``. A memory found by three
        strategies at middling rank beats one found by a single strategy at rank 1,
        which is the behaviour we want: agreement across independent signals is
        stronger evidence than one confident source.
        """
        scores: dict[int, float] = {}
        sources: dict[int, dict[str, int]] = {}

        for strategy, ids in rankings.items():
            weight = self._weights.get(strategy, 1.0)
            for rank, memory_id in enumerate(ids, start=1):
                scores[memory_id] = scores.get(memory_id, 0.0) + weight / (RRF_K + rank)
                sources.setdefault(memory_id, {})[strategy] = rank

        return {mid: (score, sources[mid]) for mid, score in scores.items()}

    def _fetch(self, ids: list[int], *, layer: str | None) -> dict[int, MemoryRecord]:
        """Load records for the fused ids in one query."""
        if not ids:
            return {}

        placeholders = ",".join("?" * len(ids))
        clause = " AND layer = ?" if layer else ""
        params: list[Any] = [*ids]
        if layer:
            params.append(layer)

        rows = (
            self._store.connection()
            .execute(
                f"SELECT * FROM memories WHERE id IN ({placeholders})"
                f" AND superseded_by IS NULL{clause}",
                params,
            )
            .fetchall()
        )
        return {int(r["id"]): MemoryRecord.from_row(r) for r in rows}


def _sanitize_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    FTS5 treats characters like ``"``, ``*``, ``:``, ``-``, and ``(`` as operators, so
    a user question containing any of them is a syntax error rather than a search.
    Each word is quoted and the whole thing OR-ed, which searches for the terms
    without letting punctuation become syntax.
    """
    words = [w.strip('"').replace('"', "") for w in query.split()]
    tokens = [f'"{w}"' for w in words if w and any(c.isalnum() for c in w)]
    return " OR ".join(tokens)
