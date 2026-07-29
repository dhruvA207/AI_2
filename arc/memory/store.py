"""The memory database.

One SQLite file at ``~/.arc/memory.db`` holding all three layers, the entity graph,
the FTS5 keyword index, and the vector index. §4.2 is explicit that this stays a single
portable file with no server — the whole memory becomes one backup-able artifact that
survives the macOS-to-Windows move by being copied.

Connections are per-thread. SQLite objects cannot cross threads safely, and Phase 4
will dispatch tools concurrently, so the store hands out a thread-local connection
rather than sharing one and hoping.
"""

from __future__ import annotations

import json
import sqlite3
import struct
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from arc.errors import MemoryError as ArcMemoryError
from arc.log import get_logger
from arc.paths import arc_home

_log = get_logger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"

#: Bumped when the schema changes in a way that needs migration.
SCHEMA_VERSION = 1


def now() -> str:
    """Return an ISO-8601 UTC timestamp.

    One helper so every table stores the same format and lexical ordering matches
    chronological ordering — which is what lets `ORDER BY occurred_at` work without
    parsing.
    """
    return datetime.now(UTC).isoformat()


def serialize_vector(values: list[float]) -> bytes:
    """Pack a float vector into sqlite-vec's compact binary form.

    Little-endian float32. Passing JSON works too but costs roughly 8x the bytes and
    a parse on every comparison.
    """
    return struct.pack(f"<{len(values)}f", *values)


def deserialize_vector(blob: bytes) -> list[float]:
    """Unpack a stored vector."""
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """One row of the memories table."""

    id: int
    layer: str
    kind: str
    content: str
    occurred_at: str
    created_at: str
    salience: float
    confidence: float
    access_count: int = 0
    accessed_at: str | None = None
    source: str | None = None
    source_url: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] | None = None
    superseded_by: int | None = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> MemoryRecord:
        """Build from a query result, parsing the JSON metadata column."""
        keys = row.keys()
        raw = row["metadata"] if "metadata" in keys else "{}"
        try:
            metadata = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            metadata = {}
        return cls(
            id=row["id"],
            layer=row["layer"],
            kind=row["kind"],
            content=row["content"],
            occurred_at=row["occurred_at"],
            created_at=row["created_at"],
            salience=row["salience"],
            confidence=row["confidence"],
            access_count=row["access_count"] if "access_count" in keys else 0,
            accessed_at=row["accessed_at"] if "accessed_at" in keys else None,
            source=row["source"] if "source" in keys else None,
            source_url=row["source_url"] if "source_url" in keys else None,
            session_id=row["session_id"] if "session_id" in keys else None,
            metadata=metadata,
            superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "id": self.id,
            "layer": self.layer,
            "kind": self.kind,
            "content": self.content,
            "occurred_at": self.occurred_at,
            "created_at": self.created_at,
            "salience": round(self.salience, 4),
            "confidence": round(self.confidence, 4),
            "access_count": self.access_count,
            "source": self.source,
            "source_url": self.source_url,
            "session_id": self.session_id,
            "metadata": self.metadata or {},
            "superseded_by": self.superseded_by,
        }


class MemoryStore:
    """SQLite-backed storage for every memory layer."""

    def __init__(self, path: Path | None = None, *, dimension: int = 384) -> None:
        """Open (and create if needed) the memory database.

        ``dimension`` must match the embedder — sqlite-vec bakes it into the virtual
        table's DDL, so a mismatch is caught at insert time rather than silently
        producing nonsense distances.
        """
        self._path = path if path is not None else arc_home() / "memory.db"
        self._dimension = dimension
        self._local = threading.local()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @property
    def path(self) -> Path:
        """Where the database lives."""
        return self._path

    @property
    def dimension(self) -> int:
        """Embedding dimension this store was built for."""
        return self._dimension

    def connection(self) -> sqlite3.Connection:
        """Return this thread's connection, opening one if needed.

        SQLite connections are not safe to share across threads. Rather than
        serialising every call behind a lock, each thread gets its own; WAL mode makes
        concurrent readers cheap.
        """
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn

        conn = sqlite3.connect(self._path, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.enable_load_extension(True)
            import sqlite_vec

            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            conn.close()
            raise ArcMemoryError(
                f"could not load sqlite-vec: {exc}. Install it with: pip install 'arc[memory]'"
            ) from exc

        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        self._local.conn = conn
        return conn

    def _initialize(self) -> None:
        """Create the schema and the dimension-dependent vector table."""
        conn = self.connection()
        try:
            conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
        except sqlite3.Error as exc:
            raise ArcMemoryError(f"could not initialise schema: {exc}") from exc

        # vec0 needs the dimension in its DDL, so this cannot live in schema.sql.
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_vectors USING vec0(
                memory_id INTEGER PRIMARY KEY,
                embedding FLOAT[{self._dimension}]
            )
            """
        )
        conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('dimension', ?)",
            (str(self._dimension),),
        )

        stored = conn.execute("SELECT value FROM meta WHERE key = 'dimension'").fetchone()
        if stored and int(stored["value"]) != self._dimension:
            raise ArcMemoryError(
                f"{self._path} was built for {stored['value']}-dimensional embeddings, "
                f"but the current embedder produces {self._dimension}. Re-embed with "
                f"`arc memory reindex`, or point ARC at a different database."
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block in one transaction, rolling back on failure.

        Explicit because ``isolation_level=None`` puts the connection in autocommit,
        which is what we want for reads but not for a multi-table write like adding a
        memory plus its vector plus its entity links.
        """
        conn = self.connection()
        conn.execute("BEGIN")
        try:
            yield conn
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    # ── Writes ──────────────────────────────────────────────────────────────────

    def add(
        self,
        *,
        layer: str,
        kind: str,
        content: str,
        embedding: list[float] | None = None,
        occurred_at: str | None = None,
        salience: float = 1.0,
        confidence: float = 1.0,
        source: str | None = None,
        source_url: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Insert a memory and its vector, returning the new id."""
        if not content.strip():
            raise ArcMemoryError("refusing to store an empty memory")
        if embedding is not None and len(embedding) != self._dimension:
            raise ArcMemoryError(
                f"embedding has {len(embedding)} dimensions, store expects {self._dimension}"
            )

        timestamp = now()
        with self.transaction() as conn:
            # Create the session row if the caller never opened one. Without this the
            # foreign key rejects the insert, and because MemoryService.remember_turn
            # deliberately swallows write failures, turns would vanish silently — the
            # worst possible failure for a memory system.
            if session_id is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, ?)",
                    (session_id, timestamp),
                )

            cursor = conn.execute(
                """
                INSERT INTO memories (
                    layer, kind, content, occurred_at, created_at,
                    salience, confidence, source, source_url, session_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    layer,
                    kind,
                    content,
                    occurred_at or timestamp,
                    timestamp,
                    salience,
                    confidence,
                    source,
                    source_url,
                    session_id,
                    json.dumps(metadata or {}),
                ),
            )
            memory_id = int(cursor.lastrowid or 0)

            if embedding is not None:
                conn.execute(
                    "INSERT INTO memory_vectors(memory_id, embedding) VALUES (?, ?)",
                    (memory_id, serialize_vector(embedding)),
                )

        return memory_id

    def set_embedding(self, memory_id: int, embedding: list[float]) -> None:
        """Attach or replace a memory's vector."""
        if len(embedding) != self._dimension:
            raise ArcMemoryError(
                f"embedding has {len(embedding)} dimensions, store expects {self._dimension}"
            )
        with self.transaction() as conn:
            conn.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
            conn.execute(
                "INSERT INTO memory_vectors(memory_id, embedding) VALUES (?, ?)",
                (memory_id, serialize_vector(embedding)),
            )

    def touch(self, memory_ids: list[int], *, boost: float = 0.05) -> None:
        """Record that memories were retrieved, nudging their salience up.

        Retrieval is the only evidence we have that a memory is useful, so it is what
        counters the decay consolidation applies. Salience is capped so a frequently
        hit memory cannot grow without bound and crowd out everything else.
        """
        if not memory_ids:
            return
        placeholders = ",".join("?" * len(memory_ids))
        with self.transaction() as conn:
            conn.execute(
                f"""
                UPDATE memories
                   SET accessed_at = ?,
                       access_count = access_count + 1,
                       salience = MIN(2.0, salience + ?)
                 WHERE id IN ({placeholders})
                """,
                (now(), boost, *memory_ids),
            )

    def supersede(self, memory_ids: list[int], replacement_id: int) -> None:
        """Mark memories as replaced without deleting them.

        §4.2 forbids silent memory mutation. Superseded rows stay readable so you can
        always see what consolidation replaced and with what.
        """
        if not memory_ids:
            return
        placeholders = ",".join("?" * len(memory_ids))
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE memories SET superseded_by = ? WHERE id IN ({placeholders})",
                (replacement_id, *memory_ids),
            )

    def forget(self, memory_id: int) -> bool:
        """Permanently delete a memory and its vector. Returns whether it existed.

        The one genuinely destructive operation, exposed because the brief asks for
        `arc memory forget`. Cascades handle entity links; the vector table has no
        foreign keys, so it is cleaned explicitly.
        """
        with self.transaction() as conn:
            conn.execute("DELETE FROM memory_vectors WHERE memory_id = ?", (memory_id,))
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            return cursor.rowcount > 0

    # ── Reads ───────────────────────────────────────────────────────────────────

    def get(self, memory_id: int) -> MemoryRecord | None:
        """Fetch one memory by id."""
        row = (
            self.connection()
            .execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
            .fetchone()
        )
        return MemoryRecord.from_row(row) if row else None

    def recent(
        self, *, limit: int = 20, layer: str | None = None, session_id: str | None = None
    ) -> list[MemoryRecord]:
        """Return the most recent live memories, newest first."""
        clauses = ["superseded_by IS NULL"]
        params: list[Any] = []
        if layer:
            clauses.append("layer = ?")
            params.append(layer)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)

        rows = (
            self.connection()
            .execute(
                f"SELECT * FROM memories WHERE {' AND '.join(clauses)} "
                f"ORDER BY occurred_at DESC LIMIT ?",
                (*params, limit),
            )
            .fetchall()
        )
        return [MemoryRecord.from_row(r) for r in rows]

    def count(self, *, layer: str | None = None, include_superseded: bool = False) -> int:
        """Count memories."""
        clauses = [] if include_superseded else ["superseded_by IS NULL"]
        params: list[Any] = []
        if layer:
            clauses.append("layer = ?")
            params.append(layer)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = (
            self.connection()
            .execute(f"SELECT COUNT(*) AS n FROM memories {where}", params)
            .fetchone()
        )
        return int(row["n"])

    def stats(self) -> dict[str, Any]:
        """Summarise the database, for ``arc memory stats``."""
        conn = self.connection()
        by_layer = {
            row["layer"]: row["n"]
            for row in conn.execute(
                "SELECT layer, COUNT(*) AS n FROM memories "
                "WHERE superseded_by IS NULL GROUP BY layer"
            ).fetchall()
        }
        scalar = conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM memories WHERE superseded_by IS NULL)     AS live,
                (SELECT COUNT(*) FROM memories WHERE superseded_by IS NOT NULL) AS superseded,
                (SELECT COUNT(*) FROM memory_vectors)                           AS vectors,
                (SELECT COUNT(*) FROM entities)                                 AS entities,
                (SELECT COUNT(*) FROM relations)                                AS relations,
                (SELECT COUNT(*) FROM sessions)                                 AS sessions
            """
        ).fetchone()

        return {
            "path": str(self._path),
            "size_mb": round(self._path.stat().st_size / 1024**2, 2)
            if self._path.exists()
            else 0.0,
            "dimension": self._dimension,
            "live_memories": scalar["live"],
            "superseded": scalar["superseded"],
            "embedded": scalar["vectors"],
            "by_layer": by_layer,
            "entities": scalar["entities"],
            "relations": scalar["relations"],
            "sessions": scalar["sessions"],
        }

    def iter_all(self, *, include_superseded: bool = False) -> Iterator[MemoryRecord]:
        """Stream every memory, for export.

        A generator rather than a list so exporting a large database does not need it
        all resident at once.
        """
        where = "" if include_superseded else "WHERE superseded_by IS NULL"
        for row in self.connection().execute(f"SELECT * FROM memories {where} ORDER BY id"):
            yield MemoryRecord.from_row(row)

    # ── Sessions ────────────────────────────────────────────────────────────────

    def start_session(self, session_id: str, *, model: str | None = None) -> None:
        """Record the start of a conversation."""
        with self.transaction() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions(id, started_at, model) VALUES (?, ?, ?)",
                (session_id, now(), model),
            )

    def end_session(self, session_id: str) -> None:
        """Mark a conversation finished and record its turn count."""
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE sessions
                   SET ended_at = ?,
                       turn_count = (SELECT COUNT(*) FROM memories WHERE session_id = ?)
                 WHERE id = ?
                """,
                (now(), session_id, session_id),
            )

    def close(self) -> None:
        """Close this thread's connection."""
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
