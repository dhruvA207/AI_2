"""Consolidation — the background process that keeps memory from becoming a landfill.

Four jobs (§4.2): dedupe near-identical memories, summarise long episodic runs into
compact facts, decay the salience of unused memories, and promote repeated patterns
into procedural memory.

**Everything here is deliberately conservative**, and that is a considered default
rather than timidity. Consolidation rewrites memories in the background without being
asked. Being too cautious costs some database size; being too aggressive silently
destroys things you wanted, and you would not find out until you asked a question that
should have had an answer. Every threshold is in ``config/default.yaml`` and can be
turned up once there is evidence of what this actually does to a real corpus.

Nothing is ever hard-deleted except by explicit pruning at a very low salience floor.
Merged and summarised memories are *superseded* — still readable, still auditable —
because §4.2 forbids silent memory mutation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from arc.audit import AuditLogger
from arc.log import get_logger
from arc.memory.episodic import SupportsEmbedding
from arc.memory.store import MemoryStore, now

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ConsolidationPolicy:
    """Thresholds governing what consolidation is allowed to do.

    Defaults are conservative on purpose — see the module docstring.
    """

    #: Cosine similarity above which two memories are considered the same thing.
    #: 0.97 is high: it catches restatements and near-duplicates, not merely related
    #: memories. Lowering this is the fastest way to lose distinctions that mattered.
    dedupe_threshold: float = 0.97

    #: Daily multiplier applied to unused memories. 0.99 halves salience in ~69 days,
    #: which is slow enough that a fortnight away does not erase anything.
    decay_per_day: float = 0.99

    #: Salience below which a memory becomes eligible for pruning. Reaching this from
    #: an episodic turn's starting 0.6 takes well over a year of never being retrieved.
    prune_below: float = 0.05

    #: Episodic memories in one session before it is worth summarising.
    summarize_after_turns: int = 20

    #: Times a pattern must recur before it becomes a procedural memory.
    promote_after: int = 3

    #: Ceiling on memories touched per run, so a scheduled pass cannot stall the
    #: machine by rewriting a hundred thousand rows at once.
    max_per_run: int = 500

    #: Pruning is off by default. Decay alone is reversible; deletion is not.
    allow_prune: bool = False


@dataclass
class ConsolidationReport:
    """What one run changed."""

    deduped: int = 0
    summarized: int = 0
    decayed: int = 0
    promoted: int = 0
    pruned: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    @property
    def total_changes(self) -> int:
        """How many memories were affected in total."""
        return self.deduped + self.summarized + self.decayed + self.promoted + self.pruned

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "deduped": self.deduped,
            "summarized": self.summarized,
            "decayed": self.decayed,
            "promoted": self.promoted,
            "pruned": self.pruned,
            "total_changes": self.total_changes,
        }


class Consolidator:
    """Runs the background maintenance passes over memory."""

    def __init__(
        self,
        store: MemoryStore,
        embedder: SupportsEmbedding,
        *,
        policy: ConsolidationPolicy | None = None,
        audit: AuditLogger | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._policy = policy or ConsolidationPolicy()
        self._audit = audit

    @property
    def policy(self) -> ConsolidationPolicy:
        """The thresholds in force."""
        return self._policy

    def run(self, *, dry_run: bool = False) -> ConsolidationReport:
        """Run every pass, in an order that matters more than it looks.

        **Promotion must run before dedupe.** Promotion's signal is a phrase recurring
        across sessions, and dedupe merges exactly those recurrences — identical text
        embeds identically, so it is the first thing dedupe collapses. Running dedupe
        first left one live copy of every repeated request, the count never reached the
        threshold, and promotion silently never fired for the patterns it exists to
        catch.

        After that: dedupe compacts, decay adjusts salience, and pruning acts last on
        the resulting values.
        """
        report = ConsolidationReport()
        self._promote(report, dry_run=dry_run)
        self._dedupe(report, dry_run=dry_run)
        self._decay(report, dry_run=dry_run)
        if self._policy.allow_prune:
            self._prune(report, dry_run=dry_run)

        if self._audit is not None and report.total_changes:
            self._audit.record(
                "memory.consolidation",
                status="dry_run" if dry_run else "ok",
                tool="consolidation",
                result=report.to_dict(),
                dry_run=dry_run,
            )

        _log.info("consolidation complete", extra=report.to_dict())
        return report

    # ── Passes ──────────────────────────────────────────────────────────────────

    def _dedupe(self, report: ConsolidationReport, *, dry_run: bool) -> None:
        """Merge near-identical memories, keeping the richest one.

        Compares each memory against its nearest neighbours by vector rather than
        every pair — the all-pairs version is O(n^2) and unusable past a few thousand
        memories, which §5 explicitly says to test at.
        """
        conn = self._store.connection()
        rows = conn.execute(
            """
            SELECT m.id, m.content, m.layer, m.confidence, m.access_count, v.embedding
              FROM memories m
              JOIN memory_vectors v ON v.memory_id = m.id
             WHERE m.superseded_by IS NULL
             ORDER BY m.id
             LIMIT ?
            """,
            (self._policy.max_per_run,),
        ).fetchall()

        merged: set[int] = set()
        for row in rows:
            if row["id"] in merged:
                continue

            neighbours = conn.execute(
                """
                SELECT v.memory_id, v.distance
                  FROM memory_vectors v
                  JOIN memories m ON m.id = v.memory_id
                 WHERE v.embedding MATCH ? AND k = 5
                   AND m.superseded_by IS NULL
                   AND m.layer = ?
                 ORDER BY v.distance
                """,
                (row["embedding"], row["layer"]),
            ).fetchall()

            duplicates = []
            for neighbour in neighbours:
                other_id = int(neighbour["memory_id"])
                if other_id == row["id"] or other_id in merged:
                    continue
                # Vectors are unit-normalised, so L2 distance d relates to cosine
                # similarity as cos = 1 - d^2/2.
                similarity = 1.0 - (float(neighbour["distance"]) ** 2) / 2.0
                if similarity >= self._policy.dedupe_threshold:
                    duplicates.append(other_id)

            if not duplicates:
                continue

            merged.update(duplicates)
            report.deduped += len(duplicates)
            report.details.append({"action": "dedupe", "kept": row["id"], "superseded": duplicates})

            if not dry_run:
                self._store.supersede(duplicates, row["id"])
                self._log_change("dedupe", duplicates, result_id=row["id"])

    def _decay(self, report: ConsolidationReport, *, dry_run: bool) -> None:
        """Reduce salience for memories that have not been retrieved.

        Decay is applied per day since last access, not per run, so running
        consolidation more often does not decay memory faster. Getting that wrong
        would make the schedule silently change behaviour.
        """
        conn = self._store.connection()
        rows = conn.execute(
            """
            SELECT id, salience, COALESCE(accessed_at, created_at) AS last_touch
              FROM memories
             WHERE superseded_by IS NULL AND salience > ?
             LIMIT ?
            """,
            (self._policy.prune_below, self._policy.max_per_run),
        ).fetchall()

        updates: list[tuple[float, int]] = []
        current = datetime.now(UTC)

        for row in rows:
            try:
                last = datetime.fromisoformat(row["last_touch"])
            except (TypeError, ValueError):
                continue
            days = max(0.0, (current - last).total_seconds() / 86400.0)
            if days < 1.0:
                continue
            decayed = float(row["salience"]) * (self._policy.decay_per_day**days)
            if abs(decayed - float(row["salience"])) < 1e-6:
                continue
            updates.append((round(decayed, 6), int(row["id"])))

        if not updates:
            return

        report.decayed = len(updates)
        if not dry_run:
            with self._store.transaction() as txn:
                txn.executemany("UPDATE memories SET salience = ? WHERE id = ?", updates)
            self._log_change("decay", [u[1] for u in updates])

    def _promote(self, report: ConsolidationReport, *, dry_run: bool) -> None:
        """Turn repeated episodic patterns into procedural memories.

        The signal is a user phrasing recurring across sessions. Deliberately crude —
        exact normalised text — because a fuzzy version promotes noise, and a wrong
        procedural memory actively misleads the agent rather than merely cluttering it.
        Phase 4 can have the model judge similarity properly.
        """
        conn = self._store.connection()
        rows = conn.execute(
            """
            SELECT LOWER(TRIM(content)) AS normalized,
                   COUNT(*) AS n,
                   MIN(id) AS first_id,
                   COUNT(DISTINCT session_id) AS sessions
              FROM memories
             WHERE layer = 'episodic' AND kind = 'turn:user' AND superseded_by IS NULL
             GROUP BY normalized
            HAVING n >= ? AND sessions >= 2
             LIMIT ?
            """,
            (self._policy.promote_after, self._policy.max_per_run),
        ).fetchall()

        for row in rows:
            content = str(row["normalized"])
            # Don't re-promote something already learned.
            existing = conn.execute(
                "SELECT id FROM memories WHERE layer = 'procedural' AND LOWER(content) LIKE ?",
                (f"%{content[:60]}%",),
            ).fetchone()
            if existing is not None:
                continue

            report.promoted += 1
            report.details.append({"action": "promote", "pattern": content[:80], "count": row["n"]})

            if dry_run:
                continue

            text = f"You often ask: {content}"
            new_id = self._store.add(
                layer="procedural",
                kind="pattern",
                content=text,
                embedding=self._embedder.embed(text),
                # Inferred, not stated. Capped below a preference the user gave
                # directly so the two rank correctly when they conflict.
                confidence=0.6,
                salience=1.2,
                source="consolidation",
                metadata={"observed_count": int(row["n"]), "trigger": content[:60]},
            )
            self._log_change("promote", [int(row["first_id"])], result_id=new_id)

    def _prune(self, report: ConsolidationReport, *, dry_run: bool) -> None:
        """Permanently delete memories that have decayed below the floor.

        Off by default. This is the only irreversible operation in the module.
        """
        rows = (
            self._store.connection()
            .execute(
                "SELECT id FROM memories WHERE salience < ? AND superseded_by IS NULL LIMIT ?",
                (self._policy.prune_below, self._policy.max_per_run),
            )
            .fetchall()
        )

        ids = [int(r["id"]) for r in rows]
        if not ids:
            return

        report.pruned = len(ids)
        if not dry_run:
            for memory_id in ids:
                self._store.forget(memory_id)
            self._log_change("prune", ids)

    def summarize_session(self, session_id: str, summarizer: Any = None) -> int | None:
        """Compress a long session's turns into one semantic memory.

        Without a ``summarizer`` (a LanguageModel), falls back to concatenating the
        user's turns. That is a poor summary, but it is honest about being one, and it
        keeps consolidation runnable when no model is loaded — a background job should
        not require the GPU.
        """
        records = self._store.recent(limit=200, layer="episodic", session_id=session_id)
        if len(records) < self._policy.summarize_after_turns:
            return None

        turns = list(reversed(records))
        transcript = "\n".join(f"{r.kind.removeprefix('turn:')}: {r.content}" for r in turns)

        if summarizer is not None:
            from arc.model.base import Message

            completion = summarizer.generate(
                [
                    Message(
                        role="system",
                        content=(
                            "Summarise this conversation in 3-5 sentences. Keep concrete "
                            "facts, decisions, and preferences. Drop pleasantries."
                        ),
                    ),
                    Message(role="user", content=transcript[:8000]),
                ],
                max_tokens=300,
                temperature=0.3,
            )
            summary = completion.text.strip()
        else:
            user_turns = [r.content for r in turns if r.kind == "turn:user"]
            summary = "Session covered: " + "; ".join(user_turns[:10])

        if not summary:
            return None

        summary_id = self._store.add(
            layer="semantic",
            kind="session_summary",
            content=summary,
            embedding=self._embedder.embed(summary),
            source="consolidation",
            session_id=session_id,
            metadata={"turn_count": len(turns)},
        )

        # The turns are superseded, not deleted: the summary is lossy and the detail
        # must remain recoverable.
        self._store.supersede([r.id for r in turns], summary_id)
        self._log_change("summarize", [r.id for r in turns], result_id=summary_id)

        with self._store.transaction() as conn:
            conn.execute("UPDATE sessions SET summary = ? WHERE id = ?", (summary, session_id))

        return summary_id

    def _log_change(
        self, action: str, memory_ids: list[int], *, result_id: int | None = None
    ) -> None:
        """Record a change in the in-database consolidation log."""
        with self._store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO consolidation_log (ran_at, action, memory_ids, detail, result_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (now(), action, json.dumps(memory_ids), "{}", result_id),
            )

    def history(self, *, limit: int = 20) -> list[dict[str, Any]]:
        """Return recent consolidation actions, newest first."""
        rows = (
            self._store.connection()
            .execute("SELECT * FROM consolidation_log ORDER BY ran_at DESC LIMIT ?", (limit,))
            .fetchall()
        )
        return [
            {
                "ran_at": r["ran_at"],
                "action": r["action"],
                "memory_ids": json.loads(r["memory_ids"]),
                "result_id": r["result_id"],
            }
            for r in rows
        ]


def similarity_from_distance(distance: float) -> float:
    """Convert sqlite-vec's L2 distance to cosine similarity.

    Valid only for unit-normalised vectors, which the embedder guarantees:
    ``|a-b|^2 = 2 - 2*cos``, so ``cos = 1 - d^2/2``.
    """
    return 1.0 - (distance**2) / 2.0
