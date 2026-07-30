"""Wiring the memory subsystem together.

The layers, retriever, and consolidator each take a store and an embedder. Rather than
making every caller construct five objects in the right order, this assembles them once
from config and hands back a single facade.

It also owns the decision of *when* memory is used, which is genuinely a policy
question: what gets remembered from a chat turn, and what gets recalled before one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc.audit import AuditLogger
from arc.config import Config
from arc.log import get_logger
from arc.memory.consolidation import ConsolidationPolicy, Consolidator
from arc.memory.embedder import Embedder
from arc.memory.episodic import EpisodicMemory, SupportsEmbedding
from arc.memory.procedural import ProceduralMemory
from arc.memory.retrieval import Hit, Retriever
from arc.memory.semantic import SemanticMemory
from arc.memory.store import MemoryRecord, MemoryStore
from arc.paths import arc_home

_log = get_logger(__name__)


@dataclass
class MemoryService:
    """Everything memory-related, assembled and ready to use."""

    store: MemoryStore
    embedder: SupportsEmbedding
    episodic: EpisodicMemory
    semantic: SemanticMemory
    procedural: ProceduralMemory
    retriever: Retriever
    consolidator: Consolidator
    retrieval_limit: int = 6
    #: Candidates each retrieval strategy contributes before fusion.
    per_strategy: int = 20
    #: Tokens held back for the model's reply when packing a prompt.
    reserve_for_reply: int = 1024

    @classmethod
    def from_config(
        cls,
        config: Config,
        *,
        embedder: SupportsEmbedding | None = None,
        audit: AuditLogger | None = None,
        path: Path | None = None,
    ) -> MemoryService:
        """Build the whole subsystem from ``config``.

        ``embedder`` is injectable so tests can pass the hash-based stand-in and skip
        a model download and ONNX startup.
        """
        section = config.section("memory")
        embed_config = section.get("embedder") or {}

        resolved = embedder or Embedder(
            repo=str(embed_config.get("repo", "BAAI/bge-small-en-v1.5")),
            dimension=int(embed_config.get("dimension", 384)),
            max_tokens=int(embed_config.get("max_tokens", 512)),
        )

        db_path = path or (
            Path(str(section["path"])) if section.get("path") else arc_home() / "memory.db"
        )
        store = MemoryStore(db_path, dimension=resolved.dimension)

        semantic = SemanticMemory(store, resolved)
        retrieval_config = section.get("retrieval") or {}
        consolidation_config = section.get("consolidation") or {}

        return cls(
            store=store,
            embedder=resolved,
            episodic=EpisodicMemory(store, resolved),
            semantic=semantic,
            procedural=ProceduralMemory(store, resolved),
            retriever=Retriever(
                store,
                resolved,
                semantic,
                weights=retrieval_config.get("weights"),
            ),
            consolidator=Consolidator(
                store,
                resolved,
                policy=_policy_from(consolidation_config),
                audit=audit,
            ),
            retrieval_limit=int(retrieval_config.get("limit", 6)),
            per_strategy=int(retrieval_config.get("per_strategy", 20)),
            reserve_for_reply=int((section.get("working") or {}).get("reserve_for_reply", 1024)),
        )

    # ── Chat integration ────────────────────────────────────────────────────────

    def recall(self, query: str, *, limit: int | None = None) -> list[Hit]:
        """Retrieve memories relevant to a query.

        Failures are swallowed and logged rather than raised: a chat turn that cannot
        reach memory should answer without it, not crash. This is the one place where
        degrading quietly beats failing loudly, because the alternative makes the whole
        assistant unusable whenever the database is busy.
        """
        try:
            return self.retriever.search(
                query,
                limit=limit or self.retrieval_limit,
                per_strategy=self.per_strategy,
            )
        except Exception as exc:
            _log.warning("recall failed, continuing without memory: %s", exc)
            return []

    def applicable_procedures(self, text: str) -> list[MemoryRecord]:
        """Return procedural memories whose trigger matches this input."""
        try:
            matches = self.procedural.matching(text)
            records = [self.store.get(p.id) for p in matches]
            return [r for r in records if r is not None]
        except Exception as exc:
            _log.warning("procedure lookup failed: %s", exc)
            return []

    def remember_turn(self, role: str, content: str, *, session_id: str) -> int | None:
        """Store one side of a conversation and wire it into the entity graph.

        Entity extraction runs here rather than in ``EpisodicMemory`` because it needs
        the semantic layer, and episodic must not depend on it. Without this the graph
        only ever fills from explicit ``add_fact`` calls, which means it stays empty
        during ordinary use and the graph retrieval strategy contributes nothing.

        Extraction runs on user turns only. The assistant's own words are its output,
        not evidence about the world, and indexing them lets the model's inventions
        become graph nodes.

        Returns None on failure rather than raising, for the same reason as ``recall``:
        losing one turn is bad, losing the conversation is worse.
        """
        try:
            memory_id = self.episodic.record_turn(role, content, session_id=session_id)
        except Exception as exc:
            _log.warning("could not store turn: %s", exc)
            return None

        if role == "user":
            try:
                names = self.semantic.extract_entities(content)
                if names:
                    self.semantic.link(memory_id, [self.semantic.upsert_entity(n) for n in names])
            except Exception as exc:
                # The turn is already stored; failing to index it is a degradation,
                # not a reason to lose it.
                _log.warning("entity extraction failed for turn %d: %s", memory_id, exc)

        return memory_id

    def stats(self) -> dict[str, Any]:
        """Database summary plus which embedder produced the vectors."""
        return {**self.store.stats(), "embedder": self.embedder.name}

    def close(self) -> None:
        """Release the database connection."""
        self.store.close()


def _policy_from(raw: dict[str, Any]) -> ConsolidationPolicy:
    """Build a consolidation policy from config, falling back to the safe defaults."""
    defaults = ConsolidationPolicy()
    return ConsolidationPolicy(
        dedupe_threshold=float(raw.get("dedupe_threshold", defaults.dedupe_threshold)),
        decay_per_day=float(raw.get("decay_per_day", defaults.decay_per_day)),
        prune_below=float(raw.get("prune_below", defaults.prune_below)),
        summarize_after_turns=int(raw.get("summarize_after_turns", defaults.summarize_after_turns)),
        promote_after=int(raw.get("promote_after", defaults.promote_after)),
        max_per_run=int(raw.get("max_per_run", defaults.max_per_run)),
        allow_prune=bool(raw.get("allow_prune", defaults.allow_prune)),
    )
