"""Tests for the three memory layers, retrieval, and consolidation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from arc.memory import HashEmbedder, MemoryStore
from arc.memory.consolidation import (
    ConsolidationPolicy,
    Consolidator,
    similarity_from_distance,
)
from arc.memory.episodic import EpisodicMemory
from arc.memory.procedural import ProceduralMemory
from arc.memory.retrieval import Retriever, _sanitize_fts_query
from arc.memory.semantic import SemanticMemory
from arc.memory.working import Budget, WorkingMemory
from tests.fakes import FakeModel


@pytest.fixture
def parts(tmp_path: Path) -> tuple[MemoryStore, HashEmbedder]:
    return MemoryStore(tmp_path / "m.db", dimension=128), HashEmbedder(128)


# ── Episodic ────────────────────────────────────────────────────────────────────


def test_turns_are_stored_for_both_roles(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """What ARC said is as much "what happened" as what was asked."""
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    store.start_session("s1")
    episodic.record_turn("user", "question", session_id="s1")
    episodic.record_turn("assistant", "answer", session_id="s1")
    assert [r.kind for r in episodic.session_history("s1")] == ["turn:user", "turn:assistant"]


def test_session_history_is_chronological(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    store.start_session("s1")
    for i in range(5):
        episodic.record_turn("user", f"turn {i}", session_id="s1")
    assert [r.content for r in episodic.session_history("s1")] == [f"turn {i}" for i in range(5)]


def test_turns_start_below_full_salience(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """A passing remark should not compete with a deliberate fact."""
    store, emb = parts
    mid = EpisodicMemory(store, emb).record_turn("user", "chatter", session_id="s1")
    record = store.get(mid)
    assert record is not None
    assert record.salience < 1.0


# ── Semantic ────────────────────────────────────────────────────────────────────


def test_entity_is_one_node_regardless_of_kind(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """Regression: keying identity on (name, kind) split the graph into disconnected
    halves when the same thing was mentioned under two kinds."""
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    first = semantic.upsert_entity("ARC", "project")
    second = semantic.upsert_entity("ARC", "concept")
    assert first == second


def test_entity_kind_is_promoted_not_demoted(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    semantic.upsert_entity("Helios", "concept")
    semantic.upsert_entity("Helios", "project")
    entity = semantic.find_entity("Helios")
    assert entity is not None
    assert entity.kind == "project"

    semantic.upsert_entity("Helios", "concept")
    entity = semantic.find_entity("Helios")
    assert entity is not None
    assert entity.kind == "project"


def test_entity_matching_is_case_insensitive(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    assert semantic.upsert_entity("Dhruv") == semantic.upsert_entity("dhruv")


def test_empty_entity_name_is_rejected(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    with pytest.raises(ValueError, match="cannot be empty"):
        SemanticMemory(store, emb).upsert_entity("   ")


def test_relations_are_visible_from_both_directions(
    parts: tuple[MemoryStore, HashEmbedder],
) -> None:
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    semantic.relate("Dhruv", "owns", "ARC")
    semantic.relate("ARC", "uses", "Qwen3")
    assert len(semantic.relations_for("ARC")) == 2


def test_multi_hop_traversal(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """The capability a pure vector store cannot provide at all."""
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    semantic.relate("Dhruv", "owns", "ARC")
    semantic.relate("ARC", "uses", "Qwen3")
    semantic.relate("Qwen3", "licensed_under", "Apache-2.0")

    names = [e.name for e in semantic.neighbours("Dhruv", hops=3)]
    assert "ARC" in names
    assert "Qwen3" in names
    assert "Apache-2.0" in names


def test_traversal_excludes_the_origin(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """Regression: an undirected walk returns to its start at depth 2."""
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    semantic.relate("A", "links", "B")
    assert "A" not in [e.name for e in semantic.neighbours("A", hops=2)]


def test_traversal_respects_hop_limit(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    semantic.relate("A", "links", "B")
    semantic.relate("B", "links", "C")
    assert [e.name for e in semantic.neighbours("A", hops=1)] == ["B"]


def test_traversal_of_unknown_entity(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    assert SemanticMemory(store, emb).neighbours("nobody") == []


def test_relate_is_idempotent(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    semantic.relate("A", "links", "B")
    semantic.relate("A", "links", "B")
    assert len(semantic.relations_for("A")) == 1


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Dhruv works on ARC", ["Dhruv", "ARC"]),
        ("the router failed", []),
        ("When did That happen", []),
    ],
)
def test_entity_extraction(
    parts: tuple[MemoryStore, HashEmbedder], text: str, expected: list[str]
) -> None:
    store, emb = parts
    assert SemanticMemory(store, emb).extract_entities(text) == expected


def test_web_facts_carry_provenance_and_lower_confidence(
    parts: tuple[MemoryStore, HashEmbedder],
) -> None:
    """§4.4: a web fact must cite its source and not outrank what the user said."""
    store, emb = parts
    mid = SemanticMemory(store, emb).add_web_fact("Python 3.13 is out", source_url="https://x.dev")
    record = store.get(mid)
    assert record is not None
    assert record.source_url == "https://x.dev"
    assert record.confidence < 1.0
    assert "retrieved_at" in (record.metadata or {})


def test_memories_for_entities(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    semantic.add_fact("Dhruv likes Rust", entities=["Dhruv"])
    assert len(semantic.memories_for_entities(["Dhruv"])) == 1
    assert semantic.memories_for_entities([]) == []


# ── Procedural ──────────────────────────────────────────────────────────────────


def test_stated_preference_outranks_inferred(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    procedural = ProceduralMemory(store, emb)
    stated = procedural.add_preference("Always use tabs", confidence=1.0)
    inferred = procedural.add_workflow("tidy up", ["a", "b"], confidence=0.6)
    assert store.get(stated).confidence > store.get(inferred).confidence  # type: ignore[union-attr]


def test_trigger_matching_is_substring_based(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    procedural = ProceduralMemory(store, emb)
    procedural.add_preference("means the Downloads folder", trigger="clean up")
    assert len(procedural.matching("could you clean up please")) == 1
    assert procedural.matching("something unrelated") == []


def test_reinforce_raises_confidence_asymptotically(
    parts: tuple[MemoryStore, HashEmbedder],
) -> None:
    """An inferred pattern must never reach the certainty of a stated one."""
    store, emb = parts
    procedural = ProceduralMemory(store, emb)
    mid = procedural.add_workflow("x", ["y"], confidence=0.5)
    for _ in range(50):
        procedural.reinforce(mid)
    record = store.get(mid)
    assert record is not None
    assert record.confidence <= 0.99
    assert (record.metadata or {})["observed_count"] == 51


def test_reinforce_on_missing_memory_is_a_noop(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    ProceduralMemory(store, emb).reinforce(999)


# ── Retrieval ───────────────────────────────────────────────────────────────────


@pytest.fixture
def populated(parts: tuple[MemoryStore, HashEmbedder]) -> tuple[MemoryStore, Retriever]:
    store, emb = parts
    semantic = SemanticMemory(store, emb)
    episodic = EpisodicMemory(store, emb)
    store.start_session("s1")
    semantic.add_fact("Dhruv prefers dark mode in his editor", entities=["Dhruv"])
    semantic.add_fact("ARC is a local-first assistant", entities=["ARC"])
    semantic.relate("Dhruv", "owns", "ARC")
    episodic.record_turn("user", "how do I fix the router fallback bug", session_id="s1")
    ProceduralMemory(store, emb).add_preference("cleanup means Downloads", trigger="cleanup")
    return store, Retriever(store, emb, semantic)


def test_relevance_beats_layer_salience(populated: tuple[MemoryStore, Retriever]) -> None:
    """Regression: multiplying by salience let a 1.5-salience preference outrank a
    turn that matched the query text exactly."""
    _, retriever = populated
    hits = retriever.search("router fallback bug", limit=1)
    assert "router" in hits[0].record.content


def test_keyword_finds_exact_terms(populated: tuple[MemoryStore, Retriever]) -> None:
    _, retriever = populated
    assert retriever.by_keyword("router") != []


def test_graph_reaches_related_memories(populated: tuple[MemoryStore, Retriever]) -> None:
    _, retriever = populated
    assert retriever.by_graph("Dhruv") != []


def test_search_records_which_strategies_matched(
    populated: tuple[MemoryStore, Retriever],
) -> None:
    """ "Why did this come back?" is the first question when retrieval misbehaves."""
    _, retriever = populated
    hits = retriever.search("Dhruv dark mode", limit=3)
    assert any(len(h.sources) > 1 for h in hits)


def test_search_on_empty_query(populated: tuple[MemoryStore, Retriever]) -> None:
    _, retriever = populated
    assert retriever.search("   ") == []


def test_search_touches_returned_memories(populated: tuple[MemoryStore, Retriever]) -> None:
    store, retriever = populated
    hits = retriever.search("dark mode", limit=1)
    record = store.get(hits[0].record.id)
    assert record is not None
    assert record.access_count >= 1


def test_search_can_skip_touching(populated: tuple[MemoryStore, Retriever]) -> None:
    store, retriever = populated
    hits = retriever.search("dark mode", limit=1, touch=False)
    record = store.get(hits[0].record.id)
    assert record is not None
    assert record.access_count == 0


def test_superseded_memories_are_never_retrieved(
    populated: tuple[MemoryStore, Retriever],
) -> None:
    store, retriever = populated
    target = retriever.search("dark mode", limit=1)[0].record.id
    replacement = store.add(layer="semantic", kind="fact", content="unrelated")
    store.supersede([target], replacement)
    assert all(h.record.id != target for h in retriever.search("dark mode", limit=10))


@pytest.mark.parametrize(
    "query",
    ['what "is" this', "a:b:c", "foo* bar(", "-minus", "NEAR(x y)", "''", "AND OR NOT"],
)
def test_fts_punctuation_does_not_crash_retrieval(
    populated: tuple[MemoryStore, Retriever], query: str
) -> None:
    """FTS5 treats quotes, colons, parens and hyphens as operators, so an ordinary
    question containing any of them would be a syntax error rather than a search."""
    _, retriever = populated
    retriever.search(query, limit=3)


def test_sanitizer_quotes_terms() -> None:
    assert _sanitize_fts_query('hello "world"') == '"hello" OR "world"'
    assert _sanitize_fts_query("...") == ""


def test_one_failing_strategy_does_not_fail_the_search(
    populated: tuple[MemoryStore, Retriever], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Degraded retrieval beats no retrieval."""
    _, retriever = populated

    def boom(*_a: object, **_k: object) -> list[int]:
        raise RuntimeError("vector index corrupt")

    monkeypatch.setattr(retriever, "by_vector", boom)
    assert retriever.search("router", limit=3) != []


# ── Working memory ──────────────────────────────────────────────────────────────


def test_budget_reserves_room_for_the_reply() -> None:
    budget = Budget(total=1000, reserved_for_reply=200)
    assert budget.usable == 800
    assert budget.allocation("retrieved") == 280


def test_budget_cannot_go_negative() -> None:
    assert Budget(total=100, reserved_for_reply=500).usable == 0


def test_pack_memories_respects_the_allowance(
    populated: tuple[MemoryStore, Retriever],
) -> None:
    _, retriever = populated
    hits = retriever.search("Dhruv", limit=10)
    working = WorkingMemory(counter=FakeModel(), budget=Budget(total=100, reserved_for_reply=50))
    assert len(working.pack_memories(hits)) < len(hits)


def test_pack_memories_skips_rather_than_stops(
    populated: tuple[MemoryStore, Retriever],
) -> None:
    """A long memory must not block shorter, still-relevant ones behind it."""
    _, retriever = populated
    hits = retriever.search("Dhruv", limit=10)
    working = WorkingMemory.for_model(FakeModel(context_length=100000), reserve=0)
    assert len(working.pack_memories(hits)) == len(hits)


def test_pack_history_keeps_the_most_recent(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """Dropping the start of a conversation beats dropping what was just said."""
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    store.start_session("s1")
    for i in range(20):
        episodic.record_turn("user", f"turn number {i}", session_id="s1")

    records = episodic.session_history("s1")
    working = WorkingMemory(counter=FakeModel(), budget=Budget(total=200, reserved_for_reply=0))
    kept = working.pack_history(records)
    assert kept
    assert kept[-1].content == "turn number 19"


def test_render_includes_provenance(populated: tuple[MemoryStore, Retriever]) -> None:
    """§4.4: ARC cannot cite a source it was never shown."""
    store, retriever = populated
    SemanticMemory(store, HashEmbedder(128)).add_web_fact("X is true", source_url="https://s.io")
    hits = retriever.search("X is true", limit=5)
    rendered = WorkingMemory.for_model(FakeModel()).render_memories(hits)
    assert "https://s.io" in rendered


def test_render_empty_sections() -> None:
    working = WorkingMemory.for_model(FakeModel())
    assert working.render_memories([]) == ""
    assert working.render_procedures([]) == ""


# ── Consolidation ───────────────────────────────────────────────────────────────


def test_similarity_conversion() -> None:
    """Unit vectors: |a-b|^2 = 2 - 2cos, so cos = 1 - d^2/2."""
    assert similarity_from_distance(0.0) == pytest.approx(1.0)
    assert similarity_from_distance(2.0**0.5) == pytest.approx(0.0)


def test_dedupe_supersedes_identical_memories(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    for _ in range(3):
        store.add(
            layer="semantic",
            kind="fact",
            content="the sky is blue",
            embedding=emb.embed("the sky is blue"),
        )

    report = Consolidator(store, emb).run()
    assert report.deduped == 2
    assert store.count() == 1
    assert store.count(include_superseded=True) == 3


def test_dedupe_leaves_distinct_memories_alone(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    for text in ("cats are soft", "quantum tunnelling is weird", "the build is broken"):
        store.add(layer="semantic", kind="fact", content=text, embedding=emb.embed(text))
    assert Consolidator(store, emb).run().deduped == 0


def test_dedupe_does_not_merge_across_layers(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """A turn and a fact with the same words are different kinds of thing."""
    store, emb = parts
    store.add(
        layer="semantic", kind="fact", content="same words", embedding=emb.embed("same words")
    )
    store.add(
        layer="episodic", kind="event", content="same words", embedding=emb.embed("same words")
    )
    assert Consolidator(store, emb).run().deduped == 0


def test_dry_run_changes_nothing(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    for _ in range(3):
        store.add(
            layer="semantic", kind="fact", content="duplicate", embedding=emb.embed("duplicate")
        )
    report = Consolidator(store, emb).run(dry_run=True)
    assert report.deduped == 2
    assert store.count() == 3


def test_decay_reduces_salience_of_stale_memories(
    parts: tuple[MemoryStore, HashEmbedder],
) -> None:
    store, emb = parts
    old = (datetime.now(UTC) - timedelta(days=100)).isoformat()
    mid = store.add(
        layer="semantic",
        kind="fact",
        content="forgotten",
        embedding=emb.embed("forgotten"),
        occurred_at=old,
    )
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET created_at = ? WHERE id = ?", (old, mid))

    Consolidator(store, emb).run()
    record = store.get(mid)
    assert record is not None
    assert record.salience < 1.0


def test_decay_leaves_fresh_memories_alone(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """Decay is per-day, so running consolidation twice must not decay twice."""
    store, emb = parts
    mid = store.add(layer="semantic", kind="fact", content="fresh", embedding=emb.embed("fresh"))
    Consolidator(store, emb).run()
    Consolidator(store, emb).run()
    record = store.get(mid)
    assert record is not None
    assert record.salience == pytest.approx(1.0)


def test_pruning_is_off_by_default(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """Decay is reversible; deletion is not."""
    store, emb = parts
    mid = store.add(layer="semantic", kind="fact", content="doomed", embedding=emb.embed("doomed"))
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET salience = 0.001 WHERE id = ?", (mid,))
    assert Consolidator(store, emb).run().pruned == 0
    assert store.get(mid) is not None


def test_pruning_deletes_when_enabled(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    mid = store.add(layer="semantic", kind="fact", content="doomed", embedding=emb.embed("doomed"))
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET salience = 0.001 WHERE id = ?", (mid,))

    consolidator = Consolidator(store, emb, policy=ConsolidationPolicy(allow_prune=True))
    assert consolidator.run().pruned == 1
    assert store.get(mid) is None


def test_promotion_requires_repetition_across_sessions(
    parts: tuple[MemoryStore, HashEmbedder],
) -> None:
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    for session in ("s1", "s2", "s3"):
        store.start_session(session)
        episodic.record_turn("user", "clean up my downloads", session_id=session)

    assert Consolidator(store, emb).run().promoted == 1
    assert store.count(layer="procedural") == 1


def test_promotion_ignores_one_off_phrases(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """One coincidence must not become a rule."""
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    store.start_session("s1")
    episodic.record_turn("user", "something said once", session_id="s1")
    assert Consolidator(store, emb).run().promoted == 0


def test_promoted_patterns_are_less_confident_than_stated_ones(
    parts: tuple[MemoryStore, HashEmbedder],
) -> None:
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    for session in ("s1", "s2", "s3"):
        store.start_session(session)
        episodic.record_turn("user", "run the tests", session_id=session)
    Consolidator(store, emb).run()

    promoted = store.recent(limit=1, layer="procedural")[0]
    stated = ProceduralMemory(store, emb).add_preference("stated directly")
    assert promoted.confidence < store.get(stated).confidence  # type: ignore[union-attr]


def test_consolidation_is_logged(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """§4.2 forbids silent memory mutation."""
    store, emb = parts
    for _ in range(2):
        store.add(layer="semantic", kind="fact", content="dupe", embedding=emb.embed("dupe"))

    consolidator = Consolidator(store, emb)
    consolidator.run()
    history = consolidator.history()
    assert any(entry["action"] == "dedupe" for entry in history)


def test_summarize_needs_enough_turns(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    store.start_session("s1")
    for i in range(3):
        episodic.record_turn("user", f"turn {i}", session_id="s1")
    assert Consolidator(store, emb).summarize_session("s1") is None


def test_summarize_supersedes_the_turns(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """The summary is lossy, so the detail must remain recoverable."""
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    store.start_session("s1")
    for i in range(25):
        episodic.record_turn("user", f"turn {i}", session_id="s1")

    summary_id = Consolidator(store, emb).summarize_session("s1")
    assert summary_id is not None
    assert store.count(layer="episodic") == 0
    assert store.count(layer="episodic", include_superseded=True) == 25


def test_turn_for_an_unopened_session_is_still_stored(
    parts: tuple[MemoryStore, HashEmbedder],
) -> None:
    """Regression: the sessions foreign key rejected turns when start_session was
    never called, and MemoryService.remember_turn swallows write failures — so turns
    disappeared silently, the worst failure a memory system can have."""
    store, emb = parts
    mid = EpisodicMemory(store, emb).record_turn("user", "orphan", session_id="never-opened")
    assert store.get(mid) is not None


def test_promotion_survives_dedupe(parts: tuple[MemoryStore, HashEmbedder]) -> None:
    """Regression: dedupe ran first and merged the repeated phrasings that promotion
    counts, so a pattern could never reach the threshold that promotes it."""
    store, emb = parts
    episodic = EpisodicMemory(store, emb)
    for session in ("s1", "s2", "s3", "s4"):
        store.start_session(session)
        episodic.record_turn("user", "deploy to staging", session_id=session)

    report = Consolidator(store, emb).run()
    assert report.promoted == 1
    assert report.deduped > 0  # dedupe still ran, just after promotion
