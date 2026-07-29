"""Tests for the memory store: schema, persistence, and the search indexes."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from arc.errors import MemoryError as ArcMemoryError
from arc.memory import HashEmbedder, MemoryStore
from arc.memory.store import deserialize_vector, serialize_vector


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    """A small-dimension store, so tests stay fast."""
    return MemoryStore(tmp_path / "m.db", dimension=64)


@pytest.fixture
def embedder() -> HashEmbedder:
    return HashEmbedder(64)


def test_vector_roundtrip() -> None:
    values = [0.5, -0.25, 1.0]
    assert deserialize_vector(serialize_vector(values)) == pytest.approx(values)


def test_add_and_get(store: MemoryStore, embedder: HashEmbedder) -> None:
    mid = store.add(
        layer="semantic", kind="fact", content="hello", embedding=embedder.embed("hello")
    )
    record = store.get(mid)
    assert record is not None
    assert record.content == "hello"
    assert record.layer == "semantic"


def test_empty_content_is_rejected(store: MemoryStore) -> None:
    with pytest.raises(ArcMemoryError, match="empty memory"):
        store.add(layer="semantic", kind="fact", content="   ")


def test_wrong_dimension_is_rejected(store: MemoryStore) -> None:
    """A mismatch must fail loudly rather than produce nonsense distances."""
    with pytest.raises(ArcMemoryError, match="dimensions"):
        store.add(layer="semantic", kind="fact", content="x", embedding=[0.1] * 999)


def test_invalid_layer_is_rejected(store: MemoryStore) -> None:
    """The CHECK constraint is the last line of defence against a typo'd layer."""
    with pytest.raises(sqlite3.IntegrityError):
        store.add(layer="telepathic", kind="fact", content="x")


def test_persists_across_connections(tmp_path: Path, embedder: HashEmbedder) -> None:
    """The whole point: one file that survives the process that wrote it."""
    path = tmp_path / "m.db"
    first = MemoryStore(path, dimension=64)
    first.add(layer="semantic", kind="fact", content="durable", embedding=embedder.embed("durable"))
    first.close()

    second = MemoryStore(path, dimension=64)
    assert second.count() == 1
    assert second.recent()[0].content == "durable"


def test_dimension_mismatch_on_reopen_is_reported(tmp_path: Path) -> None:
    """Re-opening with a different embedder must say so, not silently misbehave."""
    path = tmp_path / "m.db"
    MemoryStore(path, dimension=64).close()
    with pytest.raises(ArcMemoryError, match="384"):
        MemoryStore(path, dimension=384)


def test_fts_index_is_populated_by_trigger(store: MemoryStore, embedder: HashEmbedder) -> None:
    """External-content FTS5 returns stale results without the triggers."""
    store.add(
        layer="episodic", kind="event", content="the router exploded", embedding=embedder.embed("x")
    )
    rows = (
        store.connection()
        .execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'router'")
        .fetchall()
    )
    assert len(rows) == 1


def test_fts_index_follows_updates(store: MemoryStore, embedder: HashEmbedder) -> None:
    mid = store.add(
        layer="episodic", kind="event", content="original text", embedding=embedder.embed("x")
    )
    with store.transaction() as conn:
        conn.execute("UPDATE memories SET content = 'replacement text' WHERE id = ?", (mid,))

    stale = (
        store.connection()
        .execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'original'")
        .fetchall()
    )
    fresh = (
        store.connection()
        .execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'replacement'")
        .fetchall()
    )
    assert stale == []
    assert len(fresh) == 1


def test_fts_index_follows_deletes(store: MemoryStore, embedder: HashEmbedder) -> None:
    mid = store.add(
        layer="episodic", kind="event", content="ephemeral", embedding=embedder.embed("x")
    )
    store.forget(mid)
    rows = (
        store.connection()
        .execute("SELECT rowid FROM memories_fts WHERE memories_fts MATCH 'ephemeral'")
        .fetchall()
    )
    assert rows == []


def test_forget_removes_the_vector_too(store: MemoryStore, embedder: HashEmbedder) -> None:
    """memory_vectors has no foreign key, so an orphan would survive a naive delete."""
    mid = store.add(layer="semantic", kind="fact", content="gone", embedding=embedder.embed("gone"))
    store.forget(mid)
    rows = store.connection().execute("SELECT * FROM memory_vectors").fetchall()
    assert rows == []


def test_forget_reports_whether_it_existed(store: MemoryStore) -> None:
    assert store.forget(999) is False


def test_touch_raises_salience_and_counts(store: MemoryStore, embedder: HashEmbedder) -> None:
    mid = store.add(layer="semantic", kind="fact", content="x", embedding=embedder.embed("x"))
    store.touch([mid])
    record = store.get(mid)
    assert record is not None
    assert record.salience > 1.0
    assert record.access_count == 1


def test_touch_is_capped(store: MemoryStore, embedder: HashEmbedder) -> None:
    """A frequently-hit memory must not grow without bound and crowd out everything."""
    mid = store.add(layer="semantic", kind="fact", content="x", embedding=embedder.embed("x"))
    for _ in range(200):
        store.touch([mid])
    record = store.get(mid)
    assert record is not None
    assert record.salience <= 2.0


def test_touch_with_empty_list_is_a_noop(store: MemoryStore) -> None:
    store.touch([])


def test_supersede_retains_the_original(store: MemoryStore, embedder: HashEmbedder) -> None:
    """BRIEF §4.2 forbids silent mutation: superseded memories stay readable."""
    old = store.add(layer="semantic", kind="fact", content="old", embedding=embedder.embed("old"))
    new = store.add(layer="semantic", kind="fact", content="new", embedding=embedder.embed("new"))
    store.supersede([old], new)

    record = store.get(old)
    assert record is not None
    assert record.superseded_by == new
    assert record.content == "old"


def test_superseded_memories_are_excluded_from_live_queries(
    store: MemoryStore, embedder: HashEmbedder
) -> None:
    old = store.add(layer="semantic", kind="fact", content="old", embedding=embedder.embed("old"))
    new = store.add(layer="semantic", kind="fact", content="new", embedding=embedder.embed("new"))
    store.supersede([old], new)

    assert store.count() == 1
    assert store.count(include_superseded=True) == 2
    assert [r.id for r in store.recent()] == [new]


def test_recent_filters_by_layer(store: MemoryStore, embedder: HashEmbedder) -> None:
    store.add(layer="episodic", kind="event", content="e", embedding=embedder.embed("e"))
    store.add(layer="semantic", kind="fact", content="s", embedding=embedder.embed("s"))
    assert len(store.recent(layer="semantic")) == 1


def test_transaction_rolls_back_on_failure(store: MemoryStore) -> None:
    with pytest.raises(RuntimeError), store.transaction() as conn:
        conn.execute(
            "INSERT INTO memories (layer, kind, content, occurred_at, created_at) "
            "VALUES ('semantic','fact','doomed','x','x')"
        )
        raise RuntimeError("boom")
    assert store.count() == 0


def test_session_lifecycle(store: MemoryStore, embedder: HashEmbedder) -> None:
    store.start_session("s1", model="test")
    store.add(
        layer="episodic",
        kind="turn:user",
        content="hi",
        session_id="s1",
        embedding=embedder.embed("hi"),
    )
    store.end_session("s1")
    row = store.connection().execute("SELECT * FROM sessions WHERE id = 's1'").fetchone()
    assert row["turn_count"] == 1
    assert row["ended_at"] is not None


def test_stats_reports_layers(store: MemoryStore, embedder: HashEmbedder) -> None:
    store.add(layer="episodic", kind="event", content="a", embedding=embedder.embed("a"))
    store.add(layer="semantic", kind="fact", content="b", embedding=embedder.embed("b"))
    stats = store.stats()
    assert stats["live_memories"] == 2
    assert stats["by_layer"] == {"episodic": 1, "semantic": 1}
    assert stats["embedded"] == 2


def test_iter_all_streams_every_memory(store: MemoryStore, embedder: HashEmbedder) -> None:
    for i in range(5):
        store.add(
            layer="episodic", kind="event", content=f"m{i}", embedding=embedder.embed(f"m{i}")
        )
    assert len(list(store.iter_all())) == 5


def test_metadata_roundtrips_as_json(store: MemoryStore, embedder: HashEmbedder) -> None:
    mid = store.add(
        layer="semantic",
        kind="fact",
        content="x",
        embedding=embedder.embed("x"),
        metadata={"nested": {"a": 1}, "list": [1, 2]},
    )
    record = store.get(mid)
    assert record is not None
    assert record.metadata == {"nested": {"a": 1}, "list": [1, 2]}


def test_hash_embedder_is_stable_across_calls() -> None:
    """Regression: builtin hash() is salted per process, so vectors would not match."""
    a = HashEmbedder(64).embed("stable text")
    b = HashEmbedder(64).embed("stable text")
    assert a == b


def test_hash_embedder_is_normalized() -> None:
    vector = HashEmbedder(64).embed("some words here")
    assert sum(x * x for x in vector) == pytest.approx(1.0, abs=1e-5)


def test_hash_embedder_handles_empty_text() -> None:
    """A zero vector must not produce a divide-by-zero."""
    assert HashEmbedder(64).embed("") == [0.0] * 64
