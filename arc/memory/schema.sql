-- ARC memory schema.
--
-- One SQLite file holds all three memory layers plus the search indexes. The brief
-- (§4.2) is explicit that this must stay a single portable file with no server, so
-- everything here — vectors, full-text, and the entity graph — lives in the same
-- database rather than in a sidecar service.
--
-- Vector tables are created separately in store.py, because vec0 virtual tables need
-- the embedding dimension baked into their DDL and that comes from the embedder.

PRAGMA journal_mode = WAL;      -- Readers never block the writer; consolidation runs
                                -- in the background while chat is still reading.
PRAGMA foreign_keys = ON;
PRAGMA synchronous = NORMAL;    -- WAL makes this durable enough short of power loss.

-- ── Core memory table ────────────────────────────────────────────────────────────
-- All three layers share one table rather than three. They differ in *lifecycle*, not
-- in shape, and a single table means retrieval can rank across layers in one query
-- instead of merging three result sets with incomparable scores.
CREATE TABLE IF NOT EXISTS memories (
    id            INTEGER PRIMARY KEY,
    layer         TEXT    NOT NULL CHECK (layer IN ('episodic', 'semantic', 'procedural')),
    kind          TEXT    NOT NULL,   -- turn, event, fact, preference, workflow, ...
    content       TEXT    NOT NULL,

    -- When the memory refers to, which is not always when it was written: a fact
    -- learned today may describe last Tuesday.
    occurred_at   TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    accessed_at   TEXT,
    access_count  INTEGER NOT NULL DEFAULT 0,

    -- Salience decays with disuse and is boosted by retrieval; consolidation prunes
    -- the bottom. Kept explicit rather than derived so decay is auditable.
    salience      REAL    NOT NULL DEFAULT 1.0,

    -- How much to trust this. Web-sourced facts arrive below 1.0 and time-sensitive
    -- categories get re-verified rather than trusted forever (§4.4).
    confidence    REAL    NOT NULL DEFAULT 1.0,

    source        TEXT,               -- 'chat', 'web', 'consolidation', 'user', ...
    source_url    TEXT,               -- provenance for anything learned from the web
    session_id    TEXT,
    metadata      TEXT    NOT NULL DEFAULT '{}',   -- JSON

    -- Set when consolidation supersedes this memory with a summary or a merge.
    -- Superseded rows are retained, not deleted: §4.2 forbids silent memory mutation,
    -- and being able to see what was replaced is the whole point of the audit trail.
    superseded_by INTEGER REFERENCES memories(id) ON DELETE SET NULL,

    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_layer      ON memories(layer);
CREATE INDEX IF NOT EXISTS idx_memories_occurred   ON memories(occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_salience   ON memories(salience DESC);
CREATE INDEX IF NOT EXISTS idx_memories_session    ON memories(session_id);
-- Partial index: almost every query wants live memories only, and this keeps the
-- superseded ones from bloating the scan.
CREATE INDEX IF NOT EXISTS idx_memories_live       ON memories(layer, salience DESC)
    WHERE superseded_by IS NULL;

-- ── Conversation sessions ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  TEXT NOT NULL,
    ended_at    TEXT,
    model       TEXT,
    title       TEXT,               -- written by consolidation from the transcript
    summary     TEXT,               -- compact form, so old sessions cost few tokens
    turn_count  INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);

-- ── Semantic layer: the entity graph ─────────────────────────────────────────────
-- Nodes and edges in SQLite rather than a graph database, per §4.2. Multi-hop
-- traversal is a recursive CTE, which is fast enough at personal-memory scale and
-- costs no extra dependency.
CREATE TABLE IF NOT EXISTS entities (
    id           INTEGER PRIMARY KEY,
    name         TEXT NOT NULL,
    kind         TEXT NOT NULL,     -- person, project, file, tool, concept, ...
    -- Lowercased name, for case-insensitive uniqueness without a functional index.
    normalized   TEXT NOT NULL,
    first_seen   TEXT NOT NULL,
    last_seen    TEXT NOT NULL,
    mention_count INTEGER NOT NULL DEFAULT 1,
    metadata     TEXT NOT NULL DEFAULT '{}',
    -- Identity is the name alone, deliberately NOT (normalized, kind). Keying on both
    -- meant that mentioning "ARC" as a project and then relating it as a concept
    -- created two nodes for one thing and silently cut the graph in half: traversal
    -- from one node could not reach edges attached to the other. Kind is an attribute
    -- that gets refined over time, not part of the key.
    UNIQUE (normalized)
);

CREATE INDEX IF NOT EXISTS idx_entities_normalized ON entities(normalized);

-- Typed, directed relationships. Multi-hop questions ("who is X and how do they
-- relate to Y?") walk these.
CREATE TABLE IF NOT EXISTS relations (
    id          INTEGER PRIMARY KEY,
    subject_id  INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate   TEXT    NOT NULL,
    object_id   INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    confidence  REAL    NOT NULL DEFAULT 1.0,
    created_at  TEXT    NOT NULL,
    memory_id   INTEGER REFERENCES memories(id) ON DELETE CASCADE,
    UNIQUE (subject_id, predicate, object_id)
);

CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object  ON relations(object_id);

-- Which memories mention which entities. This is what lets a query about "Dhruv"
-- pull in memories that never contain that literal string.
CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id INTEGER NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id);

-- ── Keyword search ───────────────────────────────────────────────────────────────
-- FTS5 external-content table: the text lives in `memories`, and this holds only the
-- index. Halves the storage and keeps `memories` the single source of truth.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    content,
    content='memories',
    content_rowid='id',
    tokenize='porter unicode61'
);

-- Triggers keep the index in step. Without these an external-content FTS5 table
-- silently returns stale results, which is a miserable bug to chase.
CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content);
END;

-- ── Consolidation audit ──────────────────────────────────────────────────────────
-- §4.2: "Log everything it changes to the audit log — I don't want silent memory
-- mutation." This is the in-database half; the JSONL audit log gets the same events.
CREATE TABLE IF NOT EXISTS consolidation_log (
    id          INTEGER PRIMARY KEY,
    ran_at      TEXT NOT NULL,
    action      TEXT NOT NULL,     -- dedupe, summarize, decay, promote, prune
    memory_ids  TEXT NOT NULL,     -- JSON array of affected ids
    detail      TEXT NOT NULL DEFAULT '{}',
    result_id   INTEGER REFERENCES memories(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_consolidation_ran ON consolidation_log(ran_at DESC);
