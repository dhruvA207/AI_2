# Backing up `~/.arc/`

ARC keeps every piece of state it accumulates in one directory. That is deliberate
(§4.2): your memory, your audit trail, and your model weights are one thing you can
copy, not several things you have to reassemble.

This is not a migration guide — ARC runs on this Mac and is staying here
(`DECISIONS.md` ADR-021). It exists because a machine can fail without being replaced,
and because the memory database is the one part of ARC that cannot be rebuilt from the
repository.

## What is in there

```
~/.arc/
├── memory.db            ← the only irreplaceable thing
├── config.yaml          ← machine-local overrides (which model is active)
├── audit/*.jsonl        ← append-only record of everything ARC did
├── tasks/*.jsonl        ← task journals, used by `arc task resume`
├── logs/*.jsonl         ← operational logs; disposable
├── models/              ← downloaded weights; large, and re-downloadable
├── screenshots/         ← captures; disposable
└── hardware.json        ← re-probed on demand; disposable
```

**Back up `memory.db`, `config.yaml`, `audit/`, and `tasks/`.** Skip `models/` — it is
several gigabytes and `arc model pull` will fetch it again. Skip `logs/`,
`screenshots/`, and `hardware.json`.

## Backing up

The database uses WAL mode, so copying `memory.db` while ARC is running can capture it
mid-transaction. SQLite's own backup command handles that correctly:

```bash
mkdir -p ~/arc-backup
sqlite3 ~/.arc/memory.db ".backup '$HOME/arc-backup/memory.db'"
cp ~/.arc/config.yaml ~/arc-backup/ 2>/dev/null
cp -R ~/.arc/audit ~/.arc/tasks ~/arc-backup/
```

Or stop ARC first (`arc-kill`, and quit any `arc serve`) and copy the directory
directly. Either works; the first does not require stopping anything.

## Restoring

```bash
arc-kill                       # nothing should be holding the database
pkill -f "arc serve"

cp ~/arc-backup/memory.db ~/.arc/memory.db
cp ~/arc-backup/config.yaml ~/.arc/ 2>/dev/null
cp -R ~/arc-backup/audit ~/arc-backup/tasks ~/.arc/

arc doctor                     # confirms the tree is readable
arc memory stats               # confirms the memories came back
arc model pull qwen3-4b-instruct   # weights were deliberately not backed up
```

**Do not copy `hardware.json` from a backup.** It describes the machine as it was, and
everything downstream — model size, quantization, memory headroom — sizes off it. Let
`arc doctor` or `arc probe` write a fresh one.

## Verifying a backup is good

A backup you have not restored is a hypothesis. Check it against a scratch copy rather
than your live data:

```bash
ARC_HOME=/tmp/arc-restore-test arc doctor
mkdir -p /tmp/arc-restore-test
cp ~/arc-backup/memory.db /tmp/arc-restore-test/
ARC_HOME=/tmp/arc-restore-test arc memory stats
```

`ARC_HOME` relocates the entire runtime tree, which is what makes this safe to try.

## What a corrupted database looks like

`arc memory stats` failing with a disk-image error. SQLite can often recover it:

```bash
sqlite3 ~/.arc/memory.db "PRAGMA integrity_check;"
sqlite3 ~/.arc/memory.db ".recover" | sqlite3 ~/.arc/memory-recovered.db
```

If the vector index is the damaged part, the memories themselves usually survive —
`arc memory export --output memories.json` will still dump them even when search is
broken, because export reads the table directly rather than through the index.
