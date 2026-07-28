# Architecture

**What exists today, at the end of Phase 1.** The aspirational full tree lives in
`docs/BRIEF.md` §4; this file describes only what is built, so it can be trusted.

## Current state

```
arc/
├── __main__.py      CLI entry point — doctor, probe, kill, version
├── config.py        layered YAML + env configuration
├── errors.py        exception hierarchy
├── hardware.py      probe → sizing recommendation → hardware.json
├── log.py           structured JSONL logging
├── paths.py         canonical filesystem layout
├── audit/
│   ├── logger.py    append-only action log
│   └── killswitch.py PID registry + SIGKILL
└── platform/
    ├── base.py      Platform ABC + HardwareInfo
    ├── macos.py     implemented
    ├── windows.py   stub (Phase 8)
    └── linux.py     stub

config/    default.yaml, models.yaml, policy.yaml, training.yaml
tests/     109 tests
```

Not built yet: `model/` (Phase 2), `memory/` (Phase 3), `tools/` and `agent/` (Phase 4),
`vision/` (Phase 6), `interface/` (Phase 7), `training/` (Track B).

## The two load-bearing abstractions

Everything else in the codebase is allowed to be simple. These two are not, because
retrofitting either one later would be a rewrite.

### 1. `arc/platform/` — the OS boundary

A Windows move is planned (§2). The rule: **business logic never calls an OS-specific API.**
It calls `get_platform()`, which returns a `Platform` implementation chosen at runtime.

```
caller → arc.platform.get_platform() → MacOSPlatform | WindowsPlatform | LinuxPlatform
```

Imports are deferred *inside* each branch of the factory, so `macos.py` is never imported on
Windows and each implementation stays free to use OS-specific module-level imports.

This is already paying off: `KillSwitch` calls `platform.kill_process_tree()` rather than
`os.kill`, so Windows swapping SIGKILL for `taskkill /T /F` touches exactly one file.

`HardwareInfo` lives in `platform/base.py`, not `hardware.py`, because it is the *output* of a
platform call. That keeps the dependency pointing one way (`hardware → platform`) instead of
forming a cycle.

### 2. `arc/model/` — the swappable brain (Phase 2)

Not built yet, but every Phase 1 decision was made with it in mind. `LanguageModel` (§4.1) will
be the narrowest interface that a from-scratch model could plausibly satisfy, and the backend
gets chosen from `hardware.json` rather than hardcoded.

## Data flow at startup

```
arc <command>
   ├─ Config.load()           config/*.yaml → ~/.arc/config.yaml → ARC_* env
   ├─ log.setup()             JSONL to ~/.arc/logs/, terse text to stderr
   ├─ AuditLogger()           append-only JSONL to ~/.arc/audit/
   └─ hardware.refresh()      probe → recommend → atomic write hardware.json
```

**`hardware.json` is the single source of truth for sizing.** Model size, quantization, and
later batch size all read measured facts from that file rather than assuming anything about the
machine. It is re-probed on startup rather than trusted from cache — it is cheap, and machines
genuinely change.

## Two roots, deliberately separate

| Root | What | Committed? |
|---|---|---|
| `~/arc/` | code, config defaults, docs | yes |
| `~/.arc/` | hardware.json, logs, audit, models, memory | never |

`ARC_HOME` relocates the runtime root, which is what makes the whole tree testable without
touching the real one. `arc/paths.py` reads it on every call rather than caching at import, so
a fixture can redirect it after the module is loaded.

## Conventions

- **Config over constants.** A number that influences behaviour lives in `config/`, not in a
  literal. `arc/hardware.py` is the current exception — its sizing table encodes §2 of the
  brief directly and should move to config when a second consumer needs it.
- **Fail loudly, recover gracefully.** `ArcError` subclasses distinguish what the agent loop can
  retry (`ToolError`, Phase 4) from what means the install is broken (`ConfigError`). Nothing
  swallows an exception silently.
- **Atomic writes.** Anything durable is written to a temp file and `os.replace`d, so a crash
  cannot leave a truncated file that downstream code parses as authoritative. Required again
  for training checkpoints (§6.2).
- **Structured logging**, never `print` — except in `__main__.py`, which *is* the user
  interface.
- **The audit log is append-only and never truncated.** A log with silent gaps is worse than no
  log, because it invites false confidence while debugging.
