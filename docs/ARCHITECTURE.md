# Architecture

**What exists today, at the end of Phase 7.** The aspirational full tree lives in
`docs/BRIEF.md` §4; this file describes only what is built, so it can be trusted.

## Current state

```
arc/
├── __main__.py      CLI — doctor, probe, kill, version, model, chat
├── config.py        layered YAML + env configuration
├── errors.py        exception hierarchy
├── hardware.py      probe → sizing recommendation → hardware.json
├── log.py           structured JSONL logging
├── paths.py         canonical filesystem layout
├── audit/
│   ├── logger.py    append-only action log
│   └── killswitch.py PID registry + SIGKILL
├── model/           ── THE SWAPPABLE BRAIN ──
│   ├── base.py      LanguageModel ABC — five members, deliberately narrow
│   ├── registry.py  models.yaml → typed entries, licence enforced
│   ├── router.py    backend selection from hardware.json
│   ├── manager.py   pull / status / use / remove
│   ├── mlx_backend.py   Apple Silicon fast path
│   └── llamacpp.py      portable GGUF (CPU/Metal/CUDA)
├── agent/           ── THE LOOP ──
│   ├── loop.py      perceive → plan → act → observe → store
│   ├── executor.py  tool dispatch; errors become observations
│   ├── parser.py    tolerant ReAct parsing (§4.1)
│   └── journal.py   crash recovery and resumable tasks
├── memory/          ── THE MEMORY CACHE ──
│   ├── store.py     SQLite + sqlite-vec, one portable file
│   ├── episodic.py semantic.py procedural.py
│   ├── retrieval.py hybrid: vector + BM25 + graph + recency
│   ├── working.py   context-window budget
│   └── consolidation.py
├── tools/           32 tools: filesystem, shell, code, web, screen, input
├── web/             fetch, extract, search, research, deep research
├── vision/          capture, accessibility tree, OCR
├── control/         mouse/keyboard with a visible indicator
├── interface/
│   ├── chat.py      streaming REPL
│   └── server.py    local API, keeps the model warm
└── platform/
    ├── base.py      Platform ABC + HardwareInfo
    ├── macos.py     implemented
    ├── windows.py   stub (Phase 8)
    └── linux.py     stub

config/    default.yaml, models.yaml, policy.yaml, training.yaml
tests/     587 tests
```

Not built yet: a real `platform/windows.py` (Phase 8), and `training/` (Track B).

## Measured costs

Profiled in Phase 7 rather than guessed at. Everything is fast except two things:

| Operation | Cost |
|---|---|
| CLI startup | 0.05–0.11s |
| config load | 0.015s |
| memory recall | 0.007s |
| embed batch of 32 | 0.024s |
| accessibility tree | 0.110s |
| screen capture | 0.184s |
| **OCR** | **0.72s** |
| **model load** | **1.98s** |

Model load is why `arc serve` exists: it holds the model resident, and the CLI uses it
when running. Measured 11.5s cold versus 8.7s warm for the same task.

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

### 2. `arc/model/` — the swappable brain

`LanguageModel` is the narrowest interface a from-scratch model could plausibly satisfy: five
members, and a test asserts the exact set so it cannot widen by accident (ADR-011). Everything
above it — the REPL now, the agent loop in Phase 4 — is written against this and nothing else.

```
caller → router.load_model(config, role)
              ├─ registry.resolve()      which model? (models.yaml + ~/.arc/config.yaml)
              ├─ choose_backend()        which backend? (hardware.json + config)
              └─ lazy import             MLXModel | LlamaCppModel
```

Backends are imported lazily and only inside `router.load_model`. That is what lets
`arc model list` run on a machine with no backend installed, and it is what will keep
`import arc.model` working on Windows where MLX cannot exist.

Model capabilities that genuinely vary are *reported*, not assumed. `ModelCapabilities` defaults
everything to False, so a backend that declares nothing gets the safe path — this is the hook
Phase 4's executor uses to fall back from native tool calling to prompted ReAct.

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
