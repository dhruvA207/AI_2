# Handoff — end of the build session, 2026-07-30

Written for whoever picks this up next, including a future me with no memory of the
session. **Next task: build the ARC UI.** Nothing about it has been designed or
discussed yet — start there fresh.

---

## What ARC is

A local-first personal assistant, spec'd in `docs/BRIEF.md`. It runs entirely on
Dhruv's MacBook Air: a local language model, memory that survives restarts, unrestricted
access to the machine, screen reading and control, and web research.

**All 8 phases of §5 are complete.** 587 tests, ruff + `ruff format` + `mypy --strict`
clean, ~21 commits on `main`, **nothing pushed yet**.

## Read these first

| File | Why |
|---|---|
| `docs/BRIEF.md` | The spec. Dhruv wrote it; it is the source of truth. §6 is shelved. |
| `docs/DECISIONS.md` | 23 ADRs. **Read this before changing anything architectural** — most surprising choices are deliberate and explained. |
| `docs/ARCHITECTURE.md` | What exists, plus measured performance costs. |
| `docs/DEPENDENCIES.md` | Every dependency, its licence, and why it earns its place. |
| `docs/BACKUP.md` | How to back up `~/.arc/`. Verified end to end. |

---

## Hard rules, learned the hard way

**1. Commits are Dhruv's alone.** No `Co-Authored-By: Claude` trailer, no "Generated
with Claude Code" footer. Thirteen commits were rewritten to strip it. Author is
`dhruvA207 <dhruvagrawal.v@gmail.com>`.

**2. Every dependency is Apache-2.0 or MIT.** Verify from wheel metadata or the live
model card — not from memory. This is not ceremony: a licence audit killed the intended
web stack when `trafilatura → courlan → tld` turned out to include GPL-2.0, and
`trafilatura` itself had been GPLv3+ until v1.8.0. Two MPL-2.0 packages (`certifi`,
`pathspec`) are in the tree via the MLX stack and dev tooling; recorded, not hidden.

**3. ARC runs on this Mac and is not moving.** ADR-021. Phase 8's "Windows readiness"
is cancelled. `platform/windows.py` and `linux.py` stay as stubs — the abstraction earns
its keep by letting the core import without any Apple framework, which is verified in
the test suite, not assumed.

**4. Track B is shelved.** ADR-023. No training, no fine-tuning, no GPU rental. Do not
re-propose it. `BRIEF.md` §6, `config/training.yaml`, and `ML_CURRICULUM.md` are kept as
dead records of how the scope moved.

**5. §0.3 access is deliberate.** Unrestricted filesystem, arbitrary shell, no
permission prompts, no deny-list. Dhruv specified this explicitly and said not to
negotiate it. The safeguards he asked for instead — and which must stay — are the audit
log, `arc-kill`, and `--dry-run`.

---

## The one thing that will mislead you

**The model is the weak link, not the code.** Qwen3-4B at 4-bit:

- Chained `list_directory → read_file → read_file → write_file` correctly, then summed
  2 + 1 and wrote **4**.
- Answered 17 × 23 as **491**.

When output looks wrong, check whether the tools returned the right thing before
suspecting a bug:

```bash
arc task show <id>     # exactly what each tool returned, step by step
```

`arc model use qwen3-8b` trades sustained speed for reasoning — both fit in 11.5 GB, but
the Air is fanless and the 8B throttles after 10–20 minutes (ADR-010).

---

## Current state, verified this session

```
gate            ruff, ruff format, mypy --strict all clean; 587 tests pass
processes       no stray overlay or serve processes; no stale pid files
stale endpoint  server.json self-cleans when the process is gone — verified
audit log       126 records across 3 files, 0 malformed
task journals   resume linkage correct; only genuinely unfinished tasks marked
memory          19 memories (11 test artifacts removed), persists across processes
dry-run         verified it creates no files
control         input correctly refused without an active session
kill switch     arc-kill works standalone, no config needed
```

## Commands

```bash
ARC                         # launch: the app window plus the loopback server
ARC <anything>              # same as `arc <anything>`; ~/.local/bin/ARC -> bin/ARC
arc doctor                  # environment + macOS permission grants
arc chat                    # REPL; /memory /recall /why /model /tokens
arc do "<task>"             # multi-step work; --dry-run, --no-server
arc research "<q>"          # shallow; --deep for corroborated Research Mode
arc memory stats|search|add|forget|export|consolidate
arc model list|pull|use|remove
arc control status|release|demo     # demo shows the blue glow for 5s
arc task list|show|resume
arc tools
arc serve                   # keeps the model warm; 11.5s → 8.7s per task
arc-kill                    # standalone; works when arc itself does not
```

`ARC` is a self-locating wrapper in `bin/`, symlinked onto PATH, so it works from any
directory and through the symlink. Bare `ARC` runs `arc ui`; anything else passes
straight through. macOS filesystems are case-insensitive, so `arc` reaches the same
script — either spelling works, and neither needs the venv activated.

---

## Measured performance

Profiled rather than guessed (§5). Everything is fast except two things:

| Operation | Cost |
|---|---|
| CLI startup | 0.05–0.11 s |
| memory recall | 0.007 s |
| embed batch of 32 | 0.024 s |
| accessibility tree | 0.110 s |
| screen capture | 0.184 s |
| **OCR** | **0.72 s** |
| **model load** | **1.98 s** |

Model load is why `arc serve` exists. Deep research is 1–3 minutes because claim
extraction is one model call per page — that is inherent at ~14 tok/s, not a bug.

---

## Bugs found this session, and what they teach

Twenty-plus, mostly found by *using* the thing rather than by tests passing. The ones
worth knowing because the same mistake is easy to repeat:

**Silent-failure class** — these all returned plausible results while being wrong:

- `bool` is a subclass of `int`, so every boolean tool parameter was typed `"integer"`
  and the model would send `1`/`0` for flags.
- `_coerce_arguments` read `inspect.Parameter.annotation`, which returns *strings* under
  `from __future__ import annotations` — used by every tool module. Coercion was
  completely dead in production while passing any test written without that import.
- The docstring parser broke at the blank line before `Args:`, the standard format, so
  every tool shipped with undocumented parameters.
- mdBook puts `class="sidebar-visible"` on `<html>`; substring-matching "sidebar"
  flagged whole documents as boilerplate. `doc.rust-lang.org` extracted **zero words**
  while blog spam sailed through.
- pyobjc's `AXValueGetValue` *returns* `(ok, value)` rather than filling a struct. Every
  accessibility element's frame came back `None` — 376 elements, zero usable.
- **Twelve config keys were declared and never read**, values hardcoded in the modules.
  Editing them did nothing. Worse than a missing key, which at least fails loudly.

**Interaction class** — only visible when subsystems met:

- Salience multiplied fused retrieval scores directly, and layer defaults (0.6–1.5)
  swamped RRF's ~3× range. A query containing "router bug" returned a preference about
  tidying Downloads.
- Consolidation ran dedupe *before* promotion, destroying the repeated phrasings that
  promotion counts. Promotion silently never fired.
- Memories rendered as `- fact [episodic, 2026-07-30]` got copied by the model: asked
  for "pong" it replied `pong [episodic, 2026-07-30]`, which was then stored and
  recalled, producing two markers next turn. **Instructing it not to did not work — the
  format had to change.**

**The lesson:** write adversarial probes, not just tests. Most of these passed a green
suite.

---

## Things left undone, honestly

- **No UI.** That is the next task and nothing has been designed.
- **`arc chat` does not use the warm server.** Only `arc do` does. Straightforward to
  add.
- **Deep research is slow** (1–3 min). Inherent to local inference.
- **Bing search is unreliable** and DuckDuckGo is primary (ADR-020 addendum). Dhruv
  wanted Bing/Google; they return ads and JavaScript shells to non-browser clients. A
  keyed search API is the only real fix, and he has not asked for one.
- **`robots.txt` compliance is disabled** in `config/default.yaml` at his explicit
  request. It logs a warning at every startup.
- **The VLM is not integrated.** §4.3's ladder stops at OCR. The accessibility tree plus
  OCR has been sufficient so far; a VLM would not co-reside with the 4B in 11.5 GB
  anyway without eviction.
- **`arc/model/custom.py` is not built** — it was Track B's socket.

---

## Where things live

```
arc/
├── __main__.py      CLI — every command dispatches from here
├── config.py        layered YAML + env; every key IS read (ADR-022)
├── hardware.py      probe → sizing; reads config, does not hardcode
├── audit/           append-only log + standalone kill switch
├── model/           base.py is the swappable brain — 5 members, keep it narrow
├── memory/          store, 3 layers, hybrid retrieval, consolidation
├── agent/           loop, executor, tolerant parser, journal
├── tools/           32 tools; @tool decorator derives schemas from type hints
├── web/             fetch, extract, search, research, deep
├── vision/          capture, accessibility tree, OCR
├── control/         session, blue-glow overlay, input
├── interface/       chat REPL, HTTP server
└── platform/        macos.py real; windows/linux stubs
```

Runtime state is all under `~/.arc/` — `memory.db` is the only irreplaceable part.

---

## If you build the UI next

Nothing has been decided. Worth knowing before you start:

- **`arc serve` already exists** and exposes `/health`, `/chat`, `/do`,
  `/memory/search`, `/memory/add`, `/tools` over loopback-only HTTP. A UI should
  probably talk to it rather than importing `arc` directly — that is what it was built
  for (§4 lists `interface/server.py` as "a local HTTP/WebSocket API for a future GUI").
- **The bind address is deliberately not configurable.** ARC has unrestricted machine
  access; an endpoint reaching it must not be reachable from the network. There is a
  test asserting no config key exists for it. Do not add one.
- **There is no streaming endpoint yet.** `/chat` returns a complete response. The model
  interface supports streaming (`LanguageModel.stream`), so adding SSE or WebSocket is
  the natural first step for a UI that should feel responsive at ~14 tok/s.
- **ARC's accent colour is `#4A9EFF`**, already used by the control indicator. Dhruv
  asked for that glow to look like claude-in-chrome.
- **Ask him about scope before building.** He has been decisive and has changed
  direction several times when something was not worth doing — the fastest path is to
  lay out options rather than guess.
