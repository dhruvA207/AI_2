# Decisions

Architecture decision record: what we chose, what we rejected, and why (§7). Newest last.

---

## ADR-001 — Layered config: files, then machine-local, then environment

**Decision.** `Config.load()` merges `config/*.yaml`, then `~/.arc/config.yaml`, then `ARC_*`
environment variables, later winning. `default.yaml` merges at the root; every other file
merges under a key named after itself, so `policy.yaml` is reachable as `policy.*`.

**Why.** A key's path should be guessable from the file it lives in. Machine-local overrides
need somewhere to live that is never committed — that is `~/.arc/config.yaml`. Environment
overrides use `__` as the separator so single underscores stay usable inside key names, which
they frequently are (`dry_run`, `max_steps`).

**Rejected.** A single flat config file (no override layer). Pydantic Settings (a dependency
for something ~150 lines of stdlib does).

**Consequence.** Lists *replace* rather than concatenate on merge. Appending would make it
impossible to remove a default entry, which is the more common need.

---

## ADR-002 — `hardware.json` is the single source of truth for sizing

**Decision.** A probe writes `~/.arc/hardware.json` at startup. Everything downstream — model
size, quantization, later batch size — reads that file rather than probing or assuming.

**Why.** §2 of the brief is explicit. The alternative is sizing assumptions scattered across
modules, each of which becomes wrong on a different machine.

**Consequence.** `HardwareInfo` carries a `schema_version` so a stale file from an older ARC is
detected rather than silently misread. Currently v2.

---

## ADR-003 — The `Platform` ABC is the only OS boundary

**Decision.** All OS-specific code lives behind `arc/platform/`. Business logic calls
`get_platform()` and never imports `macos` or `windows` directly. Implementations are imported
lazily inside each branch of the factory.

**Why.** A Windows move is planned (§2). This rule is what makes that a port rather than a
rewrite. Lazy imports mean each implementation can use OS-specific module-level imports without
breaking the others.

**Already paying off.** `KillSwitch` calls `platform.kill_process_tree()`, so Windows swapping
SIGKILL for `taskkill /T /F` touches one file.

**Consequence.** `platform_name()` reads `sys.platform` through a local variable, because mypy
statically narrows `sys.platform` on a darwin checkout and would flag the Windows and Linux
branches as unreachable — the branches that most need to survive.

---

## ADR-004 — The kill switch runs from a separate process and uses SIGKILL

**Decision.** Every ARC process writes a PID file to `~/.arc/run/`. `arc kill` reads those files
from its own process and SIGKILLs the trees. Nothing in the kill path touches the agent's state,
event loop, or memory.

**Why.** §0.3: an agent with mouse and keyboard control that hits a loop can make the machine
unusable, and at that point you may not be able to interact with a terminal reliably. Stopping
ARC must not depend on ARC being healthy. A graceful shutdown asks a process to cooperate; a
wedged process cannot.

**Consequence.** SIGKILL cannot be caught, so PID files outlive their processes. `reap_stale()`
exists to clean up, and runs before every kill so the report counts live processes rather than
corpses.

---

## ADR-005 — No `psutil`; shell out to OS tools instead

**Decision.** The hardware probe uses `subprocess` against `sysctl`, `system_profiler`, and
`sw_vers` rather than taking a `psutil` dependency.

**Why.** §7 says to ask whether fifty lines of our own code would do. Here they do — the parsing
is about that long. `psutil` is also BSD-3-Clause rather than the Apache-2.0/MIT §0.1 restricts
us to, so avoiding it keeps the ledger clean.

**Cost.** Each platform implements its own probe. Accepted: the probe is inherently
platform-specific, so `psutil` would have hidden the difference rather than removed it.

---

## ADR-006 — Cooling is part of sizing, not just memory

**Decision.** `HardwareInfo` carries `chassis` and `fanless`. `recommend_model()` emits a
warning on a fanless machine advising the next size down for sustained use.

**Why.** The dev machine is a fanless MacBook Air M3. Memory decides what *fits*; cooling
decides what stays fast. A fanless chassis benchmarks fine for a few minutes and then loses a
third of its throughput, so a table lookup on RAM alone quietly over-promises.

**How.** `machine_name` from `system_profiler` ("MacBook Air"), not a lookup table over
`hw.model` identifiers ("Mac15,12"). Apple stopped encoding the product line in those
identifiers, so a table would need editing for every new machine and would silently mis-report
an unknown one.

**Consequence.** `fanless` is `bool | None`. `None` means "could not determine" and must not be
reported as though it were a known hardware limit.

---

## ADR-007 — ARC runs on the Air; the Windows laptop is a training appliance

**Decision.** ARC itself runs on the MacBook Air M3 (16 GB unified). The Windows laptop
(i9-13900HS, 8 GB VRAM) is used only for Track B training, later. `arc/platform/windows.py`
stays a stub and full Windows support stays in Phase 8.

**Why.**

- Nothing between here and a working assistant is compute-bound. Phases 2–4 are interactive
  inference and I/O.
- For *inference* the Air is the better machine: ~11.5 GB usable unified memory versus ~6.5 GB
  usable VRAM after Windows' compositor reservation. A 14B model at 4-bit fits the Air and does
  not fit that GPU. The Windows box wins only at sustained training throughput (6–10×).
- §4.3's screen-reading and app-control story assumes ARC runs where work actually happens.
- Training is a batch job that emits a weights file; it does not need to live where you work.

**Consequence.** Model artifacts flow one way: train on Windows → convert to GGUF or MLX → copy
to the Air. `~/.arc/` never leaves the Air, preserving §4.2's "one backup-able artifact"
property. **VRAM, not system RAM, governs sizing on a discrete-GPU machine** —
`HardwareInfo.model_memory_gb` already branches on `unified_memory`, but that branch has not
run on real hardware yet.

---

## ADR-008 — Track B abandons from-scratch pretraining at scale

**Decision.** `docs/BRIEF.md` §6 was written around renting an 8×H100 node for 24 hours. There
is no rented compute. Track B becomes QLoRA fine-tuning of an Apache-2.0 Qwen3 base, plus an
optional 5–10M parameter from-scratch model on TinyStories for the curriculum.

**Why.** The §0.1 licensing goal never required from-scratch training — a fine-tune of
Apache-2.0 weights is fully ours, with only NOTICE attribution travelling to the derivative.
From-scratch was always about learning, and a small local model delivers most of that.

**Hard constraint.** No training run exceeds about a week, and the deliverable run is measured
in hours (a ~20K-example QLoRA SFT is roughly 5 hours). Using a pretrained base is precisely
what buys this; nothing may quietly erode it.

**Dropped from §6.3.** Dollar budgets, `hourly_rate_usd`, compute-credit thresholds, cloud
checkpoint upload, `supervisor.py` instance re-provisioning, `RENTING_GPUS.md`.

**Kept.** Checkpoint/resume with bit-exact state, and `arc train status/pause/resume`. The
justification changes from spot preemption to "it throttled overnight" and "I need my laptop
back"; the requirement is identical.

---

## ADR-009 — CLI invocations are audited, not just agent tool calls

**Decision.** `arc/__main__.py` appends a record for every invocation, including its exit code
and dry-run status.

**Why.** §0.3 asks for a log that answers "what happened at 2am." That question includes
commands run by hand, not only actions the agent took autonomously.

**Consequence.** Audit failures are suppressed in the CLI path specifically — refusing to let
`arc doctor` start because `~/.arc` is read-only would hide the exact diagnosis being asked
for. Inside an agent run (Phase 4) an audit failure stays fatal.
