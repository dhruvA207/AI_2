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

---

## ADR-010 — Qwen3-4B is the default, not the 7–8B the sizing table allows

**Decision.** `config/models.yaml` ships Qwen3-4B-Instruct (4-bit, 2.1 GB) as the active chat
model, with Qwen3-8B (4-bit, 4.6 GB) available but not default.

**Why.** `arc doctor` recommends 7–8B on memory grounds and then warns that this chassis is
fanless. Both models fit in 11.5 GB usable; the difference is what happens twenty minutes into
a session. The 4B stays responsive where the 8B throttles, and an assistant you are waiting on
is worse than a slightly weaker one that answers.

**Cost, stated plainly.** The 4B is measurably worse at reasoning. During Phase 2 testing it
answered "17 × 23" as 491 rather than 391. Arithmetic and multi-step reasoning are exactly
where the 8B earns its extra weight, so `arc model use qwen3-8b` is the right move for hard
one-off tasks. Phase 3's memory and Phase 4's tools will matter more for everyday usefulness
than this choice does.

---

## ADR-011 — The `LanguageModel` interface stays narrow on purpose

**Decision.** `arc/model/base.py` exposes five abstract members: `name`, `generate`, `stream`,
`count_tokens`, `context_length`, `capabilities`. Nothing else. A test asserts the exact set,
so widening it is a deliberate act rather than a drift.

**Why.** §4.1 requires that a model trained from scratch can eventually satisfy this interface.
Every capability added here is another thing Track B must implement before its model can drive
the agent. Logit bias, beam search, speculative decoding, and grammar-constrained output were
all considered and left out: none is required to run the agent loop.

**How variation is handled.** Capabilities that genuinely differ between models are *reported*
via `ModelCapabilities` rather than assumed. They default to False, so a backend that declares
nothing gets the prompted-ReAct fallback — which works everywhere — instead of silently
emitting native tool calls that nothing parses.

---

## ADR-012 — Backend selection is explainable, and an override may be wrong

**Decision.** `router.choose_backend()` returns a `BackendChoice` carrying the backend *and the
reason*. Precedence: an explicit `force_backend` wins, then the entry's preferred backend if
this machine has the accelerator, then llama.cpp as the universal fallback.

**Why.** A router that silently picks differently on two machines is hard to debug. `arc model
list` shows the reasoning, including when a choice is a fallback and what it fell back from.

**Deliberate sharp edge.** `force_backend` overrides even onto a machine that cannot support
it. Being able to force a wrong answer is what makes an override useful for debugging; a safe
override that silently declines would be useless.

---

## ADR-013 — Weights live in `~/.arc/models/`, not the Hugging Face cache

**Decision.** `arc model pull` downloads into `~/.arc/models/<key>/` via `snapshot_download`
with an explicit `local_dir`, rather than letting the hub client use `~/.cache/huggingface`.

**Why.** §4.2 wants `~/.arc` to be one portable, backup-able artifact. A model tucked away in
the user's cache directory would not move with it, and the first thing a migrated install would
do is re-download several gigabytes.

**Consequence.** The router prefers the local directory and falls back to the hub repo id, so a
model that was never pulled still loads (downloading on first use) rather than erroring. And
`is_downloaded()` checks for actual weight files rather than directory existence — an
interrupted download leaves a directory behind, and calling that "ready" would send the user to
a confusing load failure instead of telling them to pull again.

---

## ADR-014 — bge-small via ONNX Runtime, not PyTorch

**Decision.** Embeddings come from BAAI/bge-small-en-v1.5 (MIT) run through ONNX Runtime
with the HuggingFace `tokenizers` library. Not sentence-transformers, not PyTorch, not MLX.

**Why.**

- **Cross-platform.** The same ONNX file and the same code run on Apple Silicon, Windows,
  and Linux. MLX would be faster here but exists only on Macs, and Track A is supposed to
  survive the Windows move without a rewrite (§2).
- **Weight.** onnxruntime is ~19 MB and tokenizers ~3 MB. PyTorch is ~2.5 GB — an absurd
  price to embed short strings with a 33M-parameter model (§7).
- **Fully local.** No embedding API calls, per §3.

**Detail that is easy to get wrong.** bge uses the CLS token as the sentence
representation, not mean pooling, and prefixes *queries* (not stored passages) with
"Represent this sentence for searching relevant passages:". Getting either wrong produces
vectors that look fine and retrieve badly — the worst kind of wrong, because nothing errors.

**Measured.** 384 dimensions, unit-normalised, 12 ms per batch of 4 on the M3.

---

## ADR-015 — Consolidation is conservative by default

**Decision.** Dedupe only above 0.97 cosine similarity, decay 1% per day, promote after 3
recurrences across at least 2 sessions, and **pruning is off entirely**. Merged and
summarised memories are superseded, never deleted.

**Why.** Consolidation rewrites memory in the background without being asked. The two
failure modes are not symmetric: being too cautious costs disk space, while being too
aggressive destroys things silently and you only discover it when a question that should
have had an answer doesn't. §4.2 also forbids silent memory mutation outright, which
supersede-don't-delete satisfies directly.

Every threshold is in `config/default.yaml` and meant to be turned up once there is
evidence of what these actually do to a real corpus.

**Ordering matters more than it looks.** Promotion must run *before* dedupe. Promotion's
signal is a phrase recurring across sessions; identical text embeds identically, so it is
the first thing dedupe collapses. Running dedupe first left one live copy of every repeated
request and promotion silently never fired.

---

## ADR-016 — Recalled memories go in the system message

**Decision.** Retrieved memories are rendered into the system prompt, not replayed as
synthetic prior conversation turns.

**Why.** A fabricated exchange is indistinguishable, to the model, from something the user
actually said. It would start attributing its own recollections to them — "you told me
X" when X came from a web page. The system message keeps the provenance boundary intact,
and rendering source URLs and confidence alongside each memory is what lets ARC cite where
a fact came from when asked (§4.4).

**Consequence.** `/why` exposes the retrieval provenance — score and which of the four
strategies found each memory — because "why did it say that?" is the first question when
memory misbehaves, and the answer needs more than the result list.

---

## ADR-017 — Web research uses the standard library, because the obvious stack is GPL-tainted

**Decision.** Fetching, robots.txt, and HTML content extraction are built on
`urllib.request`, `urllib.robotparser`, and `html.parser`. **Zero third-party
dependencies**, which was not the plan.

**Why.** The intended stack was `requests` + `trafilatura`. A licence audit of the
resulting dependency tree — which §0.1 requires and §3 warns must be done rather than
assumed — found:

- **`tld`: MPL-1.1 OR GPL-2.0-only OR LGPL-2.1-or-later.** Pulled in transitively by
  trafilatura → courlan → tld. §0.1 forbids GPL outright. Trafilatura itself is
  Apache-2.0 (and only since v1.8.0 — it was GPLv3+ before, exactly the drift §3 warns
  about), but its tree is not.
- **`certifi`: MPL-2.0**, pulled in by requests.

Rather than hunt for a second-best extraction library, the stdlib turned out to cover
it in ~250 lines: `urllib.robotparser` was needed for §4.4 anyway, and Python's `ssl`
module validates against the system trust store without certifi.

**Cost, stated honestly.** The extractor is a readability-style heuristic — discard
tags that are never content, score the rest by paragraph density, keep the best
subtree. It is **not as good as trafilatura**. It is good enough to feed a summariser,
and it has no licence attached.

**What the audit did *not* fix.** `certifi` (MPL-2.0) is still in the tree via
mlx-lm → huggingface_hub → httpx, and `pathspec` (MPL-2.0) via the dev tooling. Both
predate this phase. MPL-2.0 is file-level copyleft and imposes nothing when used
unmodified, and neither is in §0.1's forbidden list — but neither is Apache or MIT
either, so they are recorded rather than glossed over. The hard violation, GPL via
`tld`, is what avoiding trafilatura actually prevented.

---

## ADR-018 — Search uses DuckDuckGo's lite endpoint, verified against robots.txt

**Decision.** `arc/web/search.py` queries `lite.duckduckgo.com/lite/`. No API key.

**Why.** §4.4 requires respecting robots.txt, and the obvious alternatives forbid
exactly what we would need to do — checked, not assumed:

| Endpoint | robots.txt |
|---|---|
| `google.com/search` | **disallowed** |
| `bing.com/search` | **disallowed** |
| `duckduckgo.com/html/` | **disallowed** |
| `lite.duckduckgo.com/lite/` | allowed |

Building robots.txt compliance and then circumventing it would make the compliance
decorative. No API key also means nothing is shared with a third-party search service
beyond the query itself, which any search necessarily reveals.

**Consequence.** ARC identifies itself honestly in its user agent rather than
impersonating a browser. A site that blocks a declared agent is a site that does not
want to be read, and evading that is not something this project should do.

---

## ADR-019 — Web facts carry provenance and expire by category

**Decision.** Every fact learned from the web is stored with its source URL, the date
retrieved, and confidence 0.7 (below anything the user stated directly). Staleness is
per-category: queries containing words like "latest", "current", "price", or "version"
get a 1-day TTL; everything else gets 90 days.

**Why.** §4.4 requires that ARC can cite a fact when asked and re-verify time-sensitive
ones rather than trusting them forever. A fact ARC cannot attribute is one it cannot
defend when questioned, and one it cannot re-check when it goes stale. The category
split matters because facts do not all age alike — "who invented the transistor" never
expires; "the latest Python release" expires in days.

**Measured.** First research run: 12.2s over the network, 5 facts stored. Identical
question immediately after: **0.01s, answered from memory, citing the source URL.**

**Consequence.** An unparseable `retrieved_at` is treated as stale. If we cannot vouch
for a fact's age, re-verifying is cheap and being confidently wrong is not.

---

## ADR-020 — Google and Bing search enabled, robots.txt compliance now opt-out

**Decision.** `config/default.yaml` sets `web.respect_robots: false` and
`web.search.backends: [google, bing, duckduckgo]`, tried in that order. This
**supersedes ADR-018's default** at the user's explicit instruction.

**Why the user's call.** Google and Bing both disallow `/search` in robots.txt, so
querying them requires the override. §0.3 is explicit that ARC does not negotiate
access on the user's behalf, and robots.txt is a convention rather than a legal
control, so this is theirs to set. It is a single config line either way.

**What was measured before enabling it**, on 2026-07-29 with an honest user agent:

| Backend | Response | Usable results |
|---|---|---|
| google | 200, 90 KB, contains `enablejs` | **0** — 3 links total, one to support.google.com |
| bing | 200, 115 KB | **9**, wrapped in `bing.com/ck/a?...&u=a1<base64>` |
| duckduckgo lite | 200 | works |

**So Google is configured first but does not function.** It serves a JavaScript shell
to any client that is not a browser. Making it the sole backend would have made search
return nothing at all.

**Consequence: backends fall through on empty results, not just on errors.** A backend
returning zero results is indistinguishable, from the caller's side, from a query with
no answers — so without fallthrough a non-functional default would make ARC look
ignorant rather than broken, and the second is far easier to diagnose. Google is tried
first as instructed; Bing answers.

**What was not done.** ARC still identifies itself honestly in its user agent and does
not execute JavaScript. Making Google work would require impersonating a browser to
defeat bot detection, which is a different kind of act from ignoring a robots.txt
directive, and is not something this project does. The disabled-compliance state is
logged as a warning at every startup so it can never be a surprise when reading the
audit trail later.

---

## ADR-020 addendum — Bing could not be made to work; DuckDuckGo restored as primary

**Requested:** use Bing for everything, avoid DuckDuckGo. **Outcome:** not achievable by
scraping. DuckDuckGo is primary again, with Bing tried second.

**What was tried**, ~60 requests over four parser revisions:

1. Naive `<h2><a>` matching — picked up the knowledge panel Bing renders *above* the
   results. "Barbara Liskov substitution principle" returned Wikipedia's "Barbara
   (given name)".
2. Scoped to Bing's organic `<li class="b_algo">` containers. No improvement — the bad
   results were inside them.
3. Query condensation to keywords, on the theory that Bing degraded past ~3 content
   words. 2/6 relevant.
4. Advert filtering on `msockid=` / `msclkid=` tracking parameters. Still 2/6.

**Why it fails.** Once Bing classifies a client as a bot it returns HTTP 200 with
arbitrary pages and paid placements dressed as organic results. `python TypeError
unhashable type list` returned **literotica.com**; `unhashable type list` returned
**foxnews.com**; `what is the capital of Mongolia` returned **capitalone.com**. This is
the worst possible failure mode — structurally valid, semantically wrong, and
indistinguishable from a real answer unless you already know the answer. It also
defeats backend fallthrough, which triggers on *empty* results.

Occasional good hits ("rust borrow checker" → LogRocket) are entity matches that
happen to coincide with the right answer, not evidence of it working.

**Kept from the attempt:** advert filtering, `b_algo` scoping, and `condense()`. All
are genuine improvements for when Bing does answer, and cost nothing. `condense` is
off by default — DuckDuckGo handles natural language better than keywords.

**The route that would honour the original preference.** A keyed search API — Brave
Search, Serper, or Tavily — returns clean JSON, needs no scraping, and would let
DuckDuckGo be dropped entirely. It costs money and sends queries to a third party,
which is why it was not assumed. It is a small backend to add on request.

**Not done, and not negotiable in this codebase:** making Bing or Google work would
require impersonating a browser to defeat bot detection. Ignoring a robots.txt
directive is the user's call; building evasion machinery is a different act.

---

## ADR-021 — ARC is a macOS application; Phase 8 re-scoped

**Decision.** ARC targets this Mac only. Phase 8's "Windows readiness" — the CI matrix,
a full `platform/windows.py`, the CUDA/vLLM backend — is cancelled. `windows.py` and
`linux.py` stay as stubs.

**Why.** ADR-007 already established that ARC runs on the Air and the Windows laptop is
a Track B training appliance. Dhruv confirmed on 2026-07-30 that he is not changing
machines at all. Building out a platform port for hardware that will never run it is
speculative work with no payoff, and §8 says not to over-engineer early.

**What stays, and why.** `arc/platform/` is not removed. It costs nothing now that it
exists, it is where the OS-specific code already lives, and it is the reason
`arc.config`, `arc.memory`, `arc.model`, `arc.agent`, and `arc.web` all import cleanly
with every Apple framework blocked — verified, not assumed. The macOS-only surface is
confined to `vision/`, `control/`, and `platform/macos.py`.

**Consequence.** Screen capture, OCR, the accessibility tree, and input control are
macOS-only by construction and say so when unavailable rather than failing quietly.

---

## ADR-022 — Config keys must be read, or they are lies

**Decision.** Every key in `config/default.yaml` is now read by the code that claims to
use it. Audited by cross-referencing the config tree against the source.

**Why.** Twelve keys were declared and then ignored, with their values hardcoded as
module constants — `hardware.os_reserve_gb`, `os_reserve_fraction`, `min_headroom_gb`,
`vlm_estimate_gb`, `embedder_estimate_gb`, `memory.working.reserve_for_reply`,
`memory.retrieval.per_strategy`, and `web.research.deep.corroboration_threshold` among
them. Editing any of them did nothing at all.

That is worse than not offering the setting. A missing key fails loudly the first time
you look for it; a key that is read from nowhere fails silently forever, and the next
person to tune memory sizing would have concluded the arithmetic was wrong rather than
that the input was ignored.

**Verified.** Setting `ARC_HARDWARE__OS_RESERVE_FRACTION=0.50` now moves usable memory
from 11.5 GB to 8.0 GB. Before the fix it changed nothing.

**Consequence.** The sizing helpers fall back to the historical defaults when config
cannot be loaded, because the hardware probe runs on a fresh install before anything is
configured.
