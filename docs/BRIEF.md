# Project ARC — Build Brief

**Claude, this is the full spec for what I want us to build together. Read all of it before you start.**

I'm Dhruv. It's July 2026. We are not doing voice, speech, or TTS yet — that comes later, so ignore
it entirely for now.

---

## 0. START HERE — WHAT I ACTUALLY MEAN

Read this section carefully. If anything later in this brief seems to contradict it, this section
wins.

### 0.1 What I mean by "no license"

I have a JARVIS right now that's built on top of a licensed GitHub repo, and I want to get away from
that. When I say I want something with "no license," here's what I actually need you to understand:

Every piece of software has a license. That's unavoidable. My real problem with what I have now
isn't that a license exists — it's that I didn't design the architecture, I don't fully understand
code I didn't write, and I might be bound by attribution or copyleft terms I haven't thought through.

So here's what I want instead: **I own 100% of the application code, and every dependency is
permissively licensed — Apache-2.0 or MIT only.** Those licenses cost me nothing. No royalties, no
usage caps, no commercial restrictions, no copyleft. Apache-2.0 also includes an explicit patent
grant that MIT doesn't. Under those terms, this system is genuinely mine — to run, modify, sell, or
close-source however I want.

**This is a hard rule for you: never pull in a GPL, AGPL, SSPL, "research-only," "non-commercial," or
custom-community-licensed dependency.** If the best library for something is copyleft, use the
second-best permissive one and note the tradeoff for me in `docs/DEPENDENCIES.md`. Start that file at
commit #1 and keep it current — I want a running ledger of every dependency and its license.

### 0.2 What I mean by "build my own LLM"

I know there are two different projects hiding in what I'm asking for, and I want both, running in
parallel. Don't blur them together:

| | **Track A — The Agent** | **Track B — The Model** |
|---|---|---|
| What it is | JARVIS itself: memory, tools, OS control, screen reading, web research, reasoning | A transformer LLM I train from raw text |
| Whose code | Mine, written by us | Mine, written by us |
| The brain | An Apache-2.0 open-weight model running locally | The model I train |
| Timeline | Weeks. I want it usable fast. | Months. This is how I learn ML. |
| Realistic quality | An actually useful assistant | A small model, roughly GPT-2 to GPT-3-small grade |

**Be blunt with me about the ceiling here and don't let it get fuzzy.** I understand that training
something competitive with Claude or GPT or Gemini from scratch would take $10M–$100M+ in compute, a
data team, and a cluster. That's not a solo project and I'm not pretending otherwise. What I *can* do
solo — and what I want — is train a real model end to end and understand every piece of it.

Because of that split, **the model has to be a swappable component behind a stable interface.**
Track A ships now on open weights. Track B trains my own model. When Track B produces something, it
drops into the same socket with a one-line config change. This is the most important architectural
decision in the whole project. Design for it from day one.

### 0.3 How much access it gets

I want **full, unrestricted access to my machine. No permission prompts, no deny-list.** It reads and
writes anywhere my user account can, runs any shell command, controls mouse and keyboard, captures
the screen, launches any app — without asking me first. That's deliberate, and I don't want you
negotiating it.

Two things I *do* want you to build, not as guardrails but because I'll need them to debug this
thing:

1. **An append-only action log** at `~/.arc/audit/*.jsonl` — every tool call, its arguments, its
   result, timestamped. When this thing does something weird at 2am I need to be able to see what
   happened.
2. **A hard kill switch** — global hotkey plus an `arc kill` command that SIGKILLs the process tree
   and releases input control immediately. An agent with mouse and keyboard control that hits a loop
   can make my machine unusable. I always need to be able to stop it.

Also build a `--dry-run` mode that logs intended actions without executing them. Not for safety — I
just know we'll need it constantly while developing.

Don't add confirmation prompts, deny-lists, or capability sandboxes. Do build the config plumbing in
`config/policy.yaml` so I can turn that stuff on later if I change my mind, but default everything to
permissive.

---

## 1. WHAT WE'RE BUILDING

**ARC** — my personal, local-first, fully-owned AI assistant. It needs to:

1. Run a language model **locally** on my machine. No cloud API required for core reasoning.
2. Have **persistent long-term memory** — a real memory system, not a chat log — that accumulates
   knowledge about me, my files, my projects, and past conversations across sessions.
3. Have **unrestricted local access** — read/write any file, run any command, launch and control any
   application.
4. **See my screen** (capture + vision model + OCR) and **control it** (mouse, keyboard).
5. **Research on the open web** when its memory isn't enough, then **write what it learned back into
   memory** so it doesn't look the same thing up twice.
6. Keep the **model swappable**, including for the model I train myself.
7. Contain **zero code copied from a third-party agent framework.** Permissive libraries only —
   PyTorch, transformers, llama.cpp, sqlite. The agent architecture is mine.

Call it `arc`. Root at `~/arc`, runtime data in `~/.arc/`.

---

## 2. MY HARDWARE SITUATION

I'm on **Apple Silicon (macOS)** right now, but I'm planning to **switch to Windows** before too
long. So:

- **Write platform-agnostic code by default.** Anything OS-specific lives behind interfaces in
  `arc/platform/`, with `macos.py`, `windows.py`, and `linux.py` implementations picked at runtime by
  a factory in `arc/platform/__init__.py`.
- **Never** call a macOS-only API from business logic. If you need `osascript`, it goes in
  `platform/macos.py` behind something like `open_application(name: str)` that `windows.py`
  implements its own way.
- `pathlib.Path` everywhere. Never string-concatenate paths, never hardcode `/` or `\`.
- The inference layer needs to support Metal/MLX (my Mac now), CUDA (Windows later), and CPU
  (fallback). Detect at startup, pick automatically, let me override in config.
- **First thing in Phase 1:** a hardware probe that writes `~/.arc/hardware.json` — OS, arch, CPU
  cores, RAM, GPU vendor/model, VRAM or unified memory, available accelerator backends. Everything
  downstream (model size, quantization, batch size) reads from that file instead of assuming.

Size the local model against available memory roughly like this:

| Memory available for the model | What to run |
|---|---|
| ≤ 8 GB | 3–4B params, 4-bit quantized |
| 16 GB | 7–8B, 4-bit |
| 32 GB | 14B 4-bit, or 8B at 8-bit |
| 64 GB+ | 30B-class MoE 4-bit |
| 128 GB+ | 70B 4-bit, or a large MoE |

---

## 3. PICKING THE BRAIN FOR TRACK A

**Apache-2.0 licensed open weights only.** As of mid-2026 my understanding of the good permissive
options is:

- **Qwen3** (Alibaba) — Apache-2.0, ~1.7B up to 235B, strong tool calling and coding. Probably my
  default.
- **Mistral Small / Mistral Large 3** — Apache-2.0, clean licensing, strong reasoning.
- **Gemma 4** — Apache-2.0, efficient, good quality per parameter, MoE variants available.

**Don't default to Llama** — Meta's community license isn't Apache or MIT and carries conditions I
don't want. Nothing tagged research-only or non-commercial.

**Verify every model's license yourself before you hardcode it** by fetching its Hugging Face model
card. Licenses change between releases and I don't want to rely on my memory or yours. Record what
you find — URL and date checked — in `docs/DEPENDENCIES.md`. If something I named turns out to be
non-permissive, pick something else and tell me why.

Same rule for the **vision model** (Qwen-VL line is the obvious default) and the **embedding model**
(small, CPU-capable, fully local — no embedding API calls).

---

## 4. ARCHITECTURE

```
~/arc/
├── arc/
│   ├── __main__.py            # CLI entrypoint
│   ├── config.py              # loads config/*.yaml, env, hardware.json
│   │
│   ├── model/                 # ── THE SWAPPABLE BRAIN ──
│   │   ├── base.py            # LanguageModel ABC — the critical interface
│   │   ├── llamacpp.py        # GGUF via llama.cpp (everywhere: CPU/Metal/CUDA)
│   │   ├── mlx_backend.py     # Apple Silicon fast path
│   │   ├── vllm_backend.py    # NVIDIA fast path, for the Windows move
│   │   ├── transformers_backend.py
│   │   ├── custom.py          # ── TRACK B PLUGS IN HERE ──
│   │   └── router.py          # picks backend from hardware.json + config
│   │
│   ├── vision/
│   │   ├── capture.py         # cross-platform screenshot, multi-monitor
│   │   ├── vlm.py             # vision-language model wrapper
│   │   └── ocr.py             # local OCR for dense text
│   │
│   ├── memory/                # ── THE MEMORY CACHE ──
│   │   ├── store.py           # SQLite + sqlite-vec, single file, no server
│   │   ├── embedder.py        # local embedding model
│   │   ├── episodic.py        # conversations, events, what happened when
│   │   ├── semantic.py        # facts, entities, relationships (graph)
│   │   ├── procedural.py      # learned workflows, my preferences
│   │   ├── working.py         # context-window budget manager
│   │   ├── retrieval.py       # hybrid: vector + BM25 + graph + recency, reranked
│   │   └── consolidation.py   # background: dedupe, summarize, decay, promote
│   │
│   ├── tools/                 # ── CAPABILITIES ──
│   │   ├── registry.py        # decorator registration → JSON schema
│   │   ├── filesystem.py      # read/write/move/delete/search, unrestricted
│   │   ├── shell.py           # arbitrary command execution
│   │   ├── apps.py            # launch/focus/quit/enumerate apps
│   │   ├── screen.py          # screenshot, read screen, describe screen
│   │   ├── input_control.py   # mouse and keyboard
│   │   ├── web.py             # search + fetch + extract → clean text
│   │   ├── code.py            # write and run code in a scratch dir
│   │   └── memory_tools.py    # its own remember/recall/forget
│   │
│   ├── agent/                 # ── THE LOOP ──
│   │   ├── loop.py            # perceive → retrieve → plan → act → observe → store
│   │   ├── planner.py         # break goals into steps
│   │   ├── executor.py        # tool dispatch, retries, error recovery
│   │   ├── context.py         # assembles the prompt from memory + state
│   │   └── prompts/           # system prompts as editable text files
│   │
│   ├── platform/              # ── OS ABSTRACTION ──
│   │   ├── base.py  macos.py  windows.py  linux.py
│   │
│   ├── audit/
│   │   ├── logger.py          # append-only JSONL of everything
│   │   └── killswitch.py      # global hotkey + `arc kill`
│   │
│   └── interface/
│       ├── cli.py             # my main interface for now
│       └── server.py          # local HTTP/WebSocket API for a future GUI
│
├── training/                  # ── TRACK B: MY OWN MODEL ──
│   ├── 01_tokenizer/          # BPE from scratch
│   ├── 02_data/               # corpus, cleaning, dedup, sharding
│   ├── 03_model/              # transformer from scratch in PyTorch
│   ├── 04_pretrain/           # training loop, schedule, checkpointing, pause/resume
│   ├── 05_finetune/           # SFT + LoRA on my own data
│   ├── 06_eval/               # perplexity, task evals, qualitative harness
│   ├── budget/                # ── credit tracking + auto-pause ──
│   └── NOTEBOOK.md            # lab notebook: every run, config, result
│
├── config/
│   ├── default.yaml  models.yaml  policy.yaml  training.yaml
├── tests/
├── docs/
│   ├── ARCHITECTURE.md  DEPENDENCIES.md  ML_CURRICULUM.md  DECISIONS.md
└── pyproject.toml
```

### 4.1 The model interface — get this exactly right

`arc/model/base.py` is what makes the brain swappable. Every backend implements it, including the
model I eventually train. Keep it as **narrow as possible** so a from-scratch model has a realistic
shot at satisfying it:

```python
class LanguageModel(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], *,
                 tools: list[ToolSchema] | None = None,
                 max_tokens: int = 2048,
                 temperature: float = 0.7,
                 stop: list[str] | None = None) -> Completion: ...

    @abstractmethod
    def stream(self, ...) -> Iterator[Token]: ...

    @abstractmethod
    def count_tokens(self, text: str) -> int: ...

    @property
    @abstractmethod
    def context_length(self) -> int: ...

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """native_tool_calling, vision, json_mode, max_context, etc."""
```

**Important:** my own model won't have native tool calling, and neither will small open models. Build
a `capabilities`-aware fallback in `agent/executor.py` — if `native_tool_calling` is False, drop to a
prompted ReAct-style protocol where the model emits a parseable block (fenced JSON) that the executor
extracts with a **tolerant** parser. Build that parser properly, with repair heuristics for truncated
or malformed JSON. This is specifically what will let my own model drive the agent later, so don't
treat it as an afterthought.

### 4.2 Memory — this is the part I care most about

When I say "its own memory cache with full capabilities," I don't mean dumping chat history into a
vector DB. I want three cooperating layers plus a consolidation process:

- **Episodic** — timestamped events and conversation turns, vector + relational hybrid. Answers "what
  did we do last Tuesday?"
- **Semantic** — extracted facts and entities with typed relationships, stored as a graph (nodes and
  edges in SQLite). Answers "who is X and how do they relate to Y?" Supports multi-hop.
- **Procedural** — learned workflows and my preferences. "When Dhruv says 'clean up downloads' he
  means this." Written by the consolidation process out of repeated episodic patterns.

**Storage:** SQLite with the `sqlite-vec` extension. One file, no server, trivially portable across
my macOS→Windows migration, and my whole memory becomes a single backup-able artifact. Don't
introduce a server-based vector DB.

**Retrieval has to be hybrid, run in parallel, and rerank:**

1. Dense vector similarity
2. BM25 keyword match (SQLite FTS5)
3. Entity/graph traversal from entities in the query
4. Recency-weighted scan

Merge, dedupe, rerank on a combined score, then fit into the working-memory budget.

**Consolidation** runs in the background on idle and on a schedule: dedupe near-identical memories,
summarize long episodic runs into compact semantic facts, decay salience of unused memories, promote
repeated patterns into procedural memory. Log everything it changes to the audit log — I don't want
silent memory mutation.

**Write-back:** when it researches something on the web, the distilled result goes into semantic
memory with source URL and timestamp. Asking the same question later should hit memory, not the
network. Facts carry confidence and age, and time-sensitive categories get re-verified instead of
trusted forever.

### 4.3 Screen access

- **Capture:** cross-platform, multi-monitor, region-croppable, with configurable downscaling before
  it hits the vision model. Full-resolution screenshots will destroy the token budget.
- **Understanding, cheapest method first:**
  1. Native accessibility tree (macOS AX API / Windows UI Automation) — structured, fast, exact. Use
     this whenever it's available; it's dramatically better than pixels.
  2. Local OCR for dense text regions.
  3. Vision-language model for real visual understanding and element localization.
- **Control:** mouse move/click/drag/scroll, keyboard type/hotkey, platform-abstracted. Prefer
  accessibility-API targeting over raw pixel coordinates — pixel targeting breaks every time a window
  moves or the resolution changes.
- Cache screen state with a short TTL so repeated queries in one reasoning step don't re-capture.

### 4.4 Web research

Fetch → extract main content → strip boilerplate → chunk → summarize against my query → write to
semantic memory with provenance. Respect `robots.txt`, rate-limit yourself, and use a headless
browser when a plain fetch returns an empty JS shell. Every fact from the web carries its source URL
and retrieval date, and it has to be able to cite that when I ask.

---

## 5. BUILD PHASES

Work these in order. **After each phase: run the tests, commit with a real message, and give me a
short summary of what works now and what I can try.** Don't barrel through all eight silently.

### Phase 1 — Foundation

Repo scaffold, `pyproject.toml`, config loader, structured logging, audit logger, kill switch,
hardware probe writing `hardware.json`, platform abstraction with macOS implemented and
Windows/Linux stubbed, an `arc doctor` command that reports my environment. Tests for config and the
probe. Start `docs/DEPENDENCIES.md`.

### Phase 2 — The brain

`LanguageModel` ABC. llama.cpp backend (works everywhere) plus the MLX fast path for my Mac. Model
management CLI — `arc model pull`, `arc model list`, `arc model use`. Router that picks a backend
from `hardware.json`. Streaming, token counting, and an `arc chat` REPL. Stateless is fine, but by
the end of this phase I want to be talking to a local model. **Verify each license before adding it
to `models.yaml`.**

### Phase 3 — Memory

SQLite + sqlite-vec schema. Local embedder. Three memory layers. Hybrid retrieval. Working-memory
budget manager. Consolidation job. `arc memory` subcommands: `search`, `add`, `stats`, `export`,
`forget`. Wire memory into `arc chat` so conversations survive restarts. **Test with several thousand
synthetic memories** — retrieval problems only show up at scale.

### Phase 4 — Tools and the agent loop

Tool registry with decorator registration and auto-generated JSON schemas. Filesystem, shell, and
code tools. The agent loop with planning, execution, error recovery, step limits. Native tool-calling
path plus the prompted-ReAct fallback and tolerant parser. Audit logging on every call. By the end of
this phase ARC should be doing real multi-step work on my files and shell.

### Phase 5 — Web

Search, fetch, extraction, headless-browser fallback, summarization, memory write-back with
provenance, staleness policy.

### Phase 6 — Screen and control

Capture, per-platform accessibility-tree readers, OCR, VLM integration, input control, element
targeting. Most platform-specific phase — keep the abstraction clean, all of this gets rewritten for
Windows.

### Phase 7 — Integration and hardening

End-to-end tests of realistic multi-tool tasks. Performance profiling — find the slow paths, which
will probably be embedding and screenshot encoding. Crash recovery and resumable tasks. Local
HTTP/WebSocket server. Startup time and model warm-loading.

### Phase 8 — Windows readiness

Full `platform/windows.py`. CI matrix running tests on macOS and Windows. CUDA/vLLM backend. A
documented procedure for moving `~/.arc/` between machines.

---

## 6. TRACK B — TRAINING MY OWN MODEL

This runs alongside Track A. It's as much a course as a build. **Write `docs/ML_CURRICULUM.md` as a
genuine teaching document** — I want to learn ML, not just end up owning a model. Explain the math
and the intuition, not just which API to call. Keep `training/NOTEBOOK.md` as a real lab notebook:
every run, its config, its loss curve, what I learned from it.

### 6.1 The curriculum

**Stage 0 — Foundations.** Before we train anything: tensors and autograd from first principles,
backprop by hand on a tiny network, a bigram language model, then a single self-attention head from
scratch. Don't let me skip this. Everything downstream is meaningless without it.

**Stage 1 — Tokenizer.** Byte-pair encoding implemented from scratch, trained on our corpus. I want
to understand why vocab size trades against sequence length, and why tokenization causes so many of
the weird failures people blame on models.

**Stage 2 — The transformer.** In PyTorch, from scratch, no `transformers` library: multi-head
attention, rotary position embeddings, RMSNorm, SwiGLU feed-forward, residual stream, causal masking,
KV cache. Modern architecture, not the 2017 original. Every component gets a docstring explaining
*why* it exists and what problem it solves.

**Stage 3 — Data.** Permissively-licensed corpus — open web-text datasets, public domain books, open
code. Clean, deduplicate with MinHash, quality-filter, shard into memory-mapped binaries. Explain the
data-quality-to-model-quality relationship, which I gather is the most underrated thing in practical
ML.

**Stage 4 — Pretraining.** AdamW, cosine schedule with warmup, gradient clipping, mixed precision,
gradient accumulation, checkpointing, resume. Instrument the loss curves and teach me to read them —
what divergence looks like, how to size a model against available compute and data.

**Stage 5 — Post-training.** SFT on instruction data. LoRA/QLoRA for cheap adaptation. Optionally
DPO. Fine-tune on **my own data** — my notes, my writing, my tool-call traces from Track A — so my
model has a personality and a competence no off-the-shelf model has.

**Stage 6 — Evaluation.** Perplexity, task benchmarks, honest qualitative harness. Include a direct
comparison against the Track A open-weight model so I develop a calibrated sense of the gap.

**Stage 7 — Integration.** Implement `arc/model/custom.py` so my trained model satisfies the
`LanguageModel` interface and I can select it with `arc model use custom`. Even if it's much weaker,
running my own model inside my own agent is the milestone that makes this whole thing real to me.

### 6.2 Compute budget — target a 24-hour run

I don't want a 3-hour toy run. **Plan the main pretraining run for roughly 24 hours of wall-clock
training on rented GPUs**, and size the model and dataset to actually use that budget well.

For reference, Karpathy's nanochat gets a GPT-2-grade model in ~3 hours on an 8×H100 node for around
$73. Scaling that to 24 hours on the same class of node means:

- **Roughly 8× the compute**, which should get me materially past GPT-2 — a ~1B-parameter-class model
  trained on meaningfully more tokens, in the neighborhood of GPT-3-small quality.
- **Cost on the order of $200–$400 on spot instances, $400–$700 on-demand**, at mid-2026 8×H100
  pricing. Give me a real estimate before we launch anything, not a hand-wave.

**Do the scaling math with me before committing.** Use Chinchilla-style compute-optimal reasoning to
pick the parameter count and token count that best spend a 24-hour budget on whatever node we rent,
and show me the arithmetic. I want to understand the tradeoff between "bigger model, less data" and
"smaller model, more data," not just be handed a config. Write the reasoning into
`docs/ML_CURRICULUM.md`.

Design the run so 24 hours is **not one fragile session**. It must survive interruption:

- Checkpoint every N steps (tune N so we lose at most ~10 minutes of work).
- Every checkpoint includes model weights, optimizer state, LR scheduler state, RNG state, dataloader
  position, and step count — everything needed for a **bit-exact resume**.
- Upload checkpoints to durable cloud storage as they're written, not just to the instance's local
  disk. Spot instances get preempted; I don't want to lose 18 hours to that.
- `training/RESUME.md` documenting the exact commands to bring a run back from a checkpoint on a
  fresh instance.

Write `training/RENTING_GPUS.md` covering provider options, spot vs on-demand tradeoffs,
checkpoint-to-cloud setup, and pre-launch cost estimation.

### 6.3 Credit-aware pause and resume — build this properly

This matters to me and I want it engineered, not bolted on. **When credits run low, training pauses
cleanly and resumes when they reset.** Two separate meters can run out, and I want both handled:

- **Compute credits** — the GPU rental budget for the training run itself.
- **My Claude Code usage limits** — the budget for *you* helping me, which resets on a window.

Note these fail differently: compute credits stop the *training job*, my Claude usage limits stop
*our working session* while a job may still be running. Handle both, and tell me clearly which one
tripped.

Build this in `training/budget/`:

**Budget tracking.** A `BudgetTracker` that reads limits from `config/training.yaml` — total budget,
warning threshold, hard-stop threshold, reset schedule. It tracks spend continuously: elapsed GPU
hours × rate for compute, and for Claude Code, a token/usage counter I can update plus a
reset-window timestamp. Persist state to `~/.arc/training/budget.json` so it survives a restart.

**Thresholds and behavior.** Make these configurable with sensible defaults:

- **Warning (default 75%)** — log it, notify me, keep training.
- **Pause (default 90%)** — checkpoint immediately, shut the run down cleanly, release the GPU
  instance so I stop paying for idle hardware, and record why it paused and when the budget resets.
- **Hard stop (100%)** — checkpoint, terminate everything, alert me loudly.

**Clean pause, not a kill.** A pause finishes the current step, writes a full checkpoint, flushes it
to cloud storage, writes a `PAUSED` state file with the step number, reason, and expected resume
time, then exits zero. Never leave a half-written checkpoint — write to a temp file and atomically
rename.

**Auto-resume.** A supervisor process (`training/budget/supervisor.py`) that runs independently of
the training job. It watches for the reset window, and when credits are back it re-provisions an
instance, pulls the latest checkpoint, and restarts training from the exact step. Two modes I want:
`--auto` resumes without asking, `--confirm` notifies me and waits for my go-ahead. Default to
`--confirm` so I don't get surprise charges.

**Manual control.** CLI commands I can use any time:

```
arc train status      # step, elapsed, spend, budget remaining, est. time/cost to finish
arc train pause       # graceful checkpoint-and-stop right now
arc train resume      # resume from latest checkpoint
arc train estimate    # projected total cost and completion time at current rate
arc train budget set  # adjust limits mid-run
```

`arc train status` should be genuinely informative — I want to glance at it and know whether I'm
going to run out before the run finishes.

**Projected-overrun detection.** Don't just react at 90%. Continuously project total spend against
remaining steps, and if the run is going to blow the budget before it completes, tell me *early* with
options: reduce steps, reduce model size, raise the budget, or accept an early stop. Surface that as
soon as the projection is confident, not at hour 22.

**Log every budget event** to the audit log — thresholds crossed, pauses, resumes, spend snapshots.
When I look back at a run I want the full financial history next to the loss curve.

### 6.4 Set my expectations honestly

Most of Stages 0–3 run fine on my Mac. Stage 4 needs rented GPUs. And be straight with me in the
curriculum doc: even with a 24-hour run, what comes out will write coherent English and follow simple
instructions. It won't be Claude. That's the expected outcome, it's still an accomplishment very few
people actually finish, and I'd rather know that going in than be disappointed at the end.

---

## 7. HOW I WANT THE CODE WRITTEN

- **Python 3.12+.** Full type hints on every public function. `mypy --strict` clean. `ruff` for
  format and lint.
- **Docstrings explain *why*, not *what*.** This codebase is a teaching artifact for me too.
- **Test as we go** with `pytest` — unit tests for pure logic, integration tests for tools, a small
  end-to-end suite. Don't defer testing to the end.
- **Treat dependencies as a liability.** Before adding one, ask whether 50 lines of my own code would
  do. Every dependency gets a row in `docs/DEPENDENCIES.md` with its license and why it's there.
- **Config over constants.** No magic numbers in logic — they go in `config/`.
- **Fail loudly, recover gracefully.** Never swallow an exception silently. The agent loop catches
  tool errors, feeds them back to the model as observations, and adapts.
- **Structured logging** (JSONL), not print statements.
- **Commit per meaningful unit** with real messages. Keep `docs/DECISIONS.md` as an architecture
  decision record — what we chose, what we rejected, and why.
- **Async where it pays** — I/O-bound tool calls, parallel retrieval. Not everywhere.

---

## 8. HOW I WANT YOU TO WORK WITH ME

- **Ask me before assuming.** When a decision has real tradeoffs — model choice, memory schema, how
  aggressive consolidation should be, how to spend the training budget — lay out the options and let
  me pick.
- **Teach me as you build.** I'm doing this to learn ML and systems engineering. When you implement
  something non-obvious, explain the concept in your summary, not just the code.
- **Tell me when something won't work.** If I ask for something infeasible, say so directly and
  propose the closest thing that is. Don't build me something impressive-looking that doesn't
  actually work.
- **Ship working increments.** Every phase ends with something I can run. Don't vanish for eight
  phases and hand me a monolith.
- **Don't over-engineer early.** Phase 1 doesn't need a plugin system. The swappable brain and the
  platform abstraction are load-bearing — build those properly and keep everything else simple until
  it needs not to be.

---

## 9. WHAT TO DO RIGHT NOW

1. Confirm you've read and understood Section 0 (licensing, the two tracks, full access) and Section
   4.1 (the swappable brain). State them back to me briefly so I know we're aligned.
2. Ask me anything that would change Phase 1 or 2 — where to put the project, Python version manager,
   preferred model family, and what my actual GPU budget is for the Track B 24-hour run.
3. Run the hardware probe on this machine and tell me what it found and what model size that implies.
4. Propose a concrete Phase 1 file list for my approval.
5. Once I approve, build Phase 1, test it, commit it, and report back.

**Don't start Phase 2 until Phase 1 runs and I've confirmed it.**