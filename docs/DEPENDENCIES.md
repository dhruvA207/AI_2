# Dependencies

Running ledger of every dependency, its licence, and why it earns its place. Required by
`docs/BRIEF.md` §0.1, started at commit #1 and kept current.

## The rule

**Application code is 100% mine. Every dependency is permissively licensed.** No GPL, AGPL,
SSPL, research-only, non-commercial, or custom-community licences — ever. If the best library
for a job is copyleft, the second-best permissive one gets used and the tradeoff is recorded
here.

Before adding anything, the question is whether fifty lines of our own code would do (§7:
"treat dependencies as a liability"). Several times so far the answer has been yes — see
[Deliberately not used](#deliberately-not-used).

## Runtime dependencies

| Package | Version | Licence | Why it is here |
|---|---|---|---|
| PyYAML | 6.0.3 | MIT | Config files. Writing a YAML parser is not a good use of anyone's time, and `json`-only config would make `config/*.yaml` unpleasant to hand-edit. |

That is the entire runtime dependency list at Phase 1. Everything else — CLI parsing, JSON,
paths, process control, threading, hardware probing — is standard library.

## Development dependencies

| Package | Version | Licence | Why it is here |
|---|---|---|---|
| pytest | 9.1.1 | MIT | Test runner (§7: test as we go). |
| ruff | 0.16.0 | MIT | Lint and format. One tool instead of flake8 + black + isort. |
| mypy | 2.3.0 | MIT | `--strict` type checking (§7). |
| types-PyYAML | 6.0.12 | Apache-2.0 | Type stubs so `--strict` passes across the YAML boundary. |
| mypy_extensions | 1.1.0 | MIT | Transitive dependency of mypy. |

## Planned, not yet added

Recorded now so the licence question is settled before the code depends on it.

| Package | Licence | Verified | For |
|---|---|---|---|
| mlx / mlx-lm | MIT | 2026-07-28 | Phase 2 Apple Silicon inference fast path. Apple Silicon only — see [DECISIONS](DECISIONS.md). |
| llama.cpp (via bindings) | MIT | not yet | Phase 2 portable GGUF backend (CPU / Metal / CUDA). |
| sqlite-vec | Apache-2.0 | not yet | Phase 3 vector search inside SQLite, no server. |
| PyTorch | **BSD-3-Clause** | 2026-07-28 | Track B training. **See the exception below.** |
| peft / trl / transformers | Apache-2.0 | not yet | Track B QLoRA fine-tuning on the Windows box. |

### Exception: PyTorch is BSD-3-Clause, not Apache-2.0 or MIT

§0.1 says "Apache-2.0 or MIT only." PyTorch is BSD-3-Clause, which is neither. It is recorded
here as a **conscious, brief-sanctioned exception** rather than quietly waved through:

- §1.7 of the brief names PyTorch explicitly as an acceptable library.
- BSD-3-Clause is functionally equivalent to MIT for our purposes: permissive, no copyleft, no
  royalties, no field-of-use restriction. It requires attribution and adds a
  no-endorsement clause. It lacks Apache-2.0's explicit patent grant.
- There is no permissive alternative at PyTorch's capability level.

## Model weights

Every model's licence is verified against its **live Hugging Face model card** before being
written into `config/models.yaml` — never from memory, ours or anyone's (§3).

| Model | Licence | Verified | Notes |
|---|---|---|---|
| [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Apache-2.0 | 2026-07-28 | 0.6B params, 32K context |
| [Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | Apache-2.0 | 2026-07-28 | 1.7B params, 32K context |

Not used, and why: **Llama** (Meta community licence — not Apache or MIT, carries conditions),
anything tagged research-only or non-commercial.

A fine-tune of Apache-2.0 weights is fully ours to run, modify, sell, or close-source. The base
model's NOTICE and attribution travel with the derivative; that is the whole cost.

## Datasets

Data licences are a separate question from code licences, and the distinction matters.

| Dataset | Licence | Verified | Notes |
|---|---|---|---|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | CDLA-Sharing-1.0 | 2026-07-28 | See below. |

**CDLA-Sharing-1.0 and why it is acceptable.** The name suggests share-alike, but
[§3.5 of the licence](https://cdla.dev/sharing-1-0/) states: *"This Agreement imposes no
obligations or restrictions on Your Use or Publication of Results."* Results are defined as the
outputs of computational use of the data — **a trained model is a Result and is completely
unencumbered.** The sharing obligation attaches only to redistributing the *data itself*.

**Operational rule that follows:** corpora live in `~/.arc/`, never in git. As long as we do not
republish the dataset, nothing propagates to our model or our code.

## Deliberately not used

| Rejected | Licence | Why not | What we do instead |
|---|---|---|---|
| psutil | BSD-3-Clause | Would be convenient for the hardware probe, but it is a dependency for ~50 lines of parsing, and macOS ships `sysctl`/`system_profiler` anyway. | `subprocess` + stdlib in `arc/platform/macos.py` |
| click / typer | BSD-3-Clause / MIT | The CLI is four subcommands and four flags. | `argparse` |
| A server-based vector DB | varies | §4.2 requires one portable SQLite file, not a service. | SQLite + sqlite-vec (Phase 3) |
| Any third-party agent framework | varies | §1.7: zero copied agent-framework code. The architecture is ours. | `arc/agent/` (Phase 4) |
