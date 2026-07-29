"""Command-line entry point.

``argparse`` rather than a CLI framework: the brief tells us to treat dependencies as
a liability (§7), and the whole surface here is four subcommands and four flags.

Every command reports through ``Check`` records rather than printing as it goes. That
buys two things the brief asks for — ``--json`` output for anything that needs to be
machine-read, and a single place where "did anything fail" is decided, so exit codes
stay honest instead of every command inventing its own convention.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from arc import __version__
from arc.audit import AuditLogger, KillSwitch
from arc.config import Config
from arc.errors import ArcError, ConfigError
from arc.hardware import ModelSizing, refresh
from arc.paths import arc_home, audit_dir, config_dir, ensure_runtime_dirs, log_dir, run_dir
from arc.platform import HardwareInfo, get_platform, platform_name

Status = Literal["ok", "warn", "fail"]

#: Optional at Phase 1, required later. Reported so `arc doctor` answers "is this
#: machine ready for Phase 2" without anyone having to remember what Phase 2 needs.
_OPTIONAL_DEPS = (
    ("mlx", "Apple Silicon inference fast path (Phase 2)"),
    ("torch", "from-scratch training track (Track B)"),
)

_SYMBOLS = {"ok": "ok  ", "warn": "warn", "fail": "FAIL"}


@dataclass(frozen=True, slots=True)
class Check:
    """One diagnostic line."""

    name: str
    status: Status
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return asdict(self)


def _render(checks: Sequence[Check], *, as_json: bool) -> None:
    """Print checks as either aligned text or one JSON document."""
    if as_json:
        payload = {
            "ok": not any(c.status == "fail" for c in checks),
            "checks": [c.to_dict() for c in checks],
        }
        print(json.dumps(payload, indent=2, default=str))
        return

    width = max((len(c.name) for c in checks), default=0)
    for check in checks:
        print(f"  {_SYMBOLS[check.status]}  {check.name.ljust(width)}  {check.detail}")


def _exit_code(checks: Sequence[Check]) -> int:
    """Fail the process if any check failed. Warnings are not failures."""
    return 1 if any(c.status == "fail" for c in checks) else 0


def _check_paths() -> list[Check]:
    """Verify the runtime tree exists and is writable.

    Checked explicitly because every later phase assumes it. A read-only ``~/.arc``
    surfaces here as one clear line rather than as a confusing traceback from the
    audit logger halfway through a task.
    """
    checks: list[Check] = []
    try:
        ensure_runtime_dirs()
    except OSError as exc:
        return [Check("runtime dirs", "fail", f"could not create: {exc}")]

    for label, path in (
        ("arc home", arc_home()),
        ("audit dir", audit_dir()),
        ("log dir", log_dir()),
        ("run dir", run_dir()),
    ):
        if not path.is_dir():
            checks.append(Check(label, "fail", f"missing: {path}"))
        elif not os.access(path, os.W_OK):
            checks.append(Check(label, "fail", f"not writable: {path}"))
        else:
            checks.append(Check(label, "ok", str(path)))
    return checks


def _check_config(directory: Path | None) -> list[Check]:
    """Load configuration and report which files contributed."""
    source = directory if directory is not None else config_dir()
    if not source.is_dir():
        return [Check("config", "fail", f"directory not found: {source}")]

    files = sorted(p.name for p in source.glob("*.yaml"))
    try:
        Config.load(directory=source)
    except ArcError as exc:
        return [Check("config", "fail", str(exc))]

    checks = [Check("config", "ok", f"{source} ({', '.join(files) or 'no files'})")]
    local = arc_home() / "config.yaml"
    checks.append(
        Check("config overrides", "ok", str(local) if local.is_file() else "none (optional)")
    )
    return checks


def _hardware_checks(info: HardwareInfo, sizing: ModelSizing, path: Path) -> list[Check]:
    """Render an already-completed probe as diagnostic lines.

    Takes the probe result rather than probing itself so that a command which needs
    both the raw data and the report does not probe twice — the macOS probe shells
    out to ``system_profiler``, which costs about a second.
    """
    cores = f"{info.cpu_cores_physical}c"
    if info.cpu_performance_cores and info.cpu_efficiency_cores:
        cores = f"{info.cpu_performance_cores}P+{info.cpu_efficiency_cores}E"

    memory = f"{info.ram_total_gb:.0f} GB"
    memory += " unified" if info.unified_memory else f" RAM, {info.vram_gb or 0:.0f} GB VRAM"

    chassis = f"{info.chassis}, " if info.chassis else ""
    checks = [
        Check("machine", "ok", f"{chassis}{info.cpu_model}, {cores}, {memory}"),
        Check("os", "ok", f"{info.os_name} {info.os_version} ({info.arch})"),
        Check("accelerators", "ok", ", ".join(info.accelerators) or "cpu only"),
        Check("hardware.json", "ok", str(path)),
        Check(
            "recommended model",
            "ok",
            f"{sizing.params_label} @ {sizing.quantization} "
            f"(~{sizing.approx_weights_gb} GB of {sizing.usable_memory_gb} GB usable)",
        ),
    ]

    if info.disk_free_gb is not None:
        low = info.disk_free_gb < 20.0
        checks.append(Check("disk free", "warn" if low else "ok", f"{info.disk_free_gb:.0f} GB"))

    # Sizing caveats are warnings, not failures: the machine works, it just will not
    # do everything the memory table implies.
    checks.extend(Check("note", "warn", note) for note in sizing.notes)
    return checks


def _check_optional_deps() -> list[Check]:
    """Report which not-yet-required packages are installed."""
    checks: list[Check] = []
    for module, why in _OPTIONAL_DEPS:
        found = importlib.util.find_spec(module) is not None
        checks.append(
            Check(module, "ok" if found else "warn", "installed" if found else f"absent — {why}")
        )
    return checks


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report the environment, per brief §5."""
    platform = get_platform()
    checks: list[Check] = [
        Check("arc", "ok", f"version {__version__}"),
        Check("python", "ok", sys.version.split()[0]),
        Check(
            "platform",
            "ok" if platform.implemented else "fail",
            platform.name if platform.implemented else f"{platform.name} is not implemented yet",
        ),
    ]
    checks += _check_config(args.config_dir)
    checks += _check_paths()
    try:
        checks += _hardware_checks(*refresh())
    except ArcError as exc:
        checks.append(Check("hardware", "fail", str(exc)))
    checks += _check_optional_deps()

    if not args.json:
        print(f"\nARC doctor — {platform_name()}\n")
    _render(checks, as_json=args.json)

    code = _exit_code(checks)
    if not args.json:
        failed = sum(1 for c in checks if c.status == "fail")
        warned = sum(1 for c in checks if c.status == "warn")
        print(f"\n{len(checks)} checks, {failed} failed, {warned} warnings\n")
    return code


def cmd_probe(args: argparse.Namespace) -> int:
    """Re-probe the machine and rewrite ``hardware.json``."""
    try:
        info, sizing, path = refresh()
    except ArcError as exc:
        print(f"probe failed: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {"hardware": info.to_dict(), "recommendation": sizing.to_dict()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    _render(_hardware_checks(info, sizing, path), as_json=False)
    print(f"\nwrote {path}\n")
    return 0


def cmd_kill(args: argparse.Namespace) -> int:
    """SIGKILL every registered ARC process tree.

    Runs entirely from this process's own state — it never asks a possibly-wedged
    agent to cooperate, which is the whole point of the design in ``killswitch.py``.
    """
    switch = KillSwitch()
    reaped = switch.reap_stale()
    entries = switch.registered()

    if args.dry_run:
        names = ", ".join(f"{e.name}:{e.pid}" for e in entries) or "none"
        print(f"[dry-run] would kill {len(entries)} process(es): {names}")
        return 0

    killed = switch.kill_all()

    if args.json:
        print(json.dumps({"killed": killed, "reaped_stale": reaped}, indent=2))
        return 0

    if not killed and not reaped:
        print("no ARC processes registered")
    else:
        if killed:
            print(f"killed {len(killed)} process(es): {', '.join(str(p) for p in killed)}")
        if reaped:
            print(f"cleaned up {reaped} stale PID file(s)")
    return 0


def cmd_version(args: argparse.Namespace) -> int:
    """Print the version."""
    if args.json:
        print(json.dumps({"version": __version__}))
    else:
        print(f"arc {__version__}")
    return 0


def _require_config(args: argparse.Namespace) -> Config:
    """Load config, or raise with a clear message. Used by commands that cannot proceed."""
    return Config.load(directory=args.config_dir)


def cmd_model(args: argparse.Namespace) -> int:
    """Manage models: list, pull, use, remove."""
    from arc.model import manager, router

    config = _require_config(args)

    if args.model_command == "list":
        statuses = manager.status_for(config)
        if args.json:
            print(json.dumps([s.to_dict() for s in statuses], indent=2))
            return 0
        if not statuses:
            print("no models in the registry (config/models.yaml)")
            return 0

        for status in statuses:
            entry = status.entry
            marks = "".join(
                (
                    "*" if status.active_for else " ",
                    "↓" if status.downloaded else " ",
                )
            )
            choice = router.choose_backend(entry, config)
            backend = choice.backend
            if choice.fallback_from:
                backend = f"{choice.backend} (fallback from {choice.fallback_from})"
            size = (
                f"{status.size_on_disk_gb} GB on disk"
                if status.size_on_disk_gb is not None
                else f"~{entry.approx_size_gb} GB to download"
                if entry.approx_size_gb
                else "size unknown"
            )
            roles = f" [{', '.join(status.active_for)}]" if status.active_for else ""
            print(f"{marks} {entry.key}{roles}")
            print(f"     {entry.repo}")
            print(f"     {backend} · {entry.licence} (verified {entry.licence_verified}) · {size}")
        print("\n* active   ↓ downloaded")
        return 0

    if args.model_command == "pull":
        path = manager.pull(config, args.key, force=args.force)
        print(f"model {args.key!r} ready at {path}")
        return 0

    if args.model_command == "use":
        target = manager.use(config, args.key, args.role)
        print(f"{args.role} model set to {args.key!r} (written to {target})")
        return 0

    if args.model_command == "remove":
        path = manager.remove(config, args.key)
        print(f"removed {path}")
        return 0

    raise ConfigError(f"unknown model subcommand {args.model_command!r}")


def _memory_service(config: Config, audit: AuditLogger | None = None) -> Any:
    """Build the memory service, or raise a clear error if it is unavailable."""
    from arc.memory.service import MemoryService

    return MemoryService.from_config(config, audit=audit)


def cmd_memory(args: argparse.Namespace) -> int:
    """Inspect and manage long-term memory."""
    config = _require_config(args)
    memory = _memory_service(config, _audit_logger(config))

    try:
        if args.memory_command == "stats":
            stats = memory.stats()
            if args.json:
                print(json.dumps(stats, indent=2))
            else:
                print(f"\n{stats['live_memories']} live memories in {stats['path']}")
                print(f"  size        {stats['size_mb']} MB")
                print(f"  embedder    {stats['embedder']} ({stats['dimension']}d)")
                print(f"  embedded    {stats['embedded']}")
                print(f"  superseded  {stats['superseded']}")
                for layer, count in sorted(stats["by_layer"].items()):
                    print(f"  {layer:11} {count}")
                print(f"  entities    {stats['entities']} ({stats['relations']} relations)")
                print(f"  sessions    {stats['sessions']}\n")
            return 0

        if args.memory_command == "search":
            hits = memory.retriever.search(args.query, limit=args.limit)
            if args.json:
                print(json.dumps([h.to_dict() for h in hits], indent=2))
                return 0
            if not hits:
                print("no matches")
                return 0
            for hit in hits:
                record = hit.record
                strategies = ", ".join(hit.sources)
                print(f"\n[{record.id}] {record.layer}/{record.kind}  score {hit.score:.4f}")
                print(f"  {record.content}")
                print(f"  found by: {strategies} · {record.occurred_at[:16]}")
                if record.source_url:
                    print(f"  source: {record.source_url}")
            print()
            return 0

        if args.memory_command == "add":
            memory_id = memory.semantic.add_fact(args.text, source="user")
            print(f"stored as memory {memory_id}")
            return 0

        if args.memory_command == "forget":
            record = memory.store.get(args.id)
            if record is None:
                print(f"no memory with id {args.id}", file=sys.stderr)
                return 1
            if not args.yes:
                print(f"would permanently delete [{args.id}] {record.content[:70]}")
                print("re-run with --yes to confirm")
                return 0
            memory.store.forget(args.id)
            print(f"deleted memory {args.id}")
            return 0

        if args.memory_command == "export":
            records = [r.to_dict() for r in memory.store.iter_all()]
            payload = json.dumps(records, indent=2)
            if args.output:
                Path(args.output).write_text(payload, encoding="utf-8")
                print(f"exported {len(records)} memories to {args.output}")
            else:
                print(payload)
            return 0

        if args.memory_command == "consolidate":
            report = memory.consolidator.run(dry_run=args.dry_run)
            if args.json:
                print(json.dumps(report.to_dict(), indent=2))
            else:
                prefix = "[dry-run] would have " if args.dry_run else ""
                print(
                    f"{prefix}deduped {report.deduped}, decayed {report.decayed}, "
                    f"promoted {report.promoted}, pruned {report.pruned}"
                )
            return 0

        raise ConfigError(f"unknown memory subcommand {args.memory_command!r}")
    finally:
        memory.close()


def cmd_tools(args: argparse.Namespace) -> int:
    """List the tools available to the agent."""
    from arc.tools import registry as tool_registry

    described = tool_registry.describe()
    if args.json:
        print(json.dumps(described, indent=2))
        return 0

    by_category: dict[str, list[dict[str, Any]]] = {}
    for entry in described:
        by_category.setdefault(entry["category"], []).append(entry)

    print(f"\n{len(described)} tools available\n")
    for category in sorted(by_category):
        print(f"{category}:")
        for entry in by_category[category]:
            marker = "*" if entry["mutating"] else " "
            required = entry["parameters"].get("required", [])
            signature = ", ".join(required)
            print(f"  {marker} {entry['name']}({signature})")
            print(f"      {entry['description']}")
    print("\n* mutating — skipped under --dry-run\n")
    return 0


def cmd_do(args: argparse.Namespace) -> int:
    """Run a multi-step task with tools."""
    from arc.agent.loop import Agent, Step
    from arc.model import router
    from arc.model.registry import resolve
    from arc.tools import registry as tool_registry

    config = _require_config(args)
    entry = resolve(config, "chat")
    audit = _audit_logger(config)

    if args.dry_run:
        print("[dry-run] mutating tools will be skipped; read-only tools still run\n")

    print(f"loading {entry.key}...", file=sys.stderr)
    model = router.load_model(config, "chat")

    memory = None
    if not args.no_memory and config.get("memory.enabled", True):
        with contextlib.suppress(ArcError):
            memory = _memory_service(config, audit)

    def show(step: Step) -> None:
        observation = step.observation
        status = "ok" if observation and observation.ok else "failed"
        print(f"  [{step.number}] {step.tool} -> {status}", file=sys.stderr)
        if step.repairs:
            # Surfaced because repeated repairs mean the model is emitting malformed
            # output, which is worth knowing before blaming the agent.
            print(f"      (parser repaired: {', '.join(step.repairs)})", file=sys.stderr)

    agent = Agent(
        model,
        tool_registry,
        memory=memory,
        audit=audit,
        max_steps=args.max_steps,
        dry_run=args.dry_run,
        on_step=None if args.json else show,
    )

    try:
        result = agent.run(args.task)
    finally:
        if memory is not None:
            memory.close()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    print(f"\n{result.answer}\n")
    if result.exhausted:
        print(f"(stopped after {args.max_steps} steps without finishing)", file=sys.stderr)
    return 0


def cmd_research(args: argparse.Namespace) -> int:
    """Research a question on the web and remember what was learned."""
    from arc.model import router
    from arc.web.research import Researcher

    config = _require_config(args)
    audit = _audit_logger(config)

    memory = None
    if not args.no_memory and config.get("memory.enabled", True):
        with contextlib.suppress(ArcError):
            memory = _memory_service(config, audit)

    print("loading model...", file=sys.stderr)
    model = router.load_model(config, "chat")
    researcher = Researcher(model, memory, max_pages=args.max_pages)

    try:
        result = researcher.research(args.query, use_memory=not args.fresh)
    finally:
        if memory is not None:
            memory.close()

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    origin = "from memory" if result.from_memory else f"read {result.pages_read} page(s)"
    print(f"\n{result.summary}\n")
    print(f"— {origin}", file=sys.stderr)
    for source in dict.fromkeys(result.sources):
        print(f"  {source}", file=sys.stderr)
    if result.memory_ids:
        print(f"  stored {len(result.memory_ids)} facts in memory", file=sys.stderr)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    """Talk to the local model."""
    from arc.interface import chat
    from arc.model import router
    from arc.model.registry import resolve

    config = _require_config(args)
    entry = resolve(config, "chat")
    choice = router.choose_backend(entry, config)

    if args.dry_run:
        print(f"[dry-run] would load {entry.key!r} via {choice.backend} ({choice.reason})")
        return 0

    print(f"loading {entry.key} via {choice.backend}...", file=sys.stderr)
    model = router.load_model(config, "chat")
    audit = _audit_logger(config)

    memory = None
    if not args.no_memory and config.get("memory.enabled", True):
        try:
            memory = _memory_service(config, audit)
        except ArcError as exc:
            # Chat without memory beats no chat at all. Reported rather than swallowed
            # so a broken database does not look like a feature that silently vanished.
            print(f"memory unavailable ({exc}); continuing without it", file=sys.stderr)

    try:
        return chat.run(
            model,
            config,
            backend=choice.backend,
            system_prompt=args.system,
            audit=audit,
            memory=memory,
        )
    finally:
        if memory is not None:
            memory.close()


def _build_parser() -> argparse.ArgumentParser:
    """Construct the CLI."""
    parser = argparse.ArgumentParser(
        prog="arc",
        description="ARC — a local-first, fully-owned AI assistant.",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="log intended actions without executing them",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="override the configured log level (debug, info, warning, error)",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=None,
        help="load configuration from this directory instead of ./config",
    )

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report this machine's environment").set_defaults(func=cmd_doctor)
    probe = sub.add_parser("probe", help="re-probe hardware and rewrite hardware.json")
    probe.add_argument(
        "--refresh",
        action="store_true",
        help="accepted for symmetry; probe always re-probes",
    )
    probe.set_defaults(func=cmd_probe)
    sub.add_parser("kill", help="SIGKILL every registered ARC process").set_defaults(func=cmd_kill)
    sub.add_parser("version", help="print the version").set_defaults(func=cmd_version)

    model = sub.add_parser("model", help="manage local models")
    model.set_defaults(func=cmd_model)
    model_sub = model.add_subparsers(dest="model_command", required=True)

    model_sub.add_parser("list", help="show the registry and what is downloaded")

    pull = model_sub.add_parser("pull", help="download a model's weights")
    pull.add_argument("key", help="registry key, as shown by `arc model list`")
    pull.add_argument("--force", action="store_true", help="re-download even if already present")

    use = model_sub.add_parser("use", help="select the active model for a role")
    use.add_argument("key", help="registry key")
    use.add_argument(
        "--role",
        default="chat",
        choices=("chat", "vision", "embedding"),
        help="which role to set (default: chat)",
    )

    remove = model_sub.add_parser("remove", help="delete a model's local weights")
    remove.add_argument("key", help="registry key")

    mem = sub.add_parser("memory", help="inspect and manage long-term memory")
    mem.set_defaults(func=cmd_memory)
    mem_sub = mem.add_subparsers(dest="memory_command", required=True)

    mem_sub.add_parser("stats", help="summarise what is stored")

    search = mem_sub.add_parser("search", help="hybrid search across all layers")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=10)

    add = mem_sub.add_parser("add", help="store a fact directly")
    add.add_argument("text")

    forget = mem_sub.add_parser("forget", help="permanently delete a memory")
    forget.add_argument("id", type=int)
    forget.add_argument("--yes", action="store_true", help="confirm; without it this is a dry run")

    export = mem_sub.add_parser("export", help="dump every memory as JSON")
    export.add_argument("--output", default=None, help="write to a file instead of stdout")

    consolidate = mem_sub.add_parser("consolidate", help="run dedupe, decay, and promotion passes")
    consolidate.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="report what would change without changing it",
    )

    tools = sub.add_parser("tools", help="list the tools available to the agent")
    tools.set_defaults(func=cmd_tools)

    do = sub.add_parser("do", help="run a multi-step task using tools")
    do.add_argument("task", help="what to do, in plain language")
    do.add_argument(
        "--max-steps",
        type=int,
        default=12,
        dest="max_steps",
        help="give up after this many tool calls (default: 12)",
    )
    do.add_argument("--no-memory", action="store_true", help="run without long-term memory")
    do.set_defaults(func=cmd_do)

    research = sub.add_parser("research", help="research a question on the web")
    research.add_argument("query", help="what to find out")
    research.add_argument(
        "--max-pages", type=int, default=3, dest="max_pages", help="how many pages to read"
    )
    research.add_argument(
        "--fresh",
        action="store_true",
        help="ignore stored answers and go to the network",
    )
    research.add_argument("--no-memory", action="store_true", help="do not store what is learned")
    research.set_defaults(func=cmd_research)

    chat = sub.add_parser("chat", help="talk to the local model")
    chat.add_argument("--system", default=None, help="system prompt for the session")
    chat.add_argument(
        "--no-memory",
        action="store_true",
        help="run without long-term memory for this session",
    )
    chat.set_defaults(func=cmd_chat)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch. Returns a process exit code.

    Returning rather than calling ``sys.exit`` keeps the whole CLI testable in-process.
    """
    args = _build_parser().parse_args(argv)

    from arc import log

    # A broken config must not stop `arc doctor` from starting — reporting exactly
    # that breakage is its job. Other commands fall back to logging defaults.
    config: Config | None = None
    with contextlib.suppress(ArcError):
        config = Config.load(directory=args.config_dir)

    level = args.log_level or (config.get("logging.level", "info") if config else "info")
    log.setup(
        level=str(level),
        console=bool(config.get("logging.console", True)) if config else True,
        console_level=str(config.get("logging.console_level", "warning")) if config else "warning",
        to_file=bool(config.get("logging.to_file", True)) if config else True,
    )

    # Every invocation is recorded, not just tool calls the agent makes later. When
    # something has gone wrong at 2am, "which commands did I run" is part of the
    # answer, and an audit log with holes in it invites false confidence (§0.3).
    audit = _audit_logger(config)

    try:
        result: int = args.func(args)
    except ArcError as exc:
        _record(audit, args, status="error", error=str(exc))
        print(f"arc: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        _record(audit, args, status="error", error="interrupted")
        return 130

    _record(audit, args, status="dry_run" if args.dry_run else "ok", exit_code=result)
    return result


def _audit_logger(config: Config | None) -> AuditLogger | None:
    """Build an audit logger, or None if auditing is disabled or unavailable.

    Returns None rather than raising: failing to write the audit log is fatal *during*
    an agent run, but refusing to let ``arc doctor`` start because ``~/.arc`` is
    read-only would hide the very diagnosis the user is asking for.
    """
    if config is not None and not config.get("audit.enabled", True):
        return None
    with contextlib.suppress(ArcError):
        return AuditLogger(
            fsync=bool(config.get("audit.fsync", False)) if config else False,
            max_field_chars=int(config.get("audit.max_field_chars", 4000)) if config else 4000,
        )
    return None


def _record(
    audit: AuditLogger | None,
    args: argparse.Namespace,
    *,
    status: Any,
    error: str | None = None,
    exit_code: int | None = None,
) -> None:
    """Append one CLI invocation to the audit log, ignoring audit failures."""
    if audit is None:
        return
    with contextlib.suppress(ArcError):
        audit.record(
            f"cli.{args.command}",
            status=status,
            tool="cli",
            args={"command": args.command, "json": args.json, "dry_run": args.dry_run},
            result={"exit_code": exit_code} if exit_code is not None else None,
            error=error,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    sys.exit(main())
