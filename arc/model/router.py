"""Backend selection.

The router answers one question: given this machine and this registry entry, which
backend actually runs it? It reads ``~/.arc/hardware.json`` rather than probing, per
ADR-002 — the probe already ran at startup and everything downstream reads its output.

Selection is deliberately explainable. ``choose_backend`` returns *why* it chose, so
``arc doctor`` and ``arc model list`` can show the reasoning instead of presenting a
decision the user has to reverse-engineer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from arc.config import Config
from arc.errors import ModelError
from arc.hardware import load_hardware_file
from arc.log import get_logger
from arc.model.base import LanguageModel
from arc.model.registry import ModelEntry, resolve
from arc.paths import models_dir

_log = get_logger(__name__)

#: Which accelerator each backend needs. A backend is only eligible if the machine
#: reports at least one of these in hardware.json's `accelerators`.
_BACKEND_REQUIREMENTS: dict[str, frozenset[str]] = {
    "mlx": frozenset({"mlx"}),
    "llamacpp": frozenset({"metal", "cuda", "cpu"}),
    "vllm": frozenset({"cuda"}),
    "transformers": frozenset({"cuda", "mps", "cpu"}),
    "custom": frozenset({"mlx", "metal", "cuda", "cpu"}),
}


@dataclass(frozen=True, slots=True)
class BackendChoice:
    """A selected backend and the reasoning behind it."""

    backend: str
    reason: str
    #: Set when the entry's preferred backend was unavailable and we fell back.
    fallback_from: str | None = None


def available_accelerators() -> list[str]:
    """Return this machine's accelerators from hardware.json.

    Falls back to CPU-only if the file is missing or unreadable, which keeps the router
    usable on a fresh install before the first probe has run.
    """
    data = load_hardware_file()
    if data is None:
        _log.warning("hardware.json missing; assuming CPU only")
        return ["cpu"]
    hardware = data.get("hardware", {})
    accelerators = hardware.get("accelerators")
    if not isinstance(accelerators, list) or not accelerators:
        return ["cpu"]
    return [str(a) for a in accelerators]


def choose_backend(entry: ModelEntry, config: Config) -> BackendChoice:
    """Pick a backend for ``entry`` on this machine.

    Order of precedence:

    1. ``models.router.force_backend`` — an explicit override always wins, including
       over a machine that cannot support it. Being able to force a wrong answer is
       what makes an override useful for debugging.
    2. The entry's own ``backend``, if this machine has the accelerator for it.
    3. llama.cpp, which runs anywhere.
    """
    forced = config.get("models.router.force_backend")
    if forced:
        return BackendChoice(backend=str(forced), reason="forced by models.router.force_backend")

    if not config.get("models.router.auto_select_backend", True):
        return BackendChoice(backend=entry.backend, reason="auto-select disabled in config")

    accelerators = set(available_accelerators())
    required = _BACKEND_REQUIREMENTS.get(entry.backend, frozenset())

    if required & accelerators:
        matched = sorted(required & accelerators)[0]
        return BackendChoice(
            backend=entry.backend,
            reason=f"{entry.backend} supported ({matched} available)",
        )

    # llama.cpp runs on plain CPU, so it is always a valid destination.
    return BackendChoice(
        backend="llamacpp",
        reason=(
            f"{entry.backend} needs one of {sorted(required)}, machine has {sorted(accelerators)}"
        ),
        fallback_from=entry.backend,
    )


def model_path(entry: ModelEntry) -> Path:
    """Return the local cache path for an entry's weights."""
    return models_dir() / entry.key


def _source_for(entry: ModelEntry) -> str:
    """Return the local weights directory if present, otherwise the hub repo id.

    Preferring the local copy is what makes ``~/.arc`` self-contained (§4.2) and keeps
    ARC working offline. Falling back to the repo id means a model that was never
    pulled still loads, downloading on first use rather than erroring.
    """
    path = model_path(entry)
    if path.is_dir() and any(path.glob("*.safetensors")):
        return str(path)
    return entry.repo


def load_model(config: Config, role: str = "chat") -> LanguageModel:
    """Resolve the active model for a role, pick a backend, and load it.

    The single entry point callers use. Backend modules are imported lazily here and
    nowhere else, so a machine missing MLX can still run ``arc model list``.
    """
    entry = resolve(config, role)
    choice = choose_backend(entry, config)

    if choice.fallback_from:
        _log.info(
            "falling back to %s: %s", choice.backend, choice.reason, extra={"model": entry.key}
        )

    if choice.backend == "mlx":
        from arc.model.mlx_backend import MLXModel

        return MLXModel(
            _source_for(entry),
            context_length=entry.context_length,
            capabilities=entry.capabilities,
            name=entry.key,
        )

    if choice.backend == "llamacpp":
        from arc.model.llamacpp import LlamaCppModel

        path = model_path(entry)
        if entry.filename:
            path = path / entry.filename
        return LlamaCppModel(
            path,
            context_length=entry.context_length,
            capabilities=entry.capabilities,
            name=entry.key,
        )

    raise ModelError(
        f"backend {choice.backend!r} is not implemented yet "
        f"(selected for {entry.key!r}: {choice.reason})"
    )
