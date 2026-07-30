"""Tests for the local API server.

The server is what makes §7's "model warm-loading" real, so the tests focus on the
resident runtime's laziness and thread-safety rather than on HTTP plumbing. They never
bind a port or load a model.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from arc.config import Config
from arc.interface import server


@pytest.fixture
def config() -> Config:
    return Config.load(use_env=False)


# ── Binding ─────────────────────────────────────────────────────────────────────


def test_server_binds_loopback_only() -> None:
    """ARC has unrestricted access to this machine (§0.3), so an endpoint reaching it
    must not be reachable from the network. Asserted rather than trusted, because a
    bind address is exactly what gets loosened 'temporarily'."""
    assert server.BIND_HOST == "127.0.0.1"


def test_bind_host_is_not_configurable() -> None:
    """There is deliberately no config key for this."""
    text = Path("arc/interface/server.py").read_text(encoding="utf-8")
    assert 'config.get("server' not in text
    assert '"0.0.0.0"' not in text


# ── Runtime laziness ────────────────────────────────────────────────────────────


def test_nothing_is_loaded_until_asked(config: Config) -> None:
    runtime = server.Runtime(config)
    status = runtime.status()
    assert status["model_loaded"] is False
    assert status["memory_loaded"] is False


def test_status_reports_uptime_and_version(config: Config) -> None:
    status = server.Runtime(config).status()
    assert status["uptime_seconds"] >= 0
    assert status["version"]


def test_memory_is_none_when_disabled(tmp_path: Path) -> None:
    (tmp_path / "default.yaml").write_text("memory:\n  enabled: false\n", encoding="utf-8")
    runtime = server.Runtime(Config.load(directory=tmp_path, use_env=False))
    assert runtime.memory is None


def test_concurrent_access_loads_once(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two requests arriving together must not both spend two seconds loading the same
    model, and the second must not proceed with a half-initialised one."""
    loads = []

    def slow_load(*_a: object, **_k: object) -> object:
        import time

        loads.append(1)
        time.sleep(0.2)
        return object()

    import arc.model.router as router

    monkeypatch.setattr(router, "load_model", slow_load)

    runtime = server.Runtime(config)
    threads = [threading.Thread(target=lambda: runtime.model) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(loads) == 1


# ── Endpoint discovery ──────────────────────────────────────────────────────────


def test_endpoint_roundtrip(arc_home_tmp: Path) -> None:
    server.write_endpoint(1234)
    found = server.running_endpoint()
    assert found == ("127.0.0.1", 1234)


def test_no_endpoint_when_none_written(arc_home_tmp: Path) -> None:
    assert server.running_endpoint() is None


def test_stale_endpoint_is_cleaned_up(arc_home_tmp: Path) -> None:
    """A crashed server leaves its file behind; without the liveness check the CLI
    would hang trying to reach a process that no longer exists."""
    server.endpoint_file().parent.mkdir(parents=True, exist_ok=True)
    server.endpoint_file().write_text(
        json.dumps({"host": "127.0.0.1", "port": 1234, "pid": 999999}), encoding="utf-8"
    )
    assert server.running_endpoint() is None
    assert not server.endpoint_file().exists()


def test_corrupt_endpoint_file_is_cleaned_up(arc_home_tmp: Path) -> None:
    server.endpoint_file().parent.mkdir(parents=True, exist_ok=True)
    server.endpoint_file().write_text("{ not json", encoding="utf-8")
    assert server.running_endpoint() is None


def test_clear_endpoint_is_safe_when_absent(arc_home_tmp: Path) -> None:
    server.clear_endpoint()


# ── Memory rendering, which the server shares with the REPL ─────────────────────


def test_rendered_memories_carry_no_imitable_markers() -> None:
    """Regression: memories rendered as "- fact [episodic, 2026-07-30]" got copied by
    the model into its own replies — asked for "pong" it answered "pong [episodic,
    2026-07-30]", and because that was stored and recalled, the next turn produced two
    markers. Telling it not to did not help; the format had to change."""
    from arc.memory.retrieval import Hit
    from arc.memory.store import MemoryRecord
    from arc.memory.working import WorkingMemory
    from tests.fakes import FakeModel

    record = MemoryRecord(
        id=1,
        layer="episodic",
        kind="turn:user",
        content="Dhruv likes Rust",
        occurred_at="2026-07-30T00:00:00",
        created_at="2026-07-30T00:00:00",
        salience=1.0,
        confidence=1.0,
    )
    rendered = WorkingMemory.for_model(FakeModel()).render_memories([Hit(record=record, score=1.0)])
    assert "[episodic" not in rendered
    assert "Dhruv likes Rust" in rendered


def test_source_urls_survive_for_citation() -> None:
    """§4.4 requires ARC to cite where a fact came from; it cannot cite what it was
    never shown."""
    from arc.memory.retrieval import Hit
    from arc.memory.store import MemoryRecord
    from arc.memory.working import WorkingMemory
    from tests.fakes import FakeModel

    record = MemoryRecord(
        id=1,
        layer="semantic",
        kind="fact",
        content="Rust prevents data races",
        occurred_at="2026-07-30T00:00:00",
        created_at="2026-07-30T00:00:00",
        salience=1.0,
        confidence=0.6,
        source_url="https://doc.rust-lang.org/",
    )
    rendered = WorkingMemory.for_model(FakeModel()).render_memories([Hit(record=record, score=1.0)])
    assert "https://doc.rust-lang.org/" in rendered
    assert "uncertain" in rendered
