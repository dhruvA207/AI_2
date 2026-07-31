"""Tests for the web UI and its static serving.

Like ``test_server.py`` these never bind a port or load a model. They cover the two
things that fail silently rather than loudly: a path-traversal guard that stops
containing traversal, and static assets that stop being packaged.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from arc.interface import server

WEBUI = Path("arc/interface/webui")


# ── Packaging ───────────────────────────────────────────────────────────────────


def test_every_shipped_file_has_a_servable_type() -> None:
    """A file the server cannot name a content type for is a 404 nobody expects.

    ``_CONTENT_TYPES`` is a closed allow-list, so adding e.g. a .png to the directory
    without adding its type ships an asset that silently never loads.
    """
    for path in WEBUI.iterdir():
        if path.is_file():
            assert path.suffix in server._CONTENT_TYPES, f"{path.name} has no content type"


def test_package_data_covers_the_webui() -> None:
    """The UI has no build step, so it ships as package data.

    Without this an installed ARC serves 404 for the UI while a source checkout works
    perfectly — the worst kind of bug, because it never reproduces in development.
    """
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    globs = data["tool"]["setuptools"]["package-data"]["arc"]
    covered = {Path(g).suffix for g in globs}
    shipped = {p.suffix for p in WEBUI.iterdir() if p.is_file()}
    assert shipped <= covered, f"not packaged: {shipped - covered}"


def test_entry_point_exists() -> None:
    assert (WEBUI / "index.html").is_file()


# ── Local-first ─────────────────────────────────────────────────────────────────


def test_ui_references_no_external_hosts() -> None:
    """ARC is local-first and the server is loopback-only.

    A CDN link would quietly make the UI depend on the network and leak that it is
    running, so the absence is asserted rather than assumed.
    """
    for path in WEBUI.iterdir():
        if not path.is_file():
            continue
        hits = re.findall(r"https?://[^\s\"')]+", path.read_text(encoding="utf-8"))
        assert not hits, f"{path.name} references {hits}"


# ── Path traversal ──────────────────────────────────────────────────────────────


def _resolves_inside(relative: str) -> bool:
    """Mirror of the containment check in ``_Handler._serve_static``."""
    target = (server.WEBUI_DIR / relative).resolve()
    root = server.WEBUI_DIR.resolve()
    return root in target.parents or target == root


def test_traversal_escapes_are_rejected() -> None:
    """The server runs with ARC's own unrestricted privileges (§0.3), so escaping the
    UI directory would expose the whole filesystem. ``..`` only normalises away after
    resolution, which is why the check happens on the resolved path."""
    for attempt in (
        "../../config.py",
        "../../../../etc/passwd",
        "./../server.py",
        "sub/../../server.py",
    ):
        assert not _resolves_inside(attempt), f"{attempt} escaped the UI directory"


def test_ordinary_names_are_accepted() -> None:
    for name in ("index.html", "app.js", "orb.js", "app.css"):
        assert _resolves_inside(name)


# ── Routing ─────────────────────────────────────────────────────────────────────


def test_health_is_still_json_after_root_became_the_ui() -> None:
    """``/`` used to return status JSON and now returns the UI, so the machine-readable
    status must remain reachable somewhere stable."""
    source = Path("arc/interface/server.py").read_text(encoding="utf-8")
    assert '"/health", "/status"' in source
    assert 'route.path == "/" or route.path.startswith("/ui/")' in source


def test_stream_endpoint_is_registered() -> None:
    source = Path("arc/interface/server.py").read_text(encoding="utf-8")
    assert '"/chat/stream"' in source


def test_prompt_assembly_is_shared_not_duplicated() -> None:
    """``/chat`` and ``/chat/stream`` must build the prompt the same way.

    The memory-provenance guidance in ``_compose`` is load-bearing — without it the
    model copies the bracketed markers into its replies — and two copies would drift.
    """
    source = Path("arc/interface/server.py").read_text(encoding="utf-8")
    assert source.count("def _compose(") == 1
    assert source.count("self._compose(") == 2
