"""Tests for control sessions, the indicator, and input gating.

The properties under test are safety properties, not features: ARC must never deliver a
synthetic event without a visible indicator, and must stop the instant the user takes
the pointer back. Both are enforced in the session layer so a new tool cannot bypass
them by forgetting to check.
"""

from __future__ import annotations

import time

import pytest

from arc.control import input as ctl
from arc.control import session as cs
from arc.errors import ControlError


@pytest.fixture(autouse=True)
def clean_sessions() -> None:
    """Never leave a session — or an indicator — running between tests."""
    cs.stop("test setup")
    yield
    cs.stop("test teardown")


# ── Gating ──────────────────────────────────────────────────────────────────────


def test_no_session_means_no_control() -> None:
    with pytest.raises(ControlError, match="does not currently have input control"):
        cs.require()


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (ctl.move_to, (100.0, 100.0)),
        (ctl.click, ()),
        (ctl.drag, (0.0, 0.0, 10.0, 10.0)),
        (ctl.scroll, (1,)),
        (ctl.type_text, ("hello",)),
        (ctl.press, ("c",)),
    ],
)
def test_every_input_function_refuses_without_a_session(function, args) -> None:
    """The enforcement point. If any of these could run unguarded, ARC could move the
    mouse with no indicator on screen."""
    with pytest.raises(ControlError):
        function(*args)


def test_input_refuses_after_release() -> None:
    session = cs.ControlSession(reason="t", show_overlay=False, watch_for_takeover=False)
    session.acquire()
    session.release("test")
    with pytest.raises(ControlError, match="released by"):
        session.check()


# ── Lifecycle ───────────────────────────────────────────────────────────────────


def test_session_reports_its_state() -> None:
    session = cs.ControlSession(
        reason="doing a thing", show_overlay=False, watch_for_takeover=False
    )
    session.acquire()
    state = session.state.to_dict()
    assert state["active"] is True
    assert state["reason"] == "doing a thing"
    session.release("done")
    assert session.state.to_dict()["active"] is False


def test_release_records_who_took_it_back() -> None:
    """Worth knowing later whether ARC finished or you interrupted it."""
    session = cs.ControlSession(show_overlay=False, watch_for_takeover=False)
    session.acquire()
    session.release("user moved the mouse")
    assert session.state.released_by == "user moved the mouse"


def test_release_is_idempotent() -> None:
    session = cs.ControlSession(show_overlay=False, watch_for_takeover=False)
    session.acquire()
    session.release("once")
    session.release("twice")
    assert session.state.released_by == "once"


def test_acquire_is_idempotent() -> None:
    session = cs.ControlSession(show_overlay=False, watch_for_takeover=False)
    session.acquire()
    started = session.state.started_at
    session.acquire()
    assert session.state.started_at == started


def test_context_manager_releases_on_exit() -> None:
    with cs.ControlSession(show_overlay=False, watch_for_takeover=False) as session:
        assert session.state.active
    assert not session.state.active


def test_context_manager_releases_on_exception() -> None:
    """Control must not survive a crash — that is how it would get stuck on."""
    session = cs.ControlSession(show_overlay=False, watch_for_takeover=False)
    with pytest.raises(RuntimeError), session:
        raise RuntimeError("boom")
    assert not session.state.active


def test_module_level_session_is_shared() -> None:
    session = cs.start(reason="shared", overlay=False)
    assert cs.current() is session
    assert cs.require() is session
    cs.stop("done")
    assert cs.current() is None


def test_starting_twice_returns_the_same_session() -> None:
    first = cs.start(reason="a", overlay=False)
    second = cs.start(reason="b", overlay=False)
    assert first is second


# ── Takeover ────────────────────────────────────────────────────────────────────


def test_pointer_drift_releases_control() -> None:
    """The property that matters most: moving the mouse yourself ends the session."""
    session = cs.ControlSession(show_overlay=False, watch_for_takeover=True)
    session.acquire()
    session.note_pointer((100.0, 100.0))

    # Stand in for the real pointer being somewhere ARC did not put it.
    original = cs.pointer_position
    cs.pointer_position = lambda: (400.0, 400.0)  # type: ignore[assignment]
    try:
        deadline = time.monotonic() + 2.0
        while session.state.active and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        cs.pointer_position = original  # type: ignore[assignment]

    assert not session.state.active
    assert "mouse" in session.state.released_by


def test_arcs_own_movement_is_not_a_takeover() -> None:
    """Otherwise the first thing ARC does would end its own session."""
    session = cs.ControlSession(show_overlay=False, watch_for_takeover=True)
    session.acquire()

    original = cs.pointer_position
    cs.pointer_position = lambda: (300.0, 300.0)  # type: ignore[assignment]
    try:
        session.note_pointer((300.0, 300.0))  # ARC says: I put it there
        time.sleep(0.35)
        assert session.state.active
    finally:
        cs.pointer_position = original  # type: ignore[assignment]
        session.release("test")


def test_small_jitter_does_not_release() -> None:
    """A trackpad reports sub-pixel motion constantly; releasing on that would make
    control unusable."""
    session = cs.ControlSession(show_overlay=False, watch_for_takeover=True)
    session.acquire()

    original = cs.pointer_position
    cs.pointer_position = lambda: (302.0, 301.0)  # type: ignore[assignment]
    try:
        session.note_pointer((300.0, 300.0))
        time.sleep(0.35)
        assert session.state.active
    finally:
        cs.pointer_position = original  # type: ignore[assignment]
        session.release("test")


def test_threshold_is_a_deliberate_nudge_not_a_twitch() -> None:
    assert 5 < cs.TAKEOVER_THRESHOLD < 40


# ── The indicator ───────────────────────────────────────────────────────────────


def test_overlay_starts_and_stops() -> None:
    session = cs.ControlSession(show_overlay=True, watch_for_takeover=False)
    session.acquire()
    assert session._overlay is not None
    assert session._overlay.poll() is None  # still running

    session.release("test")
    assert session._overlay is None


def test_overlay_does_not_outlive_an_exception() -> None:
    """A stuck glow with nothing behind it would be worse than no glow at all."""
    session = cs.ControlSession(show_overlay=True, watch_for_takeover=False)
    with pytest.raises(RuntimeError), session:
        raise RuntimeError("boom")
    assert session._overlay is None


# ── Being stoppable ─────────────────────────────────────────────────────────────


def test_holding_control_registers_with_the_kill_switch() -> None:
    """``arc-kill`` and the abort phrase both work off these PID files.

    A process holding the mouse without one is a process the documented ways of
    stopping it cannot see.
    """
    from arc.audit.killswitch import KillSwitch

    session = cs.ControlSession(show_overlay=False, watch_for_takeover=False)
    session.acquire()
    try:
        names = [entry.name for entry in KillSwitch().registered()]
        assert cs.KILL_SWITCH_NAME in names
    finally:
        session.release("test")


def test_releasing_control_deregisters() -> None:
    """Otherwise every finished session leaves a corpse for arc-kill to report."""
    from arc.audit.killswitch import KillSwitch

    session = cs.ControlSession(show_overlay=False, watch_for_takeover=False)
    session.acquire()
    session.release("test")
    assert [e.name for e in KillSwitch().registered()] == []


def test_the_abort_phrase_is_the_kill_command() -> None:
    """The panel tells the user to type the same thing they would type in a shell."""
    from arc.control import overlay

    assert overlay.KILL_PHRASE == "arc-kill"


def test_accent_is_the_documented_blue() -> None:
    from arc.control import overlay

    red, green, blue = overlay.ACCENT
    assert blue > green > red  # unmistakably blue
    assert overlay.PEAK_ALPHA < 1.0  # ambient glow, not a painted frame
