"""Tests for the control indicator's abort phrase.

The glow and the panel need a window server, so what is tested here is the part that
decides *when* ARC gets killed. That logic has to be exact in both directions: it must
fire when the user types the phrase, and it must never fire by accident, because a
false positive SIGKILLs the agent mid-task.
"""

from __future__ import annotations

from arc.control.overlay import (
    KILL_PHRASE,
    TYPING_RESET_SECONDS,
    _advance,
    consume,
)


def a_state() -> dict[str, object]:
    return {"typed": "", "stopping": False, "last_key": 0.0}


def type_phrase(state: dict[str, object], text: str, at: float = 1.0) -> bool:
    """Type ``text`` one character at a time, returning whether the abort fired."""
    fired = False
    for offset, character in enumerate(text):
        fired = consume(state, character, at + offset * 0.05) or fired
    return fired


# ── The phrase itself ───────────────────────────────────────────────────────────


def test_typing_the_phrase_fires() -> None:
    assert type_phrase(a_state(), KILL_PHRASE)


def test_the_phrase_is_the_command_name() -> None:
    """One thing to remember, whether you reach for the terminal or the panel."""
    assert KILL_PHRASE == "arc-kill"


def test_case_does_not_matter() -> None:
    assert type_phrase(a_state(), "ARC-KILL")


def test_a_prefix_alone_does_not_fire() -> None:
    state = a_state()
    assert not type_phrase(state, "arc-kil")
    assert state["typed"] == "arc-kil"


def test_unrelated_typing_does_not_fire() -> None:
    assert not type_phrase(a_state(), "the quick brown fox jumps over the lazy dog")


def test_a_wrong_character_resets_progress() -> None:
    state = a_state()
    assert not type_phrase(state, "arc-kix")
    assert state["typed"] == ""


def test_a_fumbled_start_still_gets_there() -> None:
    """Restarting on mismatch rather than clearing outright; "arc" then "arc-kill"."""
    assert type_phrase(a_state(), "arcarc-kill")


def test_progress_is_reported_for_display() -> None:
    """The panel lights up the matched prefix, so it has to be exposed."""
    state = a_state()
    type_phrase(state, "arc-")
    assert state["typed"] == "arc-"


# ── Not firing by accident ──────────────────────────────────────────────────────


def test_a_stale_half_typed_phrase_is_abandoned() -> None:
    """Stray keystrokes minutes apart must not accumulate into an abort."""
    state = a_state()
    consume(state, "arc-kil", 100.0)
    assert not consume(state, "l", 100.0 + TYPING_RESET_SECONDS + 1.0)
    assert state["typed"] == ""


def test_typing_within_the_window_is_not_abandoned() -> None:
    state = a_state()
    consume(state, "arc-kil", 100.0)
    assert consume(state, "l", 100.0 + TYPING_RESET_SECONDS - 1.0)


def test_firing_twice_is_not_possible() -> None:
    """The kill switch should be fired once, not once per subsequent keystroke."""
    state = a_state()
    assert type_phrase(state, KILL_PHRASE)
    assert not type_phrase(state, KILL_PHRASE, at=50.0)


def test_a_pasted_burst_still_fires() -> None:
    """One event can carry several characters."""
    assert consume(a_state(), KILL_PHRASE, 1.0)


# ── The prefix machine ──────────────────────────────────────────────────────────


def test_advance_builds_the_prefix() -> None:
    assert _advance("arc", "-") == "arc-"


def test_advance_restarts_on_the_first_letter() -> None:
    assert _advance("arc-k", "a") == "a"


def test_advance_clears_on_an_unrelated_character() -> None:
    assert _advance("arc", "z") == ""
