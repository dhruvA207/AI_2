"""Tests for the tool orbs: the side orbs that appear while ARC uses a tool.

These cover the wiring that was missing rather than the drawing. The orbs looked
finished for a while because they could be driven from the console — but nothing ever
emitted a tool event, so they never once appeared on their own.
"""

from __future__ import annotations

import math
from pathlib import Path

from arc.agent.loop import Agent, Step
from arc.tools import registry

WEBUI = Path("arc/interface/webui")


# ── The agent has to announce a tool before running it ──────────────────────────


def test_agent_announces_a_tool_before_running_it() -> None:
    """``on_step`` fires once a tool has returned, which is too late to show work in
    progress: a web fetch is several seconds of apparent nothing."""
    source = Path("arc/agent/loop.py").read_text(encoding="utf-8")
    start = source.index("self._on_tool_start(step)")
    execute = source.index("self._executor.execute(")
    assert start < execute, "on_tool_start must fire before the tool executes"


def test_on_tool_start_is_optional() -> None:
    """Every existing caller constructs an Agent without it."""
    import inspect

    signature = inspect.signature(Agent.__init__)
    assert signature.parameters["on_tool_start"].default is None


def test_step_carries_what_an_orb_needs() -> None:
    step = Step(number=1, thought="", tool="read_file", arguments={"path": "x"})
    assert step.tool and step.arguments and step.number


# ── Categories drive the orb colour ─────────────────────────────────────────────


def test_every_real_tool_has_a_category() -> None:
    for tool in registry.describe():
        assert tool["category"], f"{tool['name']} has no category"


def test_category_of_never_raises_on_an_unknown_tool() -> None:
    """It only colours a UI element; a renamed tool must not take a request down."""
    assert registry.category_of("no_such_tool_at_all") == "general"


def test_categories_all_have_a_colour() -> None:
    """A category with no CSS variable renders as the fallback grey, so every tool in
    that category becomes indistinguishable."""
    css = (WEBUI / "app.css").read_text(encoding="utf-8")
    for category in {t["category"] for t in registry.describe()}:
        assert f"--cat-{category}:" in css, f"category {category} has no colour"


# ── Tool events reach every open UI ─────────────────────────────────────────────


def test_tool_events_are_broadcast_not_just_returned() -> None:
    """A task can be started from the CLI or another window and is still ARC working.

    Emitting only on the requesting response meant the app showed nothing while a
    curl client saw every event — a status display that lies.
    """
    source = Path("arc/interface/server.py").read_text(encoding="utf-8")
    block = source[
        source.index("def _handle_task_stream") : source.index("def _handle_memory_search")
    ]
    assert "self.runtime.listeners" in block, "tool events are not broadcast"
    assert "tool_start" in block and "tool_end" in block


def test_ui_listens_for_tool_events_on_the_long_lived_stream() -> None:
    app = (WEBUI / "app.js").read_text(encoding="utf-8")
    events = app[app.index("function openEvents") : app.index("openEvents();")]
    assert "tool_start" in events and "tool_end" in events


# ── Layout: around the centre, not stacked ──────────────────────────────────────


def _angles(count: int) -> list[float]:
    """The golden-angle placement used by orb.js."""
    return sorted(
        (math.degrees(i * 2.39996 - math.pi / 2) + 360) % 360 for i in range(1, count + 1)
    )


def test_orbs_surround_the_centre() -> None:
    """Four or more tools must reach every quadrant rather than stacking on one side."""
    for count in (4, 6, 8):
        quadrants = {int(a // 90) for a in _angles(count)}
        assert len(quadrants) == 4, f"{count} orbs only covered {len(quadrants)} quadrants"


def test_orbs_spread_out_as_more_appear() -> None:
    gaps = []
    for count in (2, 4, 8):
        angles = _angles(count)
        gaps.append(max((angles[(i + 1) % count] - angles[i]) % 360 for i in range(count)))
    assert gaps[0] > gaps[1] > gaps[2], "orbs should crowd less as more are added"


def test_position_comes_from_the_id_not_the_index() -> None:
    """Angles keyed to list position make every remaining orb jump across the screen
    when one finishes and the list re-indexes."""
    orb = (WEBUI / "orb.js").read_text(encoding="utf-8")
    block = orb[orb.index("const tools = st.activeTools()") :]
    assert "const i = tool.id" in block
    assert "tools.forEach((tool, i)" not in block, "index-based placement reintroduced"
