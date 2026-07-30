"""Tests for screen capture, the accessibility tree, and OCR.

Coordinate mapping gets the most attention: a vision model reports a button in the
*downscaled* image, and clicking that coordinate without scaling lands somewhere else
on screen entirely. That is the bug class most likely to make screen control look
subtly broken rather than obviously broken.
"""

from __future__ import annotations

import pytest

from arc.vision.accessibility import Element, actionable_elements, find, summarize
from arc.vision.capture import Screenshot
from arc.vision.ocr import TextRegion

# ── Coordinate mapping ──────────────────────────────────────────────────────────


def a_shot(width: int = 1400, source_width: int = 2940) -> Screenshot:
    return Screenshot(
        path=__import__("pathlib").Path("/tmp/x.png"),
        width=width,
        height=910,
        source_width=source_width,
        source_height=1912,
    )


def test_scale_reflects_downscaling() -> None:
    assert a_shot().scale == pytest.approx(2.1)


def test_coordinates_map_back_to_screen() -> None:
    """The whole point of recording the scale factor."""
    assert a_shot().to_screen(700, 400) == pytest.approx((1470.0, 840.0))


def test_unscaled_capture_maps_one_to_one() -> None:
    shot = a_shot(width=1400, source_width=1400)
    assert shot.scale == 1.0
    assert shot.to_screen(700, 400) == (700.0, 400.0)


def test_zero_width_does_not_divide_by_zero() -> None:
    assert a_shot(width=0).scale == 1.0


# ── Accessibility elements ──────────────────────────────────────────────────────


def button(label: str, frame=(100.0, 200.0, 80.0, 40.0), enabled: bool = True) -> Element:
    return Element(role="AXButton", label=label, frame=frame, enabled=enabled)


def test_center_is_the_middle_of_the_frame() -> None:
    assert button("Save").center == (140.0, 220.0)


def test_element_without_a_frame_has_no_center() -> None:
    assert Element(role="AXButton", label="Ghost").center is None


def test_actionable_requires_role_frame_and_enabled() -> None:
    assert button("Save").actionable
    assert not button("Save", enabled=False).actionable
    assert not button("Save", frame=None).actionable
    assert not Element(role="AXStaticText", label="Hi", frame=(0.0, 0.0, 10.0, 10.0)).actionable


def test_containers_are_not_actionable() -> None:
    """Otherwise the list is dominated by layout nodes an agent cannot use."""
    assert not Element(role="AXGroup", frame=(0.0, 0.0, 100.0, 100.0)).actionable


def test_describe_includes_clickable_coordinates() -> None:
    described = button("Save").describe()
    assert "Save" in described
    assert "(140, 220)" in described


def test_describe_marks_disabled_elements() -> None:
    assert "[disabled]" in button("Save", enabled=False).describe()


def test_flatten_walks_the_whole_tree() -> None:
    root = Element(role="AXWindow", children=[button("A"), button("B")])
    assert len(root.flatten()) == 3


def test_actionable_elements_filters() -> None:
    root = Element(
        role="AXWindow",
        children=[button("Go"), Element(role="AXStaticText", label="Label")],
    )
    assert [e.label for e in actionable_elements(root)] == ["Go"]


def test_exact_label_matches_rank_first() -> None:
    """ "Save" must not be beaten by "Save As...", or the agent clicks the wrong thing."""
    root = Element(role="AXWindow", children=[button("Save As..."), button("Save")])
    assert find(root, "Save")[0].label == "Save"


def test_find_is_case_insensitive() -> None:
    root = Element(role="AXWindow", children=[button("Submit")])
    assert find(root, "submit")


def test_find_on_empty_query() -> None:
    assert find(Element(role="AXWindow"), "  ") == []


def test_summarize_reports_when_nothing_is_actionable() -> None:
    assert "no actionable elements" in summarize(Element(role="AXWindow"))


def test_summarize_lists_elements() -> None:
    root = Element(role="AXWindow", children=[button("Go")])
    assert "Go" in summarize(root)


# ── OCR regions ─────────────────────────────────────────────────────────────────


def test_text_region_center() -> None:
    region = TextRegion(text="Save", confidence=0.9, box=(100.0, 200.0, 60.0, 20.0))
    assert region.center == (130.0, 210.0)


def test_text_region_serializes() -> None:
    region = TextRegion(text="Save", confidence=0.912345, box=(1.0, 2.0, 3.0, 4.0))
    payload = region.to_dict()
    assert payload["text"] == "Save"
    assert payload["confidence"] == 0.912
