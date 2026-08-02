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


def a_shot(
    width: int = 1400,
    source_width: int = 2940,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    screen_size: tuple[float, float] = (0.0, 0.0),
) -> Screenshot:
    return Screenshot(
        path=__import__("pathlib").Path("/tmp/x.png"),
        width=width,
        height=910,
        source_width=source_width,
        source_height=1912,
        origin_x=origin[0],
        origin_y=origin[1],
        screen_width=screen_size[0],
        screen_height=screen_size[1],
    )


#: The built-in display: 1470x956 points, captured at 2x into 2940x1912 pixels, then
#: downscaled to 1400 wide.
def a_retina_shot() -> Screenshot:
    return a_shot(screen_size=(1470.0, 956.0))


def test_scale_reflects_downscaling() -> None:
    assert a_shot().scale == pytest.approx(2.1)


def test_retina_coordinates_land_in_points_not_pixels() -> None:
    """The backing scale factor is a second conversion stacked on the downscale.

    A 1470-point display captures to 2940 pixels, so multiplying by the pixel scale
    alone puts every click at twice the intended offset — off the display entirely on
    the right and bottom halves of the screen.
    """
    shot = a_retina_shot()
    assert shot.point_scale == pytest.approx(1.05)
    assert shot.to_screen(700, 455) == pytest.approx((735.0, 477.75))


def test_capture_maps_to_the_full_extent_of_its_display() -> None:
    """The far corner of the image must be the far corner of the screen."""
    shot = a_retina_shot()
    assert shot.to_screen(shot.width, shot.height) == pytest.approx((1470.0, 955.5))


def test_secondary_display_coordinates_include_its_origin() -> None:
    """Without the origin, a hit on the external monitor is clicked on the laptop."""
    shot = a_shot(source_width=1920, screen_size=(1920.0, 1080.0), origin=(1470.0, 0.0))
    x, _y = shot.to_screen(shot.width / 2, 0)
    assert x == pytest.approx(2430.0)


def test_region_capture_is_its_own_origin() -> None:
    shot = a_shot(source_width=1400, screen_size=(1400.0, 910.0), origin=(300.0, 120.0))
    assert shot.to_screen(0, 0) == (300.0, 120.0)


def test_unscaled_capture_maps_one_to_one() -> None:
    shot = a_shot(width=1400, source_width=1400)
    assert shot.scale == 1.0
    assert shot.to_screen(700, 400) == (700.0, 400.0)


def test_unknown_screen_size_falls_back_to_the_pixel_scale() -> None:
    """Geometry can be unavailable; that must not make coordinates worse than before."""
    assert a_shot().point_scale == a_shot().scale


def test_zero_width_does_not_divide_by_zero() -> None:
    assert a_shot(width=0).scale == 1.0
    assert a_shot(width=0, screen_size=(1470.0, 956.0)).point_scale == 1.0


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


def a_text_area(value: str) -> Element:
    return Element(role="AXTextArea", label="shell", value=value, frame=(0.0, 0.0, 900.0, 600.0))


def test_a_document_body_is_not_a_click_target() -> None:
    """Searching a terminal's scrollback for "File" used to return the text area.

    The match was real and the coordinates were real, but they pointed at the middle of
    the scrollback rather than at anything named "File" — a confident wrong answer,
    which is the worst kind for an agent about to click.
    """
    root = Element(role="AXWindow", children=[a_text_area("some log output\n" * 400)])
    assert find(root, "File") == []


def test_a_short_value_still_identifies_a_field() -> None:
    """A field's contents are a legitimate way to find it; only documents are excluded."""
    root = Element(role="AXWindow", children=[a_text_area("draft@example.com")])
    assert len(find(root, "example.com")) == 1


def test_label_matches_outrank_value_matches() -> None:
    root = Element(
        role="AXWindow",
        children=[a_text_area("Save"), button("Save")],
    )
    assert find(root, "Save")[0].role == "AXButton"


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
