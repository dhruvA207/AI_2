"""Seeing the screen.

§4.3's order of preference, cheapest first: the accessibility tree where it is
available (structured, exact, free), then OCR for dense text, then a vision model for
genuine visual understanding. Capture always downscales before anything reaches a
model — a Retina screenshot costs ~6,000 tokens untouched.
"""

from arc.vision.accessibility import (
    Element,
    actionable_elements,
    find,
    frontmost_application,
    is_trusted,
    read_tree,
    summarize,
)
from arc.vision.capture import Screenshot, capture, capture_dir, displays
from arc.vision.ocr import TextRegion, find_text, read_image, read_text

__all__ = [
    "Element",
    "Screenshot",
    "TextRegion",
    "actionable_elements",
    "capture",
    "capture_dir",
    "displays",
    "find",
    "find_text",
    "frontmost_application",
    "is_trusted",
    "read_image",
    "read_text",
    "read_tree",
    "summarize",
]
