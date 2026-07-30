"""Seeing the screen.

§4.3's order of preference, cheapest first: the accessibility tree where it is
available (structured, exact, free), then OCR for dense text, then a vision model for
genuine visual understanding. Capture always downscales before anything reaches a
model — a Retina screenshot costs ~6,000 tokens untouched.
"""

from arc.vision.capture import Screenshot, capture, capture_dir, displays

__all__ = ["Screenshot", "capture", "capture_dir", "displays"]
