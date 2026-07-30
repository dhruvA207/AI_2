"""Local OCR via macOS's Vision framework.

§4.3 puts OCR between the accessibility tree and a vision model: the tree is exact but
only covers what an application chooses to expose, and plenty does not — canvas-drawn
UI, rendered documents, screenshots, video. OCR reads those without paying for a
multimodal model.

Vision is the right engine here rather than Tesseract: it ships with macOS, runs on the
Neural Engine, needs no model download, and is markedly more accurate on UI text. It is
also entirely local, which §3 requires.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from arc.errors import PlatformError
from arc.log import get_logger

_log = get_logger(__name__)

#: Discard results below this confidence. Vision reports a lot of near-random text for
#: icons and textures, and feeding that to a model is worse than reporting less.
MIN_CONFIDENCE = 0.35


@dataclass(frozen=True, slots=True)
class TextRegion:
    """One recognised piece of text and where it sits."""

    text: str
    confidence: float
    #: (x, y, width, height) in *image pixel* coordinates, origin top-left. Vision
    #: reports normalised bottom-left coordinates; converted here so callers do not
    #: each have to remember which convention they are in.
    box: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        """Middle of the region, for clicking."""
        x, y, width, height = self.box
        return (x + width / 2, y + height / 2)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""
        return {
            "text": self.text,
            "confidence": round(self.confidence, 3),
            "box": [round(v, 1) for v in self.box],
            "center": [round(v, 1) for v in self.center],
        }


def read_image(
    path: Path, *, min_confidence: float = MIN_CONFIDENCE, fast: bool = False
) -> list[TextRegion]:
    """Recognise text in an image file.

    Args:
        path: Image to read.
        min_confidence: Drop results below this.
        fast: Trade accuracy for speed. Accurate is the default because a misread
            button label sends the agent somewhere wrong, which costs far more than
            the extra tens of milliseconds.
    """
    try:
        import Quartz
        import Vision
        from Foundation import NSURL
    except ImportError as exc:  # pragma: no cover - non-macOS
        raise PlatformError(
            "OCR needs pyobjc-framework-Vision. Install with: pip install 'arc[vision]'"
        ) from exc

    if not path.is_file():
        raise PlatformError(f"no such image: {path}")

    url = NSURL.fileURLWithPath_(str(path))
    source = Quartz.CGImageSourceCreateWithURL(url, None)
    if source is None:
        raise PlatformError(f"could not read image: {path}")
    image = Quartz.CGImageSourceCreateImageAtIndex(source, 0, None)
    if image is None:
        raise PlatformError(f"could not decode image: {path}")

    width = Quartz.CGImageGetWidth(image)
    height = Quartz.CGImageGetHeight(image)

    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(
        Vision.VNRequestTextRecognitionLevelFast
        if fast
        else Vision.VNRequestTextRecognitionLevelAccurate
    )
    request.setUsesLanguageCorrection_(True)

    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(image, None)
    success, error = handler.performRequests_error_([request], None)
    if not success:
        raise PlatformError(f"OCR failed: {error}")

    regions: list[TextRegion] = []
    for observation in request.results() or []:
        candidates = observation.topCandidates_(1)
        if not candidates:
            continue
        candidate = candidates[0]
        confidence = float(candidate.confidence())
        if confidence < min_confidence:
            continue

        # Vision uses normalised coordinates with the origin bottom-left; images and
        # screen coordinates are top-left, so y is flipped here rather than in every
        # caller.
        box = observation.boundingBox()
        regions.append(
            TextRegion(
                text=str(candidate.string()),
                confidence=confidence,
                box=(
                    float(box.origin.x) * width,
                    (1.0 - float(box.origin.y) - float(box.size.height)) * height,
                    float(box.size.width) * width,
                    float(box.size.height) * height,
                ),
            )
        )

    _log.info("ocr complete", extra={"path": str(path), "regions": len(regions)})
    return regions


def read_text(path: Path, **kwargs: Any) -> str:
    """Return an image's text as plain reading-order text."""
    regions = read_image(path, **kwargs)
    # Top to bottom, then left to right — approximate reading order, which is close
    # enough for UI and far better than Vision's detection order.
    regions = sorted(regions, key=lambda r: (round(r.box[1] / 12), r.box[0]))
    return "\n".join(region.text for region in regions)


def find_text(path: Path, needle: str, **kwargs: Any) -> list[TextRegion]:
    """Find where a string appears in an image."""
    lowered = needle.lower().strip()
    return [r for r in read_image(path, **kwargs) if lowered in r.text.lower()]
