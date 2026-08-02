"""Speech in and out.

Imports nothing platform-specific at module level: ``arc`` must keep importing on a
machine with no Apple frameworks, which is asserted in the test suite rather than
assumed. Use :func:`build` to get a session; it raises a clear ``ArcError`` when the
optional extra is missing instead of an ImportError from three layers down.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from arc.errors import ArcError
from arc.log import get_logger
from arc.voice.base import (
    LevelHandler,
    ResultHandler,
    SpeechRecognizer,
    SpeechSynthesizer,
    Transcript,
)
from arc.voice.session import VoiceSession

if TYPE_CHECKING:
    from arc.config import Config

_log = get_logger(__name__)

__all__ = [
    "ArcError",
    "LevelHandler",
    "ResultHandler",
    "SpeechRecognizer",
    "SpeechSynthesizer",
    "Transcript",
    "VoiceSession",
    "available",
    "build",
    "pump",
]


def pump(stop: Any, interval: float = 0.1) -> None:
    """Run the main run loop until ``stop`` is set. Must be called on the main thread.

    Recognition results are delivered on the main queue, so a host that never pumps
    gets levels but no transcripts. See :func:`arc.voice.macos.pump`.
    """
    from arc.voice.macos import pump as _pump

    _pump(stop, interval)


def available() -> bool:
    """Whether speech support can run on this machine."""
    from arc.platform import platform_name

    if platform_name() != "macos":
        return False
    from arc.voice import macos

    return macos.available()


def _build_synthesizer(section: dict[str, Any], audit: Any = None) -> SpeechSynthesizer:
    """Build the Gemini voice.

    Gemini is the only speech engine. The macOS ``AVSpeechSynthesizer`` backend was
    removed rather than kept as a fallback: its voices did not sound good enough to
    use, and a fallback that silently swaps ARC's voice for one Dhruv rejected is
    worse than an error saying the voice is unavailable.

    That does mean **speech output requires the network and a key**. Recognition is
    unaffected and still runs entirely on this machine.
    """
    from arc.voice.gemini import GeminiSynthesizer, load_api_key

    key = load_api_key()
    if not key:
        raise ArcError(
            "no Gemini API key, and Gemini is the only speech engine. Set "
            "GEMINI_API_KEY, or add `gemini_api_key: ...` to config/secrets.yaml "
            "(gitignored). Recognition still works without it — only speech output "
            "needs a key."
        )
    return GeminiSynthesizer(
        key,
        voice=str(section.get("voice", "Iapetus")),
        model=str(section.get("model", "gemini-2.5-flash-preview-tts")),
        audit=audit,
    )


def build(config: Config, **kwargs: Any) -> VoiceSession:
    """Construct a voice session from config.

    Every key read here is a key that actually changes behaviour — ADR-022 exists
    because twelve config keys were once declared and never read, with the real values
    hardcoded in the modules.
    """
    from arc.platform import platform_name

    if platform_name() != "macos":
        raise ArcError("voice support is macOS-only (ADR-021: ARC runs on this Mac)")

    from arc.voice.macos import AppleRecognizer

    section = config.section("voice")
    recognizer = AppleRecognizer(
        locale=str(section.get("locale", "en-US")),
        require_on_device=bool(section.get("require_on_device", True)),
        silence_ms=int(section.get("silence_ms", 900)),
        speech_margin=float(section.get("speech_margin", 0.12)),
    )
    synthesizer = _build_synthesizer(section, kwargs.get("audit"))
    return VoiceSession(
        recognizer,
        synthesizer,
        barge_in=bool(section.get("barge_in", True)),
        **kwargs,
    )
