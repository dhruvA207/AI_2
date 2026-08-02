"""The voice session: mute state, sentence-chunked speech, and barge-in.

Rests muted. ARC has unrestricted access to this machine (BRIEF §0.3), so the
microphone is never open at startup and every session that does open it is written to
the audit log — one of the three safeguards asked for in place of permission prompts.

The reason this file exists rather than the server calling the backends directly is
``feed()``. The model generates at roughly 14 tokens a second, so a forty-token reply
takes about three seconds. Waiting for it before speaking means three seconds of
silence after every question, which does not read as conversation. Feeding tokens in
and speaking each sentence as it completes puts the first audio out in well under a
second, and the rest arrives while ARC is already talking.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable

from arc.log import get_logger
from arc.voice.base import SpeechRecognizer, SpeechSynthesizer, Transcript

_log = get_logger(__name__)

#: A sentence ends at ., ! or ? followed by whitespace or the end of the string.
#: Abbreviations will occasionally split early; a clause spoken slightly short is a far
#: smaller cost than holding audio back waiting for a boundary that may never arrive.
_SENTENCE = re.compile(r"(.+?[.!?])(\s+|$)", re.DOTALL)

#: Speak anyway once the buffer passes this, so a reply with no punctuation at all
#: still produces audio instead of silently accumulating.
_MAX_CHUNK = 220


class VoiceSession:
    """Ties the recogniser and synthesiser to ARC's conversation loop."""

    def __init__(
        self,
        recognizer: SpeechRecognizer,
        synthesizer: SpeechSynthesizer,
        *,
        audit: object | None = None,
        on_transcript: Callable[[Transcript], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        barge_in: bool = True,
        resume_delay: float = 0.35,
    ) -> None:
        self.recognizer = recognizer
        self.synthesizer = synthesizer
        self._audit = audit
        self._on_transcript = on_transcript
        self._on_level = on_level
        self._barge_in = barge_in

        self._resume_delay = resume_delay
        self._lock = threading.Lock()
        self._buffer = ""
        self._spoke_any = False
        self._input_suspended = False
        self._resume_timer: threading.Timer | None = None

    # ── microphone ───────────────────────────────────────────────────────────

    @property
    def listening(self) -> bool:
        return self.recognizer.is_listening

    @property
    def speaking(self) -> bool:
        return self.synthesizer.is_speaking

    def start_listening(self) -> None:
        """Open the microphone. Safe to call when already open."""
        if self.recognizer.is_listening:
            return
        self._record("voice.listen.start", {"recognizer": self.recognizer.name})
        self.recognizer.start(self._handle_result, self._handle_level)

    def stop_listening(self) -> None:
        """Close the microphone. Safe to call when already closed."""
        if not self.recognizer.is_listening:
            return
        self.recognizer.stop()
        self._record("voice.listen.stop", {"recognizer": self.recognizer.name})

    def toggle(self) -> bool:
        """Flip the microphone and return the new state. This is what ⌘S calls."""
        if self.recognizer.is_listening:
            self.stop_listening()
        else:
            self.start_listening()
        return self.recognizer.is_listening

    # ── speech out ───────────────────────────────────────────────────────────

    def feed(self, text: str) -> None:
        """Add generated text, speaking each sentence as it completes."""
        with self._lock:
            self._buffer += text
            self._drain(final=False)

    def flush(self) -> None:
        """Speak whatever is left once generation has finished."""
        with self._lock:
            self._drain(final=True)

    def _drain(self, *, final: bool) -> None:
        while self._buffer:
            match = _SENTENCE.match(self._buffer)
            if match:
                sentence = match.group(1).strip()
                self._buffer = self._buffer[match.end() :]
            elif final or len(self._buffer) >= _MAX_CHUNK:
                sentence = self._buffer.strip()
                self._buffer = ""
            else:
                return

            if sentence:
                if not self._spoke_any:
                    self._record("voice.speak", {"synthesizer": self.synthesizer.name})
                    self._spoke_any = True
                # Deafen the microphone before any audio plays. On a laptop there is no
                # echo cancellation, so an open mic hears the speakers: ARC transcribes
                # its own reply, treats it as a new question, and answers itself.
                self._suspend_input()
                self.synthesizer.speak(sentence)

    def interrupt(self) -> None:
        """Barge-in: stop talking immediately and drop anything unspoken."""
        with self._lock:
            self._buffer = ""
        if self.synthesizer.is_speaking:
            self.synthesizer.stop()
            self._record("voice.interrupt", {})
        # Whether or not anything was playing, listening must come back — otherwise an
        # interrupt leaves the microphone deaf and ARC simply stops responding.
        if self._resume_timer is not None:
            self._resume_timer.cancel()
        self._resume_timer = threading.Timer(self._resume_delay, self._resume_input)
        self._resume_timer.daemon = True
        self._resume_timer.start()

    def reset(self) -> None:
        """Clear per-turn state without touching the microphone."""
        with self._lock:
            self._buffer = ""
            self._spoke_any = False

    def close(self) -> None:
        if self._resume_timer is not None:
            self._resume_timer.cancel()
            self._resume_timer = None
        self.stop_listening()
        self.interrupt()

    # ── internals ────────────────────────────────────────────────────────────

    def _suspend_input(self) -> None:
        """Deafen the microphone for the duration of ARC's own speech."""
        if self._input_suspended:
            return
        self._input_suspended = True
        self.recognizer.set_suspended(True)
        if self._resume_timer is not None:
            self._resume_timer.cancel()
        self._watch_for_end()

    def _watch_for_end(self) -> None:
        """Re-open the microphone once playback has finished.

        Polled rather than driven by a delegate callback: ``speakUtterance_`` queues,
        so several sentences can be outstanding and only ``isSpeaking`` knows when the
        last one has actually finished. The tail delay covers the speaker ringing out,
        which would otherwise be caught as the first word of your next sentence.
        """
        if self.synthesizer.is_speaking:
            self._resume_timer = threading.Timer(0.15, self._watch_for_end)
        else:
            self._resume_timer = threading.Timer(self._resume_delay, self._resume_input)
        self._resume_timer.daemon = True
        self._resume_timer.start()

    def _resume_input(self) -> None:
        if self.synthesizer.is_speaking:  # more was queued while we waited
            self._watch_for_end()
            return
        self._input_suspended = False
        self.recognizer.set_suspended(False)

    def _handle_result(self, transcript: Transcript) -> None:
        # Belt and braces: input is already gated while ARC speaks, so anything
        # arriving here during playback is echo that slipped through and must not be
        # mistaken for you interrupting.
        if self._input_suspended:
            return
        if self._barge_in and transcript.text.strip() and self.synthesizer.is_speaking:
            self.interrupt()
        if self._on_transcript is not None:
            self._on_transcript(transcript)

    def _handle_level(self, level: float) -> None:
        if self._on_level is not None:
            self._on_level(level)

    def _record(self, event: str, args: dict[str, object]) -> None:
        """Write to the audit log if one was supplied.

        Every microphone session is audited. Optional only so tests and the
        non-server paths can construct a session without one.
        """
        if self._audit is None:
            return
        try:
            self._audit.record(event, args=args)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - auditing must never break audio
            _log.exception("audit failed")
