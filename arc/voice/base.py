"""The swappable ear and voice.

Modelled on ``arc/model/base.py``: deliberately narrow, so a second backend stays
cheap. The temptation is to expose everything the speech frameworks can do —
alternative transcriptions, per-word confidence, phoneme timing, SSML. None of that is
required to hold a conversation, so none of it belongs here. If a capability is not
needed to listen and reply, it does not go in this file.

Two capabilities *are* reported rather than assumed, because they genuinely vary and
because getting them wrong is silent:

``on_device``
    Whether recognition happens locally. ARC is local-first (BRIEF §0), so a backend
    that would ship audio to a server must say so and be refused rather than quietly
    used. On this machine only ``en-US`` is on-device.

``is_listening`` / ``is_speaking``
    The session state machine and the UI both need to know, and polling a private
    attribute is how two sources of truth get out of sync.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

#: Called with each partial or final transcript.
ResultHandler = Callable[["Transcript"], None]

#: Called with a 0..1 amplitude roughly 20-90 times a second. Drives the orb.
LevelHandler = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class Transcript:
    """One recognition result.

    ``is_final`` matters more than it looks: partial results arrive continuously while
    you speak and are *replacements*, not additions. A caller that appends partials
    instead of replacing them produces text that stutters and repeats.
    """

    text: str
    is_final: bool


class SpeechRecognizer(ABC):
    """Turns microphone audio into text."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for logs and the audit trail."""

    @property
    @abstractmethod
    def on_device(self) -> bool:
        """Whether recognition runs locally, with no audio leaving the machine."""

    @property
    @abstractmethod
    def is_listening(self) -> bool:
        """Whether the microphone is currently open."""

    @abstractmethod
    def start(self, on_result: ResultHandler, on_level: LevelHandler) -> None:
        """Open the microphone and begin recognising.

        Both handlers are called from a background thread. Implementations must be safe
        to call when already listening — the UI can double-fire a toggle, and raising
        there would be worse than a no-op.
        """

    @abstractmethod
    def stop(self) -> None:
        """Close the microphone. Safe to call when not listening."""

    def set_suspended(self, suspended: bool) -> None:
        """Stop or resume feeding audio to the recogniser without closing the mic.

        Concrete rather than abstract, and a no-op by default, so a backend that does
        not need it stays unchanged. It exists because of physics, not preference: on a
        laptop with speakers there is no echo cancellation, so while ARC is talking the
        microphone hears ARC. Left running, the assistant transcribes its own reply,
        treats it as a new question, and answers itself — measured here as five
        utterances spoken and four self-interruptions from one question.
        """
        return None


class SpeechSynthesizer(ABC):
    """Turns text into speech."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier for logs and the audit trail."""

    @property
    @abstractmethod
    def is_speaking(self) -> bool:
        """Whether audio is currently playing."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Queue ``text`` to be spoken.

        Non-blocking by contract. The agent generates at roughly 14 tokens a second, so
        a blocking call would stall generation behind playback and defeat the entire
        point of speaking sentence by sentence.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop immediately and discard anything queued.

        This is barge-in. It must cut off mid-word rather than finishing the sentence:
        the whole signal of interrupting is that the other party stops at once.
        """
