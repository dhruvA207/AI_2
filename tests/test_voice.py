"""Tests for speech in and out.

No microphone is opened and no Apple framework is imported. What is covered here is
the logic that is easy to get wrong and silent when it is: sentence chunking,
barge-in, the local-first refusal, and the promise that ``arc`` still imports on a
machine with no speech support at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from arc.config import Config
from arc.voice.base import SpeechRecognizer, SpeechSynthesizer, Transcript
from arc.voice.session import VoiceSession


class FakeRecognizer(SpeechRecognizer):
    def __init__(self) -> None:
        self._listening = False
        self.on_result: Any = None

    @property
    def name(self) -> str:
        return "fake"

    @property
    def on_device(self) -> bool:
        return True

    @property
    def is_listening(self) -> bool:
        return self._listening

    def start(self, on_result: Any, on_level: Any) -> None:
        self._listening = True
        self.on_result = on_result

    def stop(self) -> None:
        self._listening = False


class FakeSynthesizer(SpeechSynthesizer):
    def __init__(self) -> None:
        self.spoken: list[str] = []
        self.stops = 0
        self._speaking = False

    @property
    def name(self) -> str:
        return "fake"

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        self._speaking = True

    def stop(self) -> None:
        self.stops += 1
        self._speaking = False


@pytest.fixture
def session() -> VoiceSession:
    return VoiceSession(FakeRecognizer(), FakeSynthesizer())


# ── Resting state ───────────────────────────────────────────────────────────────


def test_rests_muted(session: VoiceSession) -> None:
    """ARC has unrestricted access to this machine (§0.3), so the microphone is never
    open until something explicitly opens it."""
    assert session.listening is False


def test_toggle_opens_and_closes(session: VoiceSession) -> None:
    assert session.toggle() is True
    assert session.toggle() is False


def test_start_and_stop_are_idempotent(session: VoiceSession) -> None:
    """The UI can double-fire a toggle; raising there would be worse than a no-op."""
    session.start_listening()
    session.start_listening()
    assert session.listening
    session.stop_listening()
    session.stop_listening()
    assert not session.listening


# ── Sentence chunking ───────────────────────────────────────────────────────────


def test_speaks_each_sentence_as_it_completes(session: VoiceSession) -> None:
    """The whole point of feed(): at ~14 tok/s, waiting for the full reply is seconds
    of silence after every question."""
    synth: Any = session.synthesizer
    session.feed("The capital of France is Paris.")
    assert synth.spoken == ["The capital of France is Paris."]

    session.feed(" It has about two million")
    assert len(synth.spoken) == 1  # incomplete — nothing spoken yet
    session.feed(" people.")
    assert synth.spoken[-1] == "It has about two million people."


def test_partial_sentence_is_spoken_on_flush(session: VoiceSession) -> None:
    synth: Any = session.synthesizer
    session.feed("no punctuation here")
    assert synth.spoken == []
    session.flush()
    assert synth.spoken == ["no punctuation here"]


def test_long_run_without_punctuation_still_speaks(session: VoiceSession) -> None:
    """A reply with no sentence end must not accumulate silently forever."""
    synth: Any = session.synthesizer
    session.feed("word " * 60)
    assert synth.spoken, "nothing was spoken despite passing the chunk limit"


def test_question_and_exclamation_end_sentences(session: VoiceSession) -> None:
    synth: Any = session.synthesizer
    session.feed("Really? Yes! Done.")
    assert synth.spoken == ["Really?", "Yes!", "Done."]


# ── Barge-in ────────────────────────────────────────────────────────────────────


def test_microphone_is_deafened_while_arc_speaks(session: VoiceSession) -> None:
    """The echo fix, and the reason voice barge-in is not possible on speakers.

    Without echo cancellation an open microphone hears the speakers. Left running, ARC
    transcribes its own reply, treats it as a new question and answers itself — five
    utterances and four self-interruptions came out of a single question before this.
    So anything recognised during playback is echo by definition and is dropped.
    """
    heard: list[str] = []
    recognizer = FakeRecognizer()
    synth = FakeSynthesizer()
    talker = VoiceSession(recognizer, synth, on_transcript=lambda t: heard.append(t.text))
    talker.start_listening()
    talker.feed("A long answer.")
    assert synth.is_speaking

    recognizer.on_result(Transcript(text="A long answer", is_final=True))  # type: ignore[misc]
    assert heard == [], "ARC heard its own voice and would have answered itself"
    assert synth.stops == 0


def test_explicit_interrupt_stops_playback(session: VoiceSession) -> None:
    """Barge-in is explicit — Escape or the mic toggle — precisely because voice
    cannot be trusted to signal it while the speakers are playing."""
    synth: Any = session.synthesizer
    session.start_listening()
    session.feed("A long answer.")
    assert synth.is_speaking
    session.interrupt()
    assert synth.stops == 1
    assert not synth.is_speaking


def test_input_is_restored_after_speaking(session: VoiceSession) -> None:
    """A suspend that never lifts leaves the mic permanently deaf."""
    recognizer: Any = session.recognizer
    session.start_listening()
    session.feed("Done.")
    synth: Any = session.synthesizer
    synth._speaking = False  # playback finished
    session._resume_input()
    assert session._input_suspended is False
    heard: list[str] = []
    session._on_transcript = heard.append  # type: ignore[assignment]
    recognizer.on_result(Transcript(text="next question", is_final=True))
    assert heard, "microphone stayed deaf after ARC finished speaking"


def test_interrupt_drops_unspoken_text(session: VoiceSession) -> None:
    synth: Any = session.synthesizer
    session.feed("Spoken.")
    session.feed(" not yet finished")
    session.interrupt()
    session.flush()
    assert synth.spoken == ["Spoken."]


def test_interrupt_always_reopens_the_microphone(session: VoiceSession) -> None:
    """Interrupting while nothing is playing must not leave input suspended."""
    session.start_listening()
    session.feed("Talking.")
    session.interrupt()
    synth: Any = session.synthesizer
    synth._speaking = False
    session._resume_input()
    assert session._input_suspended is False


# ── Auditing ────────────────────────────────────────────────────────────────────


class RecordingAudit:
    def __init__(self) -> None:
        self.events: list[str] = []

    def record(self, event: str, **_: Any) -> None:
        self.events.append(event)


def test_every_microphone_session_is_audited() -> None:
    """One of the three safeguards asked for in place of permission prompts (§0.3)."""
    audit = RecordingAudit()
    session = VoiceSession(FakeRecognizer(), FakeSynthesizer(), audit=audit)
    session.start_listening()
    session.stop_listening()
    assert "voice.listen.start" in audit.events
    assert "voice.listen.stop" in audit.events


def test_a_failing_audit_never_breaks_audio() -> None:
    class Broken:
        def record(self, *_: Any, **__: Any) -> None:
            raise RuntimeError("disk full")

    session = VoiceSession(FakeRecognizer(), FakeSynthesizer(), audit=Broken())
    session.start_listening()
    assert session.listening


# ── Local-first ─────────────────────────────────────────────────────────────────


def test_config_declares_on_device_by_default() -> None:
    """Only en-US has an on-device asset on this machine; anything else would send
    audio to Apple. The default must not quietly allow that."""
    config = Config.load(use_env=False)
    section = config.section("voice")
    assert section["require_on_device"] is True
    assert section["locale"] == "en-US"


def test_every_voice_config_key_is_read() -> None:
    """ADR-022: twelve keys were once declared and never read, with the real values
    hardcoded. A key that does nothing is worse than a missing one."""
    source = Path("arc/voice/__init__.py").read_text(encoding="utf-8")
    session_source = Path("arc/interface/server.py").read_text(encoding="utf-8")
    declared = set(Config.load(use_env=False).section("voice"))
    for key in declared:
        assert f'"{key}"' in source or f'"voice.{key}"' in session_source, (
            f"config key voice.{key} is declared but never read"
        )


def test_core_imports_without_apple_frameworks() -> None:
    """``arc.voice`` must import on a machine with no speech support — the abstraction
    only earns its keep if that is verified rather than assumed."""
    import arc.voice

    assert hasattr(arc.voice, "build")
    assert hasattr(arc.voice, "available")


# ── Gemini engine ───────────────────────────────────────────────────────────────


def test_gemini_requires_a_key() -> None:
    from arc.errors import ArcError
    from arc.voice.gemini import GeminiSynthesizer

    with pytest.raises(ArcError, match="no Gemini API key"):
        GeminiSynthesizer("")


def test_key_is_never_read_through_general_config() -> None:
    """``Config.load`` merges every YAML in the directory, so a secret read through it
    would land in the same object ``--json`` output and log lines are built from."""
    source = Path("arc/voice/gemini.py").read_text(encoding="utf-8")
    assert "from arc.config import" not in source
    assert "Config.load(" not in source

    config = Config.load(use_env=False)
    assert config.get("gemini_api_key") is None
    assert "gemini_api_key" not in config.section("voice")


def test_key_travels_in_a_header_not_the_url() -> None:
    """A key in a query string ends up in server logs and proxy history.

    Checked by building the real request rather than grepping the file, so a later
    refactor that reintroduces ``?key=`` is caught even if the comment survives.
    """
    from arc.voice.gemini import GeminiSynthesizer

    synth = GeminiSynthesizer("SECRET-KEY-VALUE")
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

        def read(self) -> bytes:
            import base64
            import json

            blob = base64.b64encode(b"\x00\x01" * 100).decode()
            return json.dumps(
                {"candidates": [{"content": {"parts": [{"inlineData": {"data": blob}}]}}]}
            ).encode()

    def fake_urlopen(request: Any, **_: Any) -> Any:
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.header_items()}
        return FakeResponse()

    import arc.voice.gemini as mod

    original = mod.urllib.request.urlopen
    mod.urllib.request.urlopen = fake_urlopen  # type: ignore[assignment]
    try:
        synth._synthesize("hello")
    finally:
        mod.urllib.request.urlopen = original  # type: ignore[assignment]

    assert "SECRET-KEY-VALUE" not in captured["url"]
    assert captured["headers"].get("x-goog-api-key") == "SECRET-KEY-VALUE"


def test_secrets_file_is_gitignored() -> None:
    ignored = Path(".gitignore").read_text(encoding="utf-8")
    assert "config/secrets.yaml" in ignored


def test_gemini_is_the_only_engine() -> None:
    """The macOS synthesiser was removed, not demoted to a fallback.

    A fallback would silently swap ARC's voice for the one that was rejected for not
    sounding good enough, which is worse than saying the voice is unavailable.
    """
    assert not Path("arc/voice/macos.py").read_text(encoding="utf-8").count("AVSpeech")
    from arc.voice import macos

    assert not hasattr(macos, "AppleSynthesizer")


def test_missing_key_fails_loudly_rather_than_silently_switching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from arc.errors import ArcError
    from arc.voice import _build_synthesizer

    monkeypatch.setattr("arc.voice.gemini.load_api_key", lambda *a, **k: None)
    with pytest.raises(ArcError, match="only speech engine"):
        _build_synthesizer({"voice": "Iapetus"})


def test_default_voice_matches_jarvis() -> None:
    section = Config.load(use_env=False).section("voice")
    assert section["voice"] == "Iapetus"


def test_wav_header_matches_gemini_pcm_format() -> None:
    """Gemini returns headerless 24 kHz mono 16-bit PCM; a wrong header plays as noise
    or at the wrong pitch, which is easy to mistake for a bad voice."""
    import io
    import wave

    from arc.voice.gemini import _to_wav

    with wave.open(io.BytesIO(_to_wav(b"\x00\x01" * 2400)), "rb") as handle:
        assert handle.getframerate() == 24000
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2


# ── Voice mode ──────────────────────────────────────────────────────────────────


def test_mode_is_declared_and_read() -> None:
    """ADR-022: a config key that is declared but never read is worse than a missing
    one, because it fails silently instead of loudly."""
    assert "mode" in Config.load(use_env=False).section("voice")
    source = Path("arc/interface/server.py").read_text(encoding="utf-8")
    assert 'self.config.get("voice.mode"' in source


def test_live_mode_tells_the_ui_it_answers_itself() -> None:
    """In live mode Gemini replies out loud on its own. If the UI also posted the
    transcript to /chat/stream, a second local reply would talk over the first."""
    source = Path("arc/interface/server.py").read_text(encoding="utf-8")
    assert '"answers_itself"' in source

    app = Path("arc/interface/webui/app.js").read_text(encoding="utf-8")
    assert "answersItself" in app
    assert "!answersItself" in app, "the UI must skip send() when Gemini answers itself"


def test_live_adapter_matches_the_session_surface() -> None:
    """/voice/status and the ⌘S toggle must not care which mode is running."""
    from arc.interface.server import _LiveAdapter
    from arc.voice.session import VoiceSession

    shared = {
        "listening",
        "speaking",
        "toggle",
        "start_listening",
        "stop_listening",
        "interrupt",
        "reset",
        "feed",
        "flush",
        "close",
        "recognizer",
        "synthesizer",
    }
    for name in shared:
        assert hasattr(_LiveAdapter, name) or name in ("recognizer", "synthesizer"), name
        assert hasattr(VoiceSession, name) or name in ("recognizer", "synthesizer"), name


def test_endpoint_wait_is_not_the_old_slow_default() -> None:
    """900 ms was a fifth of the entire response time spent confirming you had
    stopped talking."""
    assert int(Config.load(use_env=False).section("voice")["silence_ms"]) <= 500
