"""Gemini text-to-speech.

**This is the one part of ARC that leaves the machine.** Everything else — the model,
memory, recognition — runs locally by design (BRIEF §0). Sending text to Google to be
spoken breaks that, and it is deliberate and temporary: Dhruv asked for it because the
built-in macOS voices do not sound smooth enough. It is opt-in via ``voice.engine``,
the audit log records every request, and ``arc voice status`` says plainly that speech
is going to Google.

Written against the REST endpoint with ``urllib`` rather than the ``google-genai`` SDK.
The SDK is Apache-2.0 itself but pulls ``httpx`` and ``websockets`` (BSD-3-Clause) and
``typing-extensions`` (PSF-2.0), which fails the Apache-2.0/MIT rule in §0.1. The whole
call is one POST and one base64 field, so the SDK buys nothing that justifies twelve
transitive packages — and §7 is explicit that dependencies are a liability.

Audio comes back as raw 24 kHz mono 16-bit PCM. It is wrapped in a WAV header with the
stdlib ``wave`` module and played through ``afplay``, which is already on every Mac and
— unlike an in-process player — can be killed instantly for barge-in.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import tempfile
import threading
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Any

from arc.errors import ArcError
from arc.log import get_logger
from arc.voice.base import SpeechSynthesizer

_log = get_logger(__name__)

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

#: Gemini's prebuilt voices. Listed so ``arc voice status`` can show the alternatives
#: without a network call; Google adds to this occasionally.
VOICES = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)

#: Gemini TTS returns signed 16-bit little-endian mono at this rate.
_SAMPLE_RATE = 24000
_CHANNELS = 1
_SAMPLE_WIDTH = 2


def load_api_key(config_dir: Path | None = None) -> str | None:
    """Find the Gemini key, without it ever going near the general config.

    Deliberately not read through :class:`~arc.config.Config`. ``Config.load`` merges
    every ``*.yaml`` in the directory, so a key placed in ``config/secrets.yaml`` would
    end up inside the same object that ``--json`` output and log lines are built from.
    Kept separate, it can only be read by the code that needs it.

    Order: ``GEMINI_API_KEY`` in the environment, then ``config/secrets.yaml``, then
    ``~/.arc/secrets.yaml``.
    """
    from arc.paths import arc_home
    from arc.paths import config_dir as default_config_dir

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if key:
        return key

    candidates = [
        (config_dir or default_config_dir()) / "secrets.yaml",
        arc_home() / "secrets.yaml",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            _log.warning("could not parse %s", path)
            continue
        if isinstance(data, dict):
            value = str(data.get("gemini_api_key", "") or "").strip()
            if value:
                return value
    return None


def _to_wav(pcm: bytes) -> bytes:
    """Wrap raw PCM in a WAV container so a normal player can handle it."""
    import io

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(_CHANNELS)
        handle.setsampwidth(_SAMPLE_WIDTH)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(pcm)
    return buffer.getvalue()


class GeminiSynthesizer(SpeechSynthesizer):
    """Speech output via Gemini's TTS models. Sends text to Google."""

    def __init__(
        self,
        api_key: str,
        *,
        voice: str = "Charon",
        model: str = "gemini-2.5-flash-preview-tts",
        timeout: float = 30.0,
        audit: Any = None,
    ) -> None:
        if not api_key:
            raise ArcError(
                "no Gemini API key. Set GEMINI_API_KEY, or put `gemini_api_key: ...` in "
                "config/secrets.yaml (already gitignored)."
            )
        self._key = api_key
        self._voice = voice
        self._model = model
        self._timeout = timeout
        self._audit = audit

        # One worker so sentences play in the order they were generated. Without it,
        # two sentences synthesised concurrently would race and play over each other.
        self._pending: list[str] = []  # text waiting to be synthesised
        self._ready: list[bytes] = []  # audio waiting to be played
        self._lock = threading.Lock()
        self._synth_thread: threading.Thread | None = None
        self._play_thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self._synthesizing = False
        self._stopped = threading.Event()
        self._wake = threading.Event()

    @property
    def name(self) -> str:
        return f"gemini:{self._voice}"

    @property
    def is_speaking(self) -> bool:
        """True from the moment text is queued until the last audio has played.

        The in-flight synthesis window matters and is easy to miss: between the worker
        popping a sentence and ``afplay`` starting there is a second or two of network
        call during which the queue is empty and no process exists. Reporting False
        there tells :class:`~arc.voice.session.VoiceSession` that ARC has finished
        talking, so it re-opens the microphone just in time for the audio to start —
        and the self-hearing loop is back.
        """
        with self._lock:
            if self._pending or self._ready or self._synthesizing:
                return True
            process = self._process
        return process is not None and process.poll() is None

    def speak(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        with self._lock:
            self._stopped.clear()
            self._pending.append(text)
            self._ensure_workers()

    def stop(self) -> None:
        with self._lock:
            self._pending.clear()
            self._ready.clear()
            self._synthesizing = False
            process = self._process
        self._stopped.set()
        self._wake.set()
        if process is not None and process.poll() is None:
            process.kill()

    def _ensure_workers(self) -> None:
        """Start the synthesise and play threads. Caller holds the lock."""
        if self._synth_thread is None or not self._synth_thread.is_alive():
            self._synth_thread = threading.Thread(
                target=self._synth_loop, name="arc-tts-net", daemon=True
            )
            self._synth_thread.start()
        if self._play_thread is None or not self._play_thread.is_alive():
            self._play_thread = threading.Thread(
                target=self._play_loop, name="arc-tts-play", daemon=True
            )
            self._play_thread.start()

    def _synth_loop(self) -> None:
        """Fetch audio ahead of playback.

        Split from playback on purpose. Gemini takes roughly a second of network per
        second of speech, so synthesising and playing in lockstep would double every
        gap between sentences. Running them in parallel means only the *first* sentence
        pays the network cost; each later one is already fetched by the time the
        previous finishes.
        """
        while not self._stopped.is_set():
            with self._lock:
                if not self._pending:
                    self._synthesizing = False
                    return
                text = self._pending.pop(0)
                self._synthesizing = True
            try:
                wav = _to_wav(self._synthesize(text))
            except ArcError as exc:
                _log.error("gemini tts failed: %s", exc)
                self._record("voice.tts.error", {"error": str(exc)})
                continue
            finally:
                with self._lock:
                    self._synthesizing = bool(self._pending)
            if self._stopped.is_set():
                return
            with self._lock:
                self._ready.append(wav)
            self._wake.set()

    def _play_loop(self) -> None:
        while not self._stopped.is_set():
            with self._lock:
                wav = self._ready.pop(0) if self._ready else None
                more_coming = bool(self._pending) or self._synthesizing
            if wav is None:
                if not more_coming:
                    return
                self._wake.wait(0.1)
                self._wake.clear()
                continue
            self._play(wav)

    def _synthesize(self, text: str) -> bytes:
        """One POST, returning raw PCM. Raises ArcError with something actionable."""
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}}
                },
            },
        }
        request = urllib.request.Request(
            _ENDPOINT.format(model=self._model),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                # Header rather than ?key=: a query string ends up in logs and proxy
                # history, and this is a credential.
                "x-goog-api-key": self._key,
            },
            method="POST",
        )

        self._record("voice.tts.request", {"model": self._model, "voice": self._voice})
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            # Generous, because Google puts the useful part last: a 429 body leads with
            # boilerplate and only names the exhausted quota and the retry delay near
            # the end. Truncating at 300 characters hid exactly the bit worth reading.
            detail = exc.read().decode("utf-8", "replace")[:1200]
            hint = ""
            if exc.code == 429:
                hint = (
                    " — the API key's quota is exhausted, so ARC cannot speak until it "
                    "resets or billing is enabled. Recognition is unaffected."
                )
            raise ArcError(f"Gemini TTS returned {exc.code}{hint}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ArcError(
                f"could not reach Gemini ({exc.reason}). This engine needs the network; "
                f"set voice.engine to 'apple' to speak locally."
            ) from exc

        try:
            part = body["candidates"][0]["content"]["parts"][0]
            return base64.b64decode(part["inlineData"]["data"])
        except (KeyError, IndexError, ValueError) as exc:
            raise ArcError(f"unexpected Gemini TTS response: {json.dumps(body)[:300]}") from exc

    def _play(self, wav: bytes) -> None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
            handle.write(wav)
            path = handle.name
        try:
            process = subprocess.Popen(
                ["/usr/bin/afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self._lock:
                self._process = process
            process.wait()
        except Exception:  # pragma: no cover - playback must not kill the worker
            _log.exception("playback failed")
        finally:
            with self._lock:
                self._process = None
            Path(path).unlink(missing_ok=True)

    def _record(self, event: str, args: dict[str, Any]) -> None:
        """Audit the network call. The key is never included."""
        if self._audit is None:
            return
        try:
            self._audit.record(event, args=args)
        except Exception:  # pragma: no cover
            _log.exception("audit failed")
