"""Gemini Live: one bidirectional audio session for the whole conversation.

This replaces the earlier REST approach, which was the wrong API. ``generateContent``
with the ``-tts`` model costs one HTTP request *per sentence* and draws on a per-model
free-tier quota of ten requests a day — a single three-sentence reply spent three of
them. JARVIS never hit that wall because it never used that endpoint: it opens one
WebSocket to ``gemini-2.5-flash-native-audio-preview`` and streams audio both ways for
the life of the conversation. Sessions are metered by time, not by request count.

**The trade this makes is real and worth restating.** Your microphone audio is streamed
to Google while the session is open. Apple's on-device recogniser only ever sent text
off the machine — this sends the audio itself. BRIEF §0 says local-first; Dhruv chose
this deliberately, so ``arc voice status`` and ``arc doctor`` both say plainly that
audio leaves the machine, and every session is audited.

End-of-speech detection is Gemini's, not ours: ``automatic_activity_detection`` with
the same 400 ms silence window JARVIS tuned. The adaptive endpoint detector written for
the Apple path is therefore unused here — the server decides when you have stopped.

Audio format is fixed by the API: 16 kHz mono int16 up, 24 kHz mono int16 down.
"""

from __future__ import annotations

import asyncio
import contextlib
import queue
import threading
from collections.abc import Callable
from typing import Any

from arc.errors import ArcError
from arc.log import get_logger

_log = get_logger(__name__)

#: The model JARVIS uses. Native audio in and out, not a text-to-speech endpoint.
LIVE_MODEL = "models/gemini-2.5-flash-native-audio-preview-12-2025"

#: Fixed by the Live API. Do not change these to match a device — resample instead.
SEND_RATE = 16000
RECEIVE_RATE = 24000
CHANNELS = 1
BLOCK = 1024

#: Same voice JARVIS uses.
DEFAULT_VOICE = "Iapetus"


class LiveSession:
    """A running Gemini Live conversation.

    Owns its own event loop on a background thread. The rest of ARC is synchronous and
    the server is a threaded ``BaseHTTPRequestHandler``, so an asyncio-native API has to
    be fenced off behind something callable from any thread rather than leaking
    ``await`` into the interface layer.
    """

    def __init__(
        self,
        api_key: str,
        *,
        voice: str = DEFAULT_VOICE,
        model: str = LIVE_MODEL,
        system_prompt: str = "",
        silence_ms: int = 400,
        on_transcript: Callable[[str, bool], None] | None = None,
        on_level: Callable[[float], None] | None = None,
        on_state: Callable[[str], None] | None = None,
        audit: Any = None,
    ) -> None:
        if not api_key:
            raise ArcError("no Gemini API key; see config/secrets.yaml")

        self._key = api_key
        self._voice = voice
        self._model = model
        self._system = system_prompt
        self._silence_ms = silence_ms
        self._on_transcript = on_transcript
        self._on_level = on_level
        self._on_state = on_state
        self._audit = audit

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._session: Any = None
        self._task: asyncio.Task[None] | None = None
        self._out_queue: asyncio.Queue[dict[str, Any]] | None = None
        self._audio_out: queue.Queue[bytes | None] = queue.Queue()
        self._in_stream: Any = None
        self._running = threading.Event()
        self._error: str | None = None
        self._speaking = False
        self._muted = True
        #: Transcription accumulators; see ``_handle_transcriptions``.
        self._in_text = ""
        self._out_text = ""

    # ── lifecycle ────────────────────────────────────────────────────────────

    @property
    def is_open(self) -> bool:
        return self._running.is_set()

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    @property
    def muted(self) -> bool:
        return self._muted

    def open(self, timeout: float = 20.0) -> None:
        """Start the session. Blocks until the WebSocket is up or raises."""
        if self._running.is_set():
            return
        ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, args=(ready,), name="arc-live", daemon=True
        )
        self._thread.start()
        ready.wait(timeout)
        error = self._error
        if error:
            self._error = None
            raise ArcError(error)
        if not self._running.is_set():
            raise ArcError("Gemini Live session did not start within the timeout")
        self._record("voice.live.open", {"model": self._model, "voice": self._voice})

    def close(self) -> None:
        self._running.clear()
        loop = self._loop
        task = self._task
        # Cancel the task, do not stop the loop. Stopping it mid-await raises
        # "Event loop stopped before Future completed" and skips every cleanup path,
        # which leaves the WebSocket and the audio streams open.
        if loop is not None and task is not None and loop.is_running():
            loop.call_soon_threadsafe(task.cancel)
        # Release the microphone before waiting on threads: PortAudio will otherwise
        # keep the device open and the interpreter will not exit.
        stream = self._in_stream
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()
            self._in_stream = None
        self._audio_out.put(None)  # wake the player so it can exit
        thread = self._thread
        if thread is not None:
            thread.join(timeout=4.0)
        self._thread = None
        self._loop = None
        self._speaking = False
        self._record("voice.live.close", {})

    def set_muted(self, muted: bool) -> None:
        """Gate the microphone without tearing the session down.

        Muting stops audio being *sent*, so nothing is streamed to Google while muted —
        the session stays open only so unmuting is instant rather than a fresh
        handshake.
        """
        self._muted = muted
        self._record("voice.live.mute" if muted else "voice.live.unmute", {})

    # ── the asyncio side ─────────────────────────────────────────────────────

    def _run(self, ready: threading.Event) -> None:
        self._error = None

        async def runner() -> None:
            self._task = asyncio.current_task()
            await self._main(ready)

        try:
            asyncio.run(runner())
        except asyncio.CancelledError:
            pass  # ordinary shutdown
        except Exception as exc:  # pragma: no cover - surfaced through _error
            self._error = f"{type(exc).__name__}: {exc}"
            _log.exception("live session failed")
            ready.set()
        finally:
            self._running.clear()
            ready.set()

    async def _main(self, ready: threading.Event) -> None:
        from google import genai
        from google.genai import types

        self._loop = asyncio.get_running_loop()
        client = genai.Client(api_key=self._key, http_options={"api_version": "v1beta"})

        config = types.LiveConnectConfig(
            response_modalities=[types.Modality.AUDIO],
            # Thinking off: this is conversation, and a thinking budget shows up as
            # dead air before every reply.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection(
                    start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_HIGH,
                    end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_HIGH,
                    prefix_padding_ms=200,
                    silence_duration_ms=self._silence_ms,
                )
            ),
            # Both directions, or the UI has no subtitles: audio-only responses carry
            # no text, so without these `response.text` is always empty and the screen
            # stays blank while ARC talks.
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self._voice)
                )
            ),
        )
        if self._system:
            config.system_instruction = self._system

        async with client.aio.live.connect(model=self._model, config=config) as session:
            self._session = session
            self._out_queue = asyncio.Queue(maxsize=64)
            self._running.set()
            player = threading.Thread(target=self._play_thread, name="arc-live-play", daemon=True)
            player.start()
            ready.set()
            _log.info("live session open", extra={"model": self._model, "voice": self._voice})

            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(self._capture())
                tasks.create_task(self._send())
                tasks.create_task(self._receive())

    async def _capture(self) -> None:
        """Microphone -> out_queue, at the rate the API requires."""
        import math

        import sounddevice as sd

        loop = asyncio.get_running_loop()
        queue = self._out_queue
        assert queue is not None

        def callback(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
            data = bytes(indata)
            if self._on_level is not None:
                # Same sub-sampled RMS as the Apple path, so the orb behaves
                # identically whichever backend is running.
                try:
                    import struct

                    count = len(data) // 2
                    if count:
                        step = max(1, count // 64)
                        samples = struct.unpack(f"<{count}h", data[: count * 2])
                        total = sum(samples[i] ** 2 for i in range(0, count, step))
                        n = len(range(0, count, step))
                        rms = math.sqrt(total / n) / 32768 if n else 0.0
                        db = 20 * math.log10(rms + 1e-9)
                        self._on_level(max(0.0, min(1.0, (db + 45.0) / 40.0)))
                except Exception:  # pragma: no cover
                    pass
            # Muted means nothing is transmitted at all, not merely ignored.
            if self._muted:
                return
            with_data = {"data": data, "mime_type": "audio/pcm"}
            # A dropped block beats blocking the audio thread, which would glitch
            # capture for everything on the device.
            with contextlib.suppress(asyncio.QueueFull, RuntimeError):
                loop.call_soon_threadsafe(queue.put_nowait, with_data)

        # Kept on self so close() can stop it directly. Relying on the context manager
        # alone left PortAudio holding the device after the task was cancelled, and the
        # process then refused to exit even though every thread was a daemon.
        stream = sd.RawInputStream(
            samplerate=SEND_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK,
            callback=callback,
        )
        self._in_stream = stream
        stream.start()
        try:
            while self._running.is_set():
                await asyncio.sleep(0.1)
        finally:
            self._in_stream = None
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()

    async def _send(self) -> None:
        queue = self._out_queue
        assert queue is not None
        while self._running.is_set():
            chunk = await queue.get()
            await self._session.send_realtime_input(
                audio={"data": chunk["data"], "mime_type": "audio/pcm;rate=16000"}
            )

    async def _receive(self) -> None:
        while self._running.is_set():
            async for response in self._session.receive():
                if response.data:
                    if not self._speaking:
                        self._speaking = True
                        self._emit_state("SPEAKING")
                    self._audio_out.put(response.data)

                text = getattr(response, "text", None)
                if text and self._on_transcript is not None:
                    self._on_transcript(text, False)

                content = getattr(response, "server_content", None)
                if content is not None:
                    self._handle_transcriptions(content)
                    if getattr(content, "turn_complete", False):
                        if self._out_text and self._on_transcript is not None:
                            self._on_transcript(self._out_text, True)
                        self._in_text = ""
                        self._out_text = ""
                        self._speaking = False
                        self._emit_state("IDLE")
                    if getattr(content, "interrupted", False):
                        # You spoke over ARC; Gemini stops generating, so drop whatever
                        # is already queued or it keeps playing after the interruption.
                        with contextlib.suppress(queue.Empty):
                            while True:
                                self._audio_out.get_nowait()
                        self._in_text = ""
                        self._out_text = ""
                        self._speaking = False
                        self._emit_state("IDLE")

    def _handle_transcriptions(self, content: Any) -> None:
        """Accumulate transcription deltas into whole utterances.

        The Live API sends fragments — ' What', ' is', ' the', ' ca', 'pital' — not
        replacements. The Apple path sent replacements, and the UI is written to swap
        its text on every event, so forwarding raw fragments would show one syllable at
        a time. Accumulating here keeps one contract for both backends: what arrives is
        always the whole utterance so far.
        """
        if self._on_transcript is None:
            return

        block = getattr(content, "input_transcription", None)
        chunk = getattr(block, "text", None) if block is not None else None
        if chunk:
            self._in_text += chunk
            self._on_transcript(self._in_text, False)

        block = getattr(content, "output_transcription", None)
        chunk = getattr(block, "text", None) if block is not None else None
        if chunk:
            # ARC starting to answer means your question is finished; flush it so the
            # UI shows a settled question rather than a half-built one.
            if self._in_text:
                self._on_transcript(self._in_text, True)
                self._in_text = ""
            self._out_text += chunk
            self._on_transcript(self._out_text, False)

    def _play_thread(self) -> None:
        """Play received audio on a plain thread.

        Deliberately not an asyncio task. ``stream.write`` blocks until the device has
        room, and wrapping it in ``asyncio.to_thread`` produced a coroutine that could
        not be cancelled — closing the session hung indefinitely waiting for a write
        that was never going to be interrupted. A dedicated thread with a sentinel is
        both simpler and actually stoppable.
        """
        import sounddevice as sd

        stream = sd.RawOutputStream(
            samplerate=RECEIVE_RATE, channels=CHANNELS, dtype="int16", blocksize=BLOCK
        )
        stream.start()
        try:
            while self._running.is_set():
                try:
                    chunk = self._audio_out.get(timeout=0.1)
                except queue.Empty:
                    continue
                if chunk is None:
                    return
                stream.write(chunk)
        except Exception:  # pragma: no cover - playback must not kill the session
            _log.exception("playback failed")
        finally:
            with contextlib.suppress(Exception):
                stream.stop()
                stream.close()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _emit_state(self, state: str) -> None:
        if self._on_state is not None:
            try:
                self._on_state(state)
            except Exception:  # pragma: no cover
                _log.exception("state handler failed")

    def _record(self, event: str, args: dict[str, Any]) -> None:
        if self._audit is None:
            return
        try:
            self._audit.record(event, args=args)
        except Exception:  # pragma: no cover
            _log.exception("audit failed")
